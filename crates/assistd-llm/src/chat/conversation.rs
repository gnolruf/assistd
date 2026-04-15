//! Multi-turn conversation state owned by `LlamaChatClient`.
//!
//! The daemon holds a single `Conversation` behind a `tokio::sync::Mutex`;
//! every query from every client contributes a turn to the same ongoing
//! dialogue. Token budgeting is best-effort, driven by a bytes-per-token
//! heuristic that intentionally over-counts multi-byte text so we summarize
//! early rather than overflow the server's context window.

use assistd_tools::Attachment;
use async_trait::async_trait;
use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use tracing::{debug, warn};

use super::config::ChatSpec;
use super::error::ChatClientError;
use super::wire;

const SUMMARY_PREFIX: &str = "[Conversation summary] ";
const TOKENS_PER_MESSAGE_OVERHEAD: u32 = 4;
/// Conservative per-image token weight for budget math. Real usage
/// depends on the vision model, but 1000 tokens errs on the side of
/// summarizing earlier rather than overflowing.
const TOKENS_PER_IMAGE: u32 = 1000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    pub fn as_wire(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: String,
    /// Image attachments carried by this turn. Empty for every classic
    /// text-only message; populated by `push_user_with_attachments` when
    /// a tool-call (typically `see`) produced an image the model should
    /// see on its next turn.
    pub attachments: Vec<Attachment>,
}

/// Summarizer trait so unit tests can inject a fake without spinning up
/// an HTTP server. `LlamaChatClient` implements this trait for itself.
#[async_trait]
pub trait Summarizer: Send + Sync {
    async fn summarize(
        &self,
        dialogue: String,
        target_tokens: u32,
        max_tokens: u32,
    ) -> Result<String, ChatClientError>;
}

/// Mutable conversation state.
///
/// Layout invariants:
/// - `system_prompt` is injected at the head of `as_wire_messages()` only when
///   non-empty.
/// - `messages` never contains a pre-baked system-prompt message; it holds
///   user/assistant turns plus at most one synthetic summary message (role
///   `System`, content prefixed with `SUMMARY_PREFIX`) that sits at index 0.
#[derive(Debug)]
pub struct Conversation {
    system_prompt: String,
    messages: Vec<Message>,
}

impl Conversation {
    pub fn new(system_prompt: String) -> Self {
        Self {
            system_prompt,
            messages: Vec::new(),
        }
    }

    pub fn push_user(&mut self, content: String) {
        self.messages.push(Message {
            role: Role::User,
            content,
            attachments: Vec::new(),
        });
    }

    /// Append a user turn that carries one or more attachments alongside
    /// its text. When rendered to the wire, the message becomes a
    /// multimodal `content` array with one `text` part followed by one
    /// `image_url` part per attachment.
    pub fn push_user_with_attachments(&mut self, content: String, attachments: Vec<Attachment>) {
        self.messages.push(Message {
            role: Role::User,
            content,
            attachments,
        });
    }

    pub fn push_assistant(&mut self, content: String) {
        self.messages.push(Message {
            role: Role::Assistant,
            content,
            attachments: Vec::new(),
        });
    }

    /// Drop the most recent message if and only if it's a user message.
    /// Called after a pre-emit HTTP failure to keep history consistent with
    /// what the model actually saw.
    pub fn rollback_last_user(&mut self) {
        if matches!(self.messages.last().map(|m| m.role), Some(Role::User)) {
            self.messages.pop();
        }
    }

    pub fn approx_total_tokens(&self) -> u32 {
        let mut total = 0u32;
        if !self.system_prompt.is_empty() {
            total = total.saturating_add(
                TOKENS_PER_MESSAGE_OVERHEAD.saturating_add(approx_tokens(&self.system_prompt)),
            );
        }
        for m in &self.messages {
            total = total.saturating_add(approx_message_tokens(m));
        }
        total
    }

    /// Render current state as wire messages. Drops the system prompt if
    /// the user explicitly configured it empty. Messages with attachments
    /// are rendered as a multimodal `content` array (text part + one
    /// `image_url` part per attachment); text-only messages stay as plain
    /// strings for compatibility with non-vision models.
    pub fn as_wire_messages(&self) -> Vec<wire::ChatMessage<'_>> {
        let mut out = Vec::with_capacity(self.messages.len() + 1);
        if !self.system_prompt.is_empty() {
            out.push(wire::ChatMessage {
                role: Role::System.as_wire(),
                content: wire::ContentBody::Text(&self.system_prompt),
            });
        }
        for m in &self.messages {
            let content = if m.attachments.is_empty() {
                wire::ContentBody::Text(&m.content)
            } else {
                let mut parts = Vec::with_capacity(m.attachments.len() + 1);
                parts.push(wire::ContentPart::Text { text: &m.content });
                for att in &m.attachments {
                    parts.push(attachment_to_part(att));
                }
                wire::ContentBody::Parts(parts)
            };
            out.push(wire::ChatMessage {
                role: m.role.as_wire(),
                content,
            });
        }
        out
    }

    /// Ensure total approximate tokens stay under the configured budget.
    /// Triggers a summarization call if needed; on summarizer failure the
    /// caller is expected to fall back to `truncate_to_budget`.
    pub async fn ensure_budget(
        &mut self,
        summarizer: &dyn Summarizer,
        spec: &ChatSpec,
    ) -> Result<(), ChatClientError> {
        let budget = effective_budget(spec);
        if self.approx_total_tokens() <= budget {
            return Ok(());
        }

        let preserve_pairs = spec.preserve_recent_turns.max(1) as usize;
        let preserve_from = self.first_preserved_index(preserve_pairs);

        let tail_start = self.summary_insertion_index();
        if tail_start >= preserve_from {
            debug!(
                target: "assistd::chat",
                "ensure_budget: nothing to summarize, falling through to truncation"
            );
            self.truncate_to_budget(spec);
            return Ok(());
        }

        let dialogue = serialize_tail(&self.messages[tail_start..preserve_from]);
        if dialogue.trim().is_empty() {
            self.truncate_to_budget(spec);
            return Ok(());
        }

        let summary = summarizer
            .summarize(
                dialogue,
                spec.summary_target_tokens,
                spec.max_summary_tokens,
            )
            .await?;
        let trimmed = summary.trim();
        if trimmed.is_empty() {
            return Err(ChatClientError::Summarize(
                "summarizer returned empty text".into(),
            ));
        }

        let max_summary_bytes = (spec.summary_target_tokens as usize).saturating_mul(4);
        let body = if trimmed.len() > max_summary_bytes {
            truncate_utf8(trimmed, max_summary_bytes)
        } else {
            trimmed.to_string()
        };

        let summary_msg = Message {
            role: Role::System,
            content: format!("{SUMMARY_PREFIX}{body}"),
            attachments: Vec::new(),
        };

        let drop_end = preserve_from;
        self.messages.drain(tail_start..drop_end);
        self.messages.insert(tail_start, summary_msg);

        if self.approx_total_tokens() > budget {
            debug!(
                target: "assistd::chat",
                "ensure_budget: still over budget after summarize, truncating"
            );
            self.truncate_to_budget(spec);
        }
        Ok(())
    }

    /// Infallible fallback: drop the oldest non-system, non-summary messages
    /// repeatedly until we fit in budget (or we've reduced history to just
    /// the latest user message).
    pub fn truncate_to_budget(&mut self, spec: &ChatSpec) {
        let budget = effective_budget(spec);
        while self.approx_total_tokens() > budget {
            let Some(idx) = self.first_droppable_index() else {
                warn!(
                    target: "assistd::chat",
                    "truncate_to_budget: cannot drop any more messages without losing the latest user turn"
                );
                break;
            };
            self.messages.remove(idx);
        }
    }

    fn summary_insertion_index(&self) -> usize {
        if self
            .messages
            .first()
            .map(|m| m.role == Role::System && m.content.starts_with(SUMMARY_PREFIX))
            .unwrap_or(false)
        {
            1
        } else {
            0
        }
    }

    fn first_preserved_index(&self, preserve_pairs: usize) -> usize {
        let len = self.messages.len();
        let start = self.summary_insertion_index();
        let mut pairs_seen = 0usize;
        let mut idx = len;
        while idx > start && pairs_seen < preserve_pairs {
            let prev = idx - 1;
            match self.messages[prev].role {
                Role::User => {
                    pairs_seen += 1;
                    idx = prev;
                }
                Role::Assistant => {
                    if prev > start {
                        idx = prev - 1;
                    } else {
                        idx = prev;
                    }
                    pairs_seen += 1;
                }
                Role::System => {
                    idx = prev;
                }
            }
        }
        idx.max(start)
    }

    fn first_droppable_index(&self) -> Option<usize> {
        let start = self.summary_insertion_index();
        if start >= self.messages.len() {
            return None;
        }
        let last_user = self
            .messages
            .iter()
            .rposition(|m| m.role == Role::User)
            .unwrap_or(self.messages.len());
        if start == last_user {
            None
        } else {
            Some(start)
        }
    }
}

fn approx_tokens(text: &str) -> u32 {
    (text.len() as u32).div_ceil(4)
}

fn approx_message_tokens(m: &Message) -> u32 {
    let image_cost = (m.attachments.len() as u32).saturating_mul(TOKENS_PER_IMAGE);
    TOKENS_PER_MESSAGE_OVERHEAD
        .saturating_add(approx_tokens(&m.content))
        .saturating_add(image_cost)
}

fn attachment_to_part(att: &Attachment) -> wire::ContentPart<'_> {
    match att {
        Attachment::Image { mime, bytes } => wire::ContentPart::ImageUrl {
            image_url: wire::ImageUrl {
                url: format!("data:{};base64,{}", mime, B64.encode(bytes)),
            },
        },
    }
}

fn effective_budget(spec: &ChatSpec) -> u32 {
    spec.max_history_tokens.min(spec.effective_context_budget())
}

fn serialize_tail(messages: &[Message]) -> String {
    let mut out = String::new();
    for m in messages {
        out.push_str(m.role.as_wire());
        out.push_str(": ");
        out.push_str(&m.content);
        out.push('\n');
    }
    out
}

fn truncate_utf8(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    let mut out = String::with_capacity(end + 1);
    out.push_str(&s[..end]);
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use tokio::sync::Mutex;

    struct FakeSummarizer {
        reply: String,
        calls: Arc<AtomicUsize>,
        captured: Arc<Mutex<Vec<String>>>,
    }

    impl FakeSummarizer {
        fn new(reply: impl Into<String>) -> Self {
            Self {
                reply: reply.into(),
                calls: Arc::new(AtomicUsize::new(0)),
                captured: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    #[async_trait]
    impl Summarizer for FakeSummarizer {
        async fn summarize(
            &self,
            dialogue: String,
            _target_tokens: u32,
            _max_tokens: u32,
        ) -> Result<String, ChatClientError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.captured.lock().await.push(dialogue);
            Ok(self.reply.clone())
        }
    }

    struct FailingSummarizer;

    #[async_trait]
    impl Summarizer for FailingSummarizer {
        async fn summarize(
            &self,
            _dialogue: String,
            _target_tokens: u32,
            _max_tokens: u32,
        ) -> Result<String, ChatClientError> {
            Err(ChatClientError::Summarize("boom".into()))
        }
    }

    fn spec(max_history: u32, preserve: u32, ctx: u32) -> ChatSpec {
        ChatSpec {
            host: "127.0.0.1".into(),
            port: 8385,
            system_prompt: "sys".into(),
            max_history_tokens: max_history,
            summary_target_tokens: max_history / 4,
            preserve_recent_turns: preserve,
            temperature: 0.7,
            max_response_tokens: 512,
            max_summary_tokens: max_history / 2,
            request_timeout_secs: 60,
            model_context_length: ctx,
        }
    }

    #[test]
    fn push_and_rollback_user_message() {
        let mut c = Conversation::new("sys".into());
        c.push_user("hi".into());
        c.push_assistant("hello".into());
        c.rollback_last_user();
        assert_eq!(
            c.messages.len(),
            2,
            "rollback is a no-op when last is assistant"
        );

        c.push_user("again".into());
        c.rollback_last_user();
        assert_eq!(c.messages.len(), 2);
        assert!(matches!(c.messages.last().unwrap().role, Role::Assistant));
    }

    #[test]
    fn as_wire_messages_injects_system_first() {
        let mut c = Conversation::new("sys".into());
        c.push_user("hi".into());
        let wire = c.as_wire_messages();
        assert_eq!(wire.len(), 2);
        assert_eq!(wire[0].role, "system");
        assert_eq!(wire[0].content, wire::ContentBody::Text("sys"));
        assert_eq!(wire[1].role, "user");
    }

    #[test]
    fn as_wire_messages_drops_empty_system_prompt() {
        let mut c = Conversation::new(String::new());
        c.push_user("hi".into());
        let wire = c.as_wire_messages();
        assert_eq!(wire.len(), 1);
        assert_eq!(wire[0].role, "user");
    }

    #[test]
    fn approx_tokens_matches_bytes_over_four() {
        assert_eq!(approx_tokens(""), 0);
        assert_eq!(approx_tokens("abcd"), 1);
        assert_eq!(approx_tokens("abcde"), 2);
        assert_eq!(approx_tokens("hello world"), 3);
    }

    #[test]
    fn push_user_with_attachments_renders_multimodal_wire() {
        let mut c = Conversation::new(String::new());
        c.push_user_with_attachments(
            "what is this?".into(),
            vec![Attachment::Image {
                mime: "image/png".into(),
                bytes: vec![0xAB, 0xCD],
            }],
        );
        let wire = c.as_wire_messages();
        assert_eq!(wire.len(), 1);
        assert_eq!(wire[0].role, "user");
        let parts = match &wire[0].content {
            wire::ContentBody::Parts(p) => p,
            other => panic!("expected Parts, got {other:?}"),
        };
        assert_eq!(parts.len(), 2);
        match &parts[0] {
            wire::ContentPart::Text { text } => assert_eq!(*text, "what is this?"),
            other => panic!("first part must be text, got {other:?}"),
        }
        match &parts[1] {
            wire::ContentPart::ImageUrl { image_url } => {
                assert_eq!(image_url.url, "data:image/png;base64,q80=");
            }
            other => panic!("second part must be image_url, got {other:?}"),
        }
    }

    #[test]
    fn plain_text_user_still_renders_as_string_content() {
        let mut c = Conversation::new(String::new());
        c.push_user("hello".into());
        let wire = c.as_wire_messages();
        assert!(matches!(wire[0].content, wire::ContentBody::Text("hello")));
    }

    #[test]
    fn attachments_contribute_to_token_budget() {
        let mut c = Conversation::new(String::new());
        c.push_user("q".into());
        let baseline = c.approx_total_tokens();

        let mut c = Conversation::new(String::new());
        c.push_user_with_attachments(
            "q".into(),
            vec![Attachment::Image {
                mime: "image/png".into(),
                bytes: vec![0; 10],
            }],
        );
        let with_image = c.approx_total_tokens();
        assert_eq!(with_image, baseline + TOKENS_PER_IMAGE);
    }

    #[test]
    fn multimodal_wire_serializes_to_openai_shape() {
        let mut c = Conversation::new(String::new());
        c.push_user_with_attachments(
            "describe".into(),
            vec![Attachment::Image {
                mime: "image/jpeg".into(),
                bytes: vec![0x12, 0x34, 0x56],
            }],
        );
        let msgs = c.as_wire_messages();
        let json = serde_json::to_value(&msgs).unwrap();
        assert_eq!(json[0]["content"][0]["type"], "text");
        assert_eq!(json[0]["content"][0]["text"], "describe");
        assert_eq!(json[0]["content"][1]["type"], "image_url");
        assert_eq!(
            json[0]["content"][1]["image_url"]["url"],
            "data:image/jpeg;base64,EjRW"
        );
    }

    #[tokio::test]
    async fn ensure_budget_noop_when_under() {
        let mut c = Conversation::new("sys".into());
        c.push_user("hi".into());
        let fake = FakeSummarizer::new("summary");
        c.ensure_budget(&fake, &spec(10_000, 4, 20_000))
            .await
            .unwrap();
        assert_eq!(fake.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn ensure_budget_summarizes_when_over() {
        let mut c = Conversation::new("sys".into());
        for i in 0..10 {
            c.push_user(format!("user turn {i} with some filler text"));
            c.push_assistant(format!(
                "assistant reply {i} with enough text to blow the budget"
            ));
        }
        c.push_user("latest question".into());
        let before_total = c.approx_total_tokens();

        let fake = FakeSummarizer::new("the conversation covered topics 0 through 9");
        c.ensure_budget(&fake, &spec(60, 2, 10_000)).await.unwrap();

        assert_eq!(fake.calls.load(Ordering::SeqCst), 1);
        assert!(c.approx_total_tokens() <= 60 || c.approx_total_tokens() < before_total);

        let first = &c.messages[0];
        assert_eq!(first.role, Role::System);
        assert!(first.content.starts_with(SUMMARY_PREFIX));

        let last = c.messages.last().unwrap();
        assert_eq!(last.role, Role::User);
        assert_eq!(last.content, "latest question");
    }

    #[tokio::test]
    async fn ensure_budget_preserves_recent_turns_around_summary() {
        let mut c = Conversation::new("sys".into());
        for i in 0..6 {
            c.push_user(format!("user {i} message with enough length to count"));
            c.push_assistant(format!("assistant {i} message with enough length to count"));
        }
        c.push_user("current question with enough length to count".into());

        let fake = FakeSummarizer::new("early chat covered topics 0 and 1");
        c.ensure_budget(&fake, &spec(120, 2, 10_000)).await.unwrap();

        let tail_contents: Vec<_> = c
            .messages
            .iter()
            .rev()
            .take(5)
            .map(|m| m.content.clone())
            .collect();
        assert!(tail_contents.iter().any(|c| c.contains("current question")));
    }

    #[tokio::test]
    async fn ensure_budget_summarize_failure_propagates() {
        let mut c = Conversation::new("sys".into());
        for i in 0..20 {
            c.push_user(format!("user turn {i}"));
            c.push_assistant(format!("assistant reply {i}"));
        }
        let result = c
            .ensure_budget(&FailingSummarizer, &spec(40, 2, 10_000))
            .await;
        assert!(matches!(result, Err(ChatClientError::Summarize(_))));
    }

    #[test]
    fn truncate_drops_oldest_first() {
        let mut c = Conversation::new("sys".into());
        for i in 0..8 {
            c.push_user(format!("user message {i} with padding"));
            c.push_assistant(format!("assistant reply {i} with padding"));
        }
        let before = c.messages.len();
        c.truncate_to_budget(&spec(40, 2, 10_000));
        assert!(c.messages.len() < before);
        assert_eq!(c.messages.last().unwrap().role, Role::Assistant);
    }

    #[test]
    fn truncate_preserves_last_user_message() {
        let mut c = Conversation::new("sys".into());
        for i in 0..30 {
            c.push_user(format!("u{i} with filler"));
            c.push_assistant(format!("a{i} with filler"));
        }
        c.push_user("keepme with filler".into());
        c.truncate_to_budget(&spec(20, 1, 10_000));
        assert_eq!(c.messages.last().unwrap().content, "keepme with filler");
    }

    #[test]
    fn truncate_utf8_clamps_on_codepoint_boundary() {
        let s = "世界世界世界";
        let truncated = truncate_utf8(s, 4);
        assert!(truncated.ends_with('…'));
        assert!(truncated.chars().filter(|c| *c != '…').all(|c| c == '世'));
    }
}
