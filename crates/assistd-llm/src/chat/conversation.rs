//! Multi-turn conversation state owned by `LlamaChatClient`.
//!
//! The daemon holds a single `Conversation` behind a `tokio::sync::Mutex`;
//! every query from every client contributes a turn to the same ongoing
//! dialogue. Token budgeting is best-effort, driven by a bytes-per-token
//! heuristic that intentionally over-counts multi-byte text so we summarize
//! early rather than overflow the server's context window.

use assistd_config::{ChatConfig, ModelConfig};
use assistd_tools::Attachment;
use async_trait::async_trait;
use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use tracing::{debug, warn};

use super::error::ChatClientError;
use super::wire;

const SUMMARY_PREFIX: &str = "[Conversation summary] ";
/// Tool-result user messages carry this prefix so (1) the model can
/// distinguish tool output from genuine user speech when history is
/// replayed, (2) the truncator can pair the result with its assistant
/// `tool_calls` predecessor and drop both atomically.
pub const TOOL_RESULT_PREFIX: &str = "[tool:";
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

/// One tool call recorded on an assistant turn. Mirrors the OpenAI shape
/// `{id, type: "function", function: {name, arguments}}`. `arguments` is the
/// JSON-encoded argument string the model emitted, stored verbatim so the
/// replayed wire payload preserves whatever whitespace/formatting the model
/// used — some servers compare it against their own re-serialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallRecord {
    pub id: String,
    pub name: String,
    pub arguments: String,
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
    /// Non-empty only on assistant messages that requested tool calls.
    /// When present, the outgoing wire message is rendered with
    /// `content: null` (or omitted) and a `tool_calls` array.
    pub tool_calls: Vec<ToolCallRecord>,
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
            tool_calls: Vec::new(),
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
            tool_calls: Vec::new(),
        });
    }

    pub fn push_assistant(&mut self, content: String) {
        self.messages.push(Message {
            role: Role::Assistant,
            content,
            attachments: Vec::new(),
            tool_calls: Vec::new(),
        });
    }

    /// Append an assistant turn that requested tool calls. `content` is the
    /// assistant's narration (typically empty — models usually emit tool
    /// calls without accompanying text when `finish_reason: "tool_calls"`).
    /// `calls` must be non-empty; on the wire the message will render with
    /// `content` omitted and `tool_calls: [...]` populated.
    pub fn push_assistant_with_tool_calls(
        &mut self,
        content: Option<String>,
        calls: Vec<ToolCallRecord>,
    ) {
        debug_assert!(
            !calls.is_empty(),
            "push_assistant_with_tool_calls requires at least one call"
        );
        self.messages.push(Message {
            role: Role::Assistant,
            content: content.unwrap_or_default(),
            attachments: Vec::new(),
            tool_calls: calls,
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
    /// strings for compatibility with non-vision models. Assistant messages
    /// carrying `tool_calls` render with `content: null` (omitted) and a
    /// populated `tool_calls` array.
    pub fn as_wire_messages(&self) -> Vec<wire::ChatMessage<'_>> {
        let mut out = Vec::with_capacity(self.messages.len() + 1);
        if !self.system_prompt.is_empty() {
            out.push(wire::ChatMessage {
                role: Role::System.as_wire(),
                content: Some(wire::ContentBody::Text(&self.system_prompt)),
                tool_calls: None,
                tool_call_id: None,
            });
        }
        for m in &self.messages {
            if !m.tool_calls.is_empty() {
                // Assistant-with-tool_calls: omit content entirely (the
                // OpenAI spec allows null/absent; some templates require
                // absent). Any accompanying narration is dropped at
                // commit time, so `m.content` is usually empty here.
                let specs: Vec<wire::ToolCallSpec<'_>> = m
                    .tool_calls
                    .iter()
                    .map(|c| wire::ToolCallSpec {
                        id: &c.id,
                        kind: "function",
                        function: wire::FunctionCallSpec {
                            name: &c.name,
                            arguments: &c.arguments,
                        },
                    })
                    .collect();
                out.push(wire::ChatMessage {
                    role: m.role.as_wire(),
                    content: None,
                    tool_calls: Some(specs),
                    tool_call_id: None,
                });
                continue;
            }
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
                content: Some(content),
                tool_calls: None,
                tool_call_id: None,
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
        chat: &ChatConfig,
        model: &ModelConfig,
    ) -> Result<(), ChatClientError> {
        let budget = effective_budget(chat, model);
        if self.approx_total_tokens() <= budget {
            return Ok(());
        }

        let preserve_pairs = chat.preserve_recent_turns.max(1) as usize;
        let preserve_from = self.first_preserved_index(preserve_pairs);

        let tail_start = self.summary_insertion_index();
        if tail_start >= preserve_from {
            debug!(
                target: "assistd::chat",
                "ensure_budget: nothing to summarize, falling through to truncation"
            );
            self.truncate_to_budget(chat, model);
            return Ok(());
        }

        let dialogue = serialize_tail(&self.messages[tail_start..preserve_from]);
        if dialogue.trim().is_empty() {
            self.truncate_to_budget(chat, model);
            return Ok(());
        }

        let summary = summarizer
            .summarize(
                dialogue,
                chat.summary_target_tokens,
                chat.max_summary_tokens,
            )
            .await?;
        let trimmed = summary.trim();
        if trimmed.is_empty() {
            return Err(ChatClientError::Summarize(
                "summarizer returned empty text".into(),
            ));
        }

        let max_summary_bytes = (chat.summary_target_tokens as usize).saturating_mul(4);
        let body = if trimmed.len() > max_summary_bytes {
            truncate_utf8(trimmed, max_summary_bytes)
        } else {
            trimmed.to_string()
        };

        let summary_msg = Message {
            role: Role::System,
            content: format!("{SUMMARY_PREFIX}{body}"),
            attachments: Vec::new(),
            tool_calls: Vec::new(),
        };

        let drop_end = preserve_from;
        self.messages.drain(tail_start..drop_end);
        self.messages.insert(tail_start, summary_msg);

        if self.approx_total_tokens() > budget {
            debug!(
                target: "assistd::chat",
                "ensure_budget: still over budget after summarize, truncating"
            );
            self.truncate_to_budget(chat, model);
        }
        Ok(())
    }

    /// Infallible fallback: drop the oldest non-system, non-summary messages
    /// repeatedly until we fit in budget (or we've reduced history to just
    /// the latest user message). Tool-call/result pairs are dropped
    /// atomically so the wire payload never carries an assistant
    /// `tool_calls` without a matching result (or vice versa) — most
    /// server-side chat templates reject that.
    pub fn truncate_to_budget(&mut self, chat: &ChatConfig, model: &ModelConfig) {
        let budget = effective_budget(chat, model);
        while self.approx_total_tokens() > budget {
            let Some(idx) = self.first_droppable_index() else {
                warn!(
                    target: "assistd::chat",
                    "truncate_to_budget: cannot drop any more messages without losing the latest user turn"
                );
                break;
            };
            self.drop_with_pair(idx);
        }
    }

    /// Remove `idx` and, if it's half of a tool-call/result pair, the
    /// matching sibling. Handles two shapes:
    /// 1. Assistant-with-tool_calls at `idx` → also drop the immediately
    ///    following tool-result user message (if present).
    /// 2. Tool-result user message at `idx` → also drop the immediately
    ///    preceding assistant-with-tool_calls (if present). This direction
    ///    only fires if callers hit it directly; `first_droppable_index`
    ///    always returns the assistant half first.
    fn drop_with_pair(&mut self, idx: usize) {
        if idx >= self.messages.len() {
            return;
        }
        let drop_trailing_result = matches!(
            self.messages.get(idx),
            Some(m) if m.role == Role::Assistant && !m.tool_calls.is_empty()
        );
        self.messages.remove(idx);
        if drop_trailing_result
            && idx < self.messages.len()
            && self.messages[idx].role == Role::User
            && self.messages[idx].content.starts_with(TOOL_RESULT_PREFIX)
        {
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
        // If the boundary landed on a tool-result user message, its
        // preceding assistant-with-tool_calls would be orphaned on
        // summarize. Walk back to include any matching assistant half,
        // keeping the pair intact.
        while idx > start
            && self
                .messages
                .get(idx)
                .map(|m| m.role == Role::User && m.content.starts_with(TOOL_RESULT_PREFIX))
                .unwrap_or(false)
        {
            idx -= 1;
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
    // Each tool-call entry contributes its id + name + arguments verbatim
    // plus a small per-entry structural overhead (braces, type, field
    // names). `approx_tokens` over-counts slightly on purpose.
    let tool_call_bytes: usize = m
        .tool_calls
        .iter()
        .map(|c| c.id.len() + c.name.len() + c.arguments.len() + 32)
        .sum();
    let tool_call_cost = approx_tokens_bytes(tool_call_bytes);
    TOKENS_PER_MESSAGE_OVERHEAD
        .saturating_add(approx_tokens(&m.content))
        .saturating_add(image_cost)
        .saturating_add(tool_call_cost)
}

fn approx_tokens_bytes(n: usize) -> u32 {
    ((n as u32).saturating_add(3)) / 4
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

fn effective_budget(chat: &ChatConfig, model: &ModelConfig) -> u32 {
    chat.max_history_tokens
        .min(chat.effective_context_budget(model))
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

    fn spec(max_history: u32, preserve: u32, ctx: u32) -> (ChatConfig, ModelConfig) {
        let chat = ChatConfig {
            system_prompt: "sys".into(),
            max_history_tokens: max_history,
            summary_target_tokens: max_history / 4,
            preserve_recent_turns: preserve,
            temperature: 0.7,
            max_response_tokens: 512,
            max_summary_tokens: max_history / 2,
            request_timeout_secs: 60,
            summary_temperature: 0.3,
            top_p: None,
            top_k: None,
            min_p: None,
            presence_penalty: None,
        };
        let model = ModelConfig {
            name: "test-model".into(),
            context_length: ctx,
        };
        (chat, model)
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
        assert_eq!(wire[0].content, Some(wire::ContentBody::Text("sys")));
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
            Some(wire::ContentBody::Parts(p)) => p,
            other => panic!("expected Some(Parts), got {other:?}"),
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
        assert!(matches!(
            wire[0].content,
            Some(wire::ContentBody::Text("hello"))
        ));
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
        let (chat, model) = spec(10_000, 4, 20_000);
        c.ensure_budget(&fake, &chat, &model).await.unwrap();
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
        let (chat, model) = spec(60, 2, 10_000);
        c.ensure_budget(&fake, &chat, &model).await.unwrap();

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
        let (chat, model) = spec(120, 2, 10_000);
        c.ensure_budget(&fake, &chat, &model).await.unwrap();

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
        let (chat, model) = spec(40, 2, 10_000);
        let result = c.ensure_budget(&FailingSummarizer, &chat, &model).await;
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
        let (chat, model) = spec(40, 2, 10_000);
        c.truncate_to_budget(&chat, &model);
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
        let (chat, model) = spec(20, 1, 10_000);
        c.truncate_to_budget(&chat, &model);
        assert_eq!(c.messages.last().unwrap().content, "keepme with filler");
    }

    #[test]
    fn truncate_utf8_clamps_on_codepoint_boundary() {
        let s = "世界世界世界";
        let truncated = truncate_utf8(s, 4);
        assert!(truncated.ends_with('…'));
        assert!(truncated.chars().filter(|c| *c != '…').all(|c| c == '世'));
    }

    // --- tool-call conversation support ----------------------------------

    fn mk_call(id: &str, args: &str) -> ToolCallRecord {
        ToolCallRecord {
            id: id.into(),
            name: "run".into(),
            arguments: args.into(),
        }
    }

    #[test]
    fn push_assistant_with_tool_calls_records_calls() {
        let mut c = Conversation::new(String::new());
        c.push_user("do it".into());
        c.push_assistant_with_tool_calls(None, vec![mk_call("call-1", r#"{"command":"ls"}"#)]);
        let last = c.messages.last().unwrap();
        assert_eq!(last.role, Role::Assistant);
        assert_eq!(last.content, "");
        assert_eq!(last.tool_calls.len(), 1);
        assert_eq!(last.tool_calls[0].id, "call-1");
        assert_eq!(last.tool_calls[0].name, "run");
        assert_eq!(last.tool_calls[0].arguments, r#"{"command":"ls"}"#);
    }

    #[test]
    fn as_wire_messages_renders_tool_calls_with_content_absent() {
        let mut c = Conversation::new(String::new());
        c.push_user("do it".into());
        c.push_assistant_with_tool_calls(None, vec![mk_call("call-1", r#"{"command":"ls"}"#)]);
        let wire = c.as_wire_messages();
        assert_eq!(wire.len(), 2);
        assert_eq!(wire[1].role, "assistant");
        assert!(wire[1].content.is_none(), "content must be absent");
        let calls = wire[1].tool_calls.as_ref().expect("tool_calls present");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call-1");
        assert_eq!(calls[0].kind, "function");
        assert_eq!(calls[0].function.name, "run");
        assert_eq!(calls[0].function.arguments, r#"{"command":"ls"}"#);

        // Serialized JSON must omit `content` (null is not equivalent for
        // strict Jinja templates).
        let json = serde_json::to_value(&wire[1]).unwrap();
        assert!(
            json.get("content").is_none(),
            "content key must be omitted: {json}"
        );
        assert_eq!(json["tool_calls"][0]["function"]["name"], "run");
    }

    #[test]
    fn tool_calls_contribute_to_token_budget() {
        let mut c = Conversation::new(String::new());
        c.push_user("q".into());
        let baseline = c.approx_total_tokens();

        let mut c2 = Conversation::new(String::new());
        c2.push_user("q".into());
        c2.push_assistant_with_tool_calls(
            None,
            vec![mk_call(
                "call-1",
                r#"{"command":"a very long command string here to make the call nontrivial"}"#,
            )],
        );
        let with_call = c2.approx_total_tokens();
        assert!(
            with_call > baseline,
            "tool_calls should add to token count: {with_call} vs {baseline}"
        );
    }

    #[test]
    fn truncate_drops_tool_call_pair_atomically() {
        let mut c = Conversation::new(String::new());
        // Old messages we'll summarize/truncate:
        c.push_user("old q".into());
        c.push_assistant_with_tool_calls(None, vec![mk_call("c-1", r#"{"command":"ls"}"#)]);
        c.push_user_with_attachments("[tool:run]\nsome output\n".into(), Vec::new());
        c.push_assistant("old reply".into());
        // The latest turn (kept by preserve):
        c.push_user("latest".into());

        let (chat, model) = spec(10, 1, 10_000);
        c.truncate_to_budget(&chat, &model);

        // The remaining history must not carry a dangling tool_calls
        // without a matching result (or vice versa).
        for (i, m) in c.messages.iter().enumerate() {
            if m.role == Role::Assistant && !m.tool_calls.is_empty() {
                let next = c.messages.get(i + 1);
                assert!(
                    matches!(next, Some(n) if n.role == Role::User
                             && n.content.starts_with(TOOL_RESULT_PREFIX)),
                    "assistant tool_calls at index {i} has no matching result; \
                     history: {:?}",
                    c.messages
                );
            }
            if m.role == Role::User && m.content.starts_with(TOOL_RESULT_PREFIX) {
                let prev = i.checked_sub(1).and_then(|p| c.messages.get(p));
                assert!(
                    matches!(prev, Some(p) if p.role == Role::Assistant
                             && !p.tool_calls.is_empty()),
                    "tool result at index {i} has no matching assistant; \
                     history: {:?}",
                    c.messages
                );
            }
        }
    }

    #[tokio::test]
    async fn summarize_preserves_tool_call_pair_boundary() {
        let mut c = Conversation::new("sys".into());
        // Stuff enough history to force summarization.
        for i in 0..5 {
            c.push_user(format!("turn {i} question with some filler text here"));
            c.push_assistant(format!(
                "turn {i} answer with some filler text here to blow budget"
            ));
        }
        // A tool-call pair in the recent tail.
        c.push_assistant_with_tool_calls(None, vec![mk_call("c-99", r#"{"command":"ls"}"#)]);
        c.push_user_with_attachments("[tool:run]\nfoo\nbar\n".into(), Vec::new());
        c.push_user("latest".into());

        let fake = FakeSummarizer::new("summary");
        let (chat, model) = spec(60, 2, 10_000);
        c.ensure_budget(&fake, &chat, &model).await.unwrap();

        // Post-summarize, the preserved tail must not have a dangling
        // tool_calls or tool-result.
        for (i, m) in c.messages.iter().enumerate() {
            if m.role == Role::User && m.content.starts_with(TOOL_RESULT_PREFIX) {
                let prev = i.checked_sub(1).and_then(|p| c.messages.get(p));
                assert!(
                    matches!(prev, Some(p) if p.role == Role::Assistant
                             && !p.tool_calls.is_empty()),
                    "orphaned tool result at {i}"
                );
            }
        }
    }
}
