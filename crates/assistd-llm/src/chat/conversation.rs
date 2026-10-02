//! Multi-turn conversation state with token budgeting: a bytes-per-token
//! heuristic, corrected by the prompt size the server last measured.

use std::borrow::Cow;

use assistd_config::{ChatConfig, ModelConfig};
use assistd_tools::Attachment;
use async_trait::async_trait;
use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use tracing::{debug, warn};

use super::error::ChatClientError;
use super::wire;

const SUMMARY_PREFIX: &str = "[Conversation summary] ";
/// Marks an image-carrying tool result sent on the user role (templates
/// render images only on user turns); text-only results use [`Role::Tool`].
pub const TOOL_RESULT_PREFIX: &str = "[tool:";
/// Plain-text delimiters around a context block folded into its user turn;
/// templates mishandle system messages anywhere but the head of the list.
const CONTEXT_OPEN: &str = "[Context: added automatically, not written by the user]\n";
const CONTEXT_CLOSE: &str = "\n[End of context]\n\n";
const TOKENS_PER_MESSAGE_OVERHEAD: u32 = 4;
/// Conservative per-image token weight; errs toward summarizing early.
const TOKENS_PER_IMAGE: u32 = 1000;

/// Message role for conversation turns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    /// Output answering one of the preceding assistant `tool_calls`.
    Tool,
}

impl Role {
    /// Returns the OpenAI wire string for this role.
    pub fn as_wire(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

/// One tool call recorded on an assistant turn. `arguments` is the
/// JSON-encoded string the model emitted, replayed verbatim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallRecord {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// An image encoded once, on entry, as the `data:` URI it is sent as.
#[derive(Debug, Clone)]
pub struct ImageDataUri(String);

impl ImageDataUri {
    fn encode(attachment: Attachment) -> Self {
        match attachment {
            Attachment::Image { mime, bytes } => {
                let mut uri = String::with_capacity(
                    "data:;base64,".len() + mime.len() + bytes.len().div_ceil(3) * 4,
                );
                uri.push_str("data:");
                uri.push_str(&mime);
                uri.push_str(";base64,");
                B64.encode_string(bytes, &mut uri);
                Self(uri)
            }
        }
    }

    /// The URI text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// One turn in the in-memory conversation, owned by [`Conversation`].
#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: String,
    pub attachments: Vec<ImageDataUri>,
    /// Non-empty only on assistant messages that requested tool calls.
    pub tool_calls: Vec<ToolCallRecord>,
    /// On [`Role::Tool`] messages: the id of the call this answers.
    pub tool_call_id: Option<String>,
    /// Reasoning behind an assistant's `tool_calls`, sent back with them.
    pub reasoning: String,
    /// Context block rendered at the head of a user message's text,
    /// keeping earlier turns byte-identical for the prefix cache.
    pub context: Option<String>,
}

impl Message {
    /// A message with only `role` and `content` set.
    pub fn text(role: Role, content: String) -> Self {
        Self {
            role,
            content,
            attachments: Vec::new(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            reasoning: String::new(),
            context: None,
        }
    }
}

/// Condenses a stretch of dialogue into a summary when the conversation
/// outgrows its token budget.
#[async_trait]
pub trait Summarizer: Send + Sync {
    /// Summarize `dialogue` into at most `max_tokens`, targeting `target_tokens`.
    async fn summarize(
        &self,
        dialogue: String,
        target_tokens: u32,
        max_tokens: u32,
    ) -> Result<String, ChatClientError>;
}

/// Mutable conversation state. `messages` never holds the system prompt,
/// only turns plus at most one `SUMMARY_PREFIX` system message at index 0;
/// the rendered form carries a single system message, at its head.
#[derive(Debug)]
pub struct Conversation {
    system_prompt: String,
    pending_context: Option<String>,
    transient_note: Option<String>,
    messages: Vec<Message>,
    /// How far the server's measured prompt size was from the heuristic
    /// on the last measured request, in tokens.
    calibration: i32,
}

impl Conversation {
    /// Creates an empty conversation with the given static system prompt.
    pub fn new(system_prompt: String) -> Self {
        Self {
            system_prompt,
            pending_context: None,
            transient_note: None,
            messages: Vec::new(),
            calibration: 0,
        }
    }

    /// Stash the context block for the next user turn, replacing any
    /// pending one. It attaches when that turn is pushed and renders at
    /// the head of its text.
    pub fn set_transient_context(&mut self, text: String) {
        self.pending_context = Some(text);
    }

    /// Set a one-shot note rendered as the final user message,
    /// replacing any pending one.
    pub fn set_transient_note(&mut self, text: String) {
        self.transient_note = Some(text);
    }

    /// Take the pending transient note, leaving `None` behind.
    pub fn consume_transient_note(&mut self) -> Option<String> {
        self.transient_note.take()
    }

    /// The context block waiting for the next user turn, if any.
    #[cfg(test)]
    pub fn pending_context(&self) -> Option<&str> {
        self.pending_context.as_deref()
    }

    /// Appends a plain-text user turn.
    pub fn push_user(&mut self, content: String) {
        self.push_user_with_attachments(content, Vec::new());
    }

    /// Append a user turn whose wire form is a multimodal `content`
    /// array: one `text` part followed by one `image_url` part per
    /// attachment.
    pub fn push_user_with_attachments(&mut self, content: String, attachments: Vec<Attachment>) {
        self.messages.push(Message {
            attachments: encode_all(attachments),
            context: self.pending_context.take(),
            ..Message::text(Role::User, content)
        });
    }

    /// Append an image-carrying tool result as a tagged user message
    /// (see [`TOOL_RESULT_PREFIX`]). Unlike a real user turn this does
    /// not close the tool loop it belongs to.
    pub fn push_tool_result_with_attachments(
        &mut self,
        name: &str,
        content: &str,
        attachments: Vec<Attachment>,
    ) {
        self.messages.push(Message {
            attachments: encode_all(attachments),
            ..Message::text(
                Role::User,
                format!("{TOOL_RESULT_PREFIX}{name}]\n{content}"),
            )
        });
    }

    /// Appends a plain-text assistant turn.
    pub fn push_assistant(&mut self, content: String) {
        self.messages.push(Message::text(Role::Assistant, content));
    }

    /// Append an assistant turn that requested `calls` (non-empty), with
    /// the narration and reasoning streamed before them.
    pub fn push_assistant_with_tool_calls(
        &mut self,
        content: Option<String>,
        reasoning: String,
        calls: Vec<ToolCallRecord>,
    ) {
        debug_assert!(
            !calls.is_empty(),
            "push_assistant_with_tool_calls requires at least one call"
        );
        self.messages.push(Message {
            tool_calls: calls,
            reasoning,
            ..Message::text(Role::Assistant, content.unwrap_or_default())
        });
    }

    /// Append the output of one tool call as a `role: "tool"` message.
    pub fn push_tool_result(&mut self, call_id: String, content: String) {
        self.messages.push(Message {
            tool_call_id: Some(call_id),
            ..Message::text(Role::Tool, content)
        });
    }

    /// Drop the most recent message if it is a real user message (not a
    /// tool result), returning its context block to pending so a retried
    /// turn still carries it.
    pub fn rollback_last_user(&mut self) {
        if self
            .messages
            .last()
            .is_some_and(|m| m.role == Role::User && !Self::is_tool_result(m))
            && let Some(user) = self.messages.pop()
        {
            self.pending_context = user.context;
        }
    }

    /// Replace the message list wholesale and clear any pending
    /// context or note.
    pub fn replace_messages(&mut self, msgs: Vec<Message>) {
        self.messages = msgs;
        self.pending_context = None;
        self.transient_note = None;
        self.calibration = 0;
    }

    /// Drop everything from the latest non-tool-result user message onward
    /// and clear pending context and note. Returns the number removed.
    pub fn truncate_to_last_real_user(&mut self) -> usize {
        let Some(idx) = self.last_real_user_index() else {
            return 0;
        };
        let removed = self.messages.len() - idx;
        self.messages.truncate(idx);
        self.pending_context = None;
        self.transient_note = None;
        removed
    }

    /// Estimated prompt size in tokens: [`Self::heuristic_tokens`] shifted
    /// by the error the server last measured in it.
    pub fn approx_total_tokens(&self) -> u32 {
        self.heuristic_tokens()
            .saturating_add_signed(self.calibration)
    }

    /// Record that a request whose [`Self::heuristic_tokens`] was
    /// `estimated` measured `prompt_tokens` on the server.
    pub fn calibrate(&mut self, estimated: u32, prompt_tokens: u32) {
        let error = i64::from(prompt_tokens) - i64::from(estimated);
        let bounded = error.clamp(i64::from(i32::MIN), i64::from(i32::MAX));
        self.calibration = i32::try_from(bounded).unwrap_or_default();
    }

    pub fn message_count(&self) -> usize {
        self.messages.len()
    }

    /// Token count of the conversation by a bytes-per-token heuristic. It
    /// knows nothing of tool schemas or the chat template's own tokens.
    pub fn heuristic_tokens(&self) -> u32 {
        let mut total = 0u32;
        if !self.system_prompt.is_empty() {
            total = total.saturating_add(
                TOKENS_PER_MESSAGE_OVERHEAD.saturating_add(approx_tokens(&self.system_prompt)),
            );
        }
        for transient in self.pending_context.iter().chain(&self.transient_note) {
            total = total.saturating_add(
                TOKENS_PER_MESSAGE_OVERHEAD.saturating_add(approx_tokens(transient)),
            );
        }
        for m in &self.messages {
            total = total.saturating_add(approx_message_tokens(m));
        }
        total
    }

    /// Render the current state as wire messages: the system head, each
    /// turn, then the transient note as a final user message. Messages with
    /// attachments become multimodal `content` arrays; others stay strings.
    pub fn as_wire_messages(&self) -> Vec<wire::ChatMessage<'_>> {
        let mut out = Vec::with_capacity(self.messages.len() + 2);
        let turns_start = self.summary_insertion_index();
        out.extend(self.system_head_message(turns_start));
        out.extend(self.messages[turns_start..].iter().map(wire_message));
        if let Some(note) = &self.transient_note {
            let content = wire::ContentBody::Text(Cow::Borrowed(note.as_str()));
            out.push(wire::ChatMessage::plain(Role::User.as_wire(), content));
        }
        out
    }

    /// The system prompt with any summary appended, as one message.
    fn system_head_message(&self, turns_start: usize) -> Option<wire::ChatMessage<'_>> {
        let summary = self.messages[..turns_start]
            .first()
            .map(|m| m.content.as_str());
        let head = match (self.system_prompt.as_str(), summary) {
            ("", None) => return None,
            ("", Some(summary)) => Cow::Borrowed(summary),
            (prompt, None) => Cow::Borrowed(prompt),
            (prompt, Some(summary)) => Cow::Owned(format!("{prompt}\n\n{summary}")),
        };
        Some(wire::ChatMessage::plain(
            Role::System.as_wire(),
            wire::ContentBody::Text(head),
        ))
    }

    /// Keep the approximate token total under budget, folding the oldest
    /// turns and any earlier summary into one summary if needed. Returns an
    /// error, with history unchanged, if the summarizer fails or returns
    /// empty text; [`Self::truncate_to_budget`] is the infallible fallback.
    pub async fn ensure_budget(
        &mut self,
        summarizer: &dyn Summarizer,
        chat: &ChatConfig,
        model: &ModelConfig,
    ) -> Result<(), ChatClientError> {
        let budget = effective_budget(chat, model);
        if !self.exceeds_budget(chat, model) {
            return Ok(());
        }

        let preserve_pairs = chat.preserve_recent_turns.get() as usize;
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

        let dialogue = serialize_tail(&self.messages[..preserve_from]);
        if dialogue.trim().is_empty() {
            self.truncate_to_budget(chat, model);
            return Ok(());
        }

        let summary = summarizer
            .summarize(
                dialogue,
                chat.summary_target_tokens.get(),
                chat.max_summary_tokens(),
            )
            .await?;
        let trimmed = summary.trim();
        if trimmed.is_empty() {
            return Err(ChatClientError::Summarize(
                "summarizer returned empty text".into(),
            ));
        }

        let max_summary_bytes = (chat.summary_target_tokens.get() as usize).saturating_mul(4);
        let body = if trimmed.len() > max_summary_bytes {
            truncate_utf8(trimmed, max_summary_bytes)
        } else {
            trimmed.to_string()
        };

        self.messages.splice(
            ..preserve_from,
            [Message::text(
                Role::System,
                format!("{SUMMARY_PREFIX}{body}"),
            )],
        );

        if self.approx_total_tokens() > budget {
            debug!(
                target: "assistd::chat",
                "ensure_budget: still over budget after summarize, truncating"
            );
            self.truncate_to_budget(chat, model);
        }
        Ok(())
    }

    pub fn exceeds_budget(&self, chat: &ChatConfig, model: &ModelConfig) -> bool {
        self.approx_total_tokens() > effective_budget(chat, model)
    }

    /// Drop the oldest messages after any summary until the conversation
    /// fits the budget, then the oldest tool steps of the current turn. The
    /// latest user message and the newest step always stay.
    pub fn truncate_to_budget(&mut self, chat: &ChatConfig, model: &ModelConfig) {
        let budget = effective_budget(chat, model);
        while self.approx_total_tokens() > budget {
            let Some(idx) = self.first_droppable_index() else {
                warn!(
                    target: "assistd::chat",
                    "truncate_to_budget: cannot drop any more messages without losing the current turn"
                );
                break;
            };
            self.drop_with_pair(idx);
        }
    }

    /// Remove `idx` and, when it is an assistant message with tool
    /// calls, the tool results that follow it.
    fn drop_with_pair(&mut self, idx: usize) {
        if idx >= self.messages.len() {
            return;
        }
        let drop_trailing_result = matches!(
            self.messages.get(idx),
            Some(m) if m.role == Role::Assistant && !m.tool_calls.is_empty()
        );
        self.messages.remove(idx);
        while drop_trailing_result && self.messages.get(idx).is_some_and(Self::is_tool_result) {
            self.messages.remove(idx);
        }
    }

    /// True for a [`Role::Tool`] message or a [`TOOL_RESULT_PREFIX`] user message.
    fn is_tool_result(m: &Message) -> bool {
        m.role == Role::Tool || (m.role == Role::User && m.content.starts_with(TOOL_RESULT_PREFIX))
    }

    fn summary_insertion_index(&self) -> usize {
        usize::from(
            self.messages
                .first()
                .is_some_and(|m| m.role == Role::System && m.content.starts_with(SUMMARY_PREFIX)),
        )
    }

    /// Start of the newest `preserve_pairs` user/assistant pairs, widened so
    /// the boundary never splits a tool call from its results or passes the
    /// latest user message.
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
                    idx = if prev > start { prev - 1 } else { prev };
                    pairs_seen += 1;
                }
                Role::System | Role::Tool => {
                    idx = prev;
                }
            }
        }
        while idx > start && self.messages.get(idx).is_some_and(Self::is_tool_result) {
            idx -= 1;
        }
        idx.min(self.current_turn_start()).max(start)
    }

    fn first_droppable_index(&self) -> Option<usize> {
        let start = self.summary_insertion_index();
        let turn_start = self.current_turn_start();
        if start < turn_start {
            Some(start)
        } else {
            self.oldest_superseded_step(turn_start)
        }
    }

    /// The first tool step of the turn starting at `turn_start`, unless it
    /// is also the newest one.
    fn oldest_superseded_step(&self, turn_start: usize) -> Option<usize> {
        let is_step = |m: &Message| m.role == Role::Assistant && !m.tool_calls.is_empty();
        let oldest = turn_start + 1;
        let newest = self.messages.iter().rposition(is_step)?;
        (self.messages.get(oldest).is_some_and(is_step) && oldest < newest).then_some(oldest)
    }

    fn last_real_user_index(&self) -> Option<usize> {
        self.messages
            .iter()
            .rposition(|m| m.role == Role::User && !Self::is_tool_result(m))
    }

    fn current_turn_start(&self) -> usize {
        self.last_real_user_index().unwrap_or(self.messages.len())
    }
}

fn approx_tokens(text: &str) -> u32 {
    u32::try_from(text.len()).unwrap_or(u32::MAX).div_ceil(4)
}

fn approx_message_tokens(m: &Message) -> u32 {
    let image_cost = u32::try_from(m.attachments.len())
        .unwrap_or(u32::MAX)
        .saturating_mul(TOKENS_PER_IMAGE);
    let tool_call_bytes: usize = m
        .tool_calls
        .iter()
        .map(|c| c.id.len() + c.name.len() + c.arguments.len() + 32)
        .sum();
    let tool_call_cost = approx_tokens_bytes(tool_call_bytes);
    let context_cost = m.context.as_deref().map_or(0, |ctx| {
        approx_tokens_bytes(CONTEXT_OPEN.len() + ctx.len() + CONTEXT_CLOSE.len())
    });
    TOKENS_PER_MESSAGE_OVERHEAD
        .saturating_add(approx_tokens(&m.content))
        .saturating_add(approx_tokens(&m.reasoning))
        .saturating_add(image_cost)
        .saturating_add(tool_call_cost)
        .saturating_add(context_cost)
}

fn approx_tokens_bytes(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX).saturating_add(3) / 4
}

fn wire_message(message: &Message) -> wire::ChatMessage<'_> {
    if !message.tool_calls.is_empty() {
        return tool_calls_message(message);
    }
    if message.role == Role::Tool {
        let content = wire::ContentBody::Text(Cow::Borrowed(&message.content));
        return wire::ChatMessage {
            tool_call_id: message.tool_call_id.as_deref(),
            ..wire::ChatMessage::plain(message.role.as_wire(), content)
        };
    }
    wire::ChatMessage::plain(message.role.as_wire(), turn_content(message))
}

fn tool_calls_message(message: &Message) -> wire::ChatMessage<'_> {
    let specs = message
        .tool_calls
        .iter()
        .map(|call| wire::ToolCallSpec {
            id: &call.id,
            kind: "function",
            function: wire::FunctionCallSpec {
                name: &call.name,
                arguments: &call.arguments,
            },
        })
        .collect();
    wire::ChatMessage {
        role: message.role.as_wire(),
        content: (!message.content.is_empty())
            .then(|| wire::ContentBody::Text(Cow::Borrowed(&message.content))),
        tool_calls: Some(specs),
        tool_call_id: None,
        reasoning_content: (!message.reasoning.is_empty()).then_some(message.reasoning.as_str()),
    }
}

fn turn_content(message: &Message) -> wire::ContentBody<'_> {
    let text = wire_text(message);
    if message.attachments.is_empty() {
        return wire::ContentBody::Text(text);
    }
    let mut parts = Vec::with_capacity(message.attachments.len() + 1);
    parts.push(wire::ContentPart::Text { text });
    parts.extend(
        message
            .attachments
            .iter()
            .map(|image| wire::ContentPart::ImageUrl {
                image_url: wire::ImageUrl {
                    url: image.as_str(),
                },
            }),
    );
    wire::ContentBody::Parts(parts)
}

fn wire_text(message: &Message) -> Cow<'_, str> {
    match &message.context {
        Some(ctx) => Cow::Owned(format!(
            "{CONTEXT_OPEN}{}{CONTEXT_CLOSE}{}",
            neutralise_context_markers(ctx.trim_end()),
            message.content
        )),
        None => Cow::Borrowed(&message.content),
    }
}

fn neutralise_context_markers(ctx: &str) -> Cow<'_, str> {
    let open = CONTEXT_OPEN.trim();
    let close = CONTEXT_CLOSE.trim();
    if !ctx.contains(open) && !ctx.contains(close) {
        return Cow::Borrowed(ctx);
    }
    let defang = |marker: &str| marker.replace('[', "(").replace(']', ")");
    Cow::Owned(
        ctx.replace(open, &defang(open))
            .replace(close, &defang(close)),
    )
}

fn encode_all(attachments: Vec<Attachment>) -> Vec<ImageDataUri> {
    attachments.into_iter().map(ImageDataUri::encode).collect()
}

fn effective_budget(chat: &ChatConfig, model: &ModelConfig) -> u32 {
    chat.max_history_tokens.get().min(model.context_budget())
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
    let end = s.floor_char_boundary(max_bytes);
    let mut out = String::with_capacity(end + 1);
    out.push_str(&s[..end]);
    out.push('…');
    out
}

#[cfg(test)]
mod tests;
