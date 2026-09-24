//! Multi-turn conversation state. Token budgeting is best-effort,
//! driven by a bytes-per-token heuristic that intentionally over-counts
//! multi-byte text so summarization runs early rather than the server's
//! context window overflowing.

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
/// Tool results that carry images stay on the user role, because chat
/// templates render image parts only on user turns. This prefix marks
/// those so the model can still tell them from genuine user speech and
/// the truncator can pair them with their assistant `tool_calls`
/// predecessor. Text-only results use [`Role::Tool`] instead.
pub const TOOL_RESULT_PREFIX: &str = "[tool:";
/// Delimiters around a context block folded into its user turn. Plain
/// text, because chat templates disagree about system messages anywhere
/// but the head of the list: some drop them silently and some reject the
/// request, while every template renders the text of a user turn. The
/// opening line says who wrote the block so the model does not read it
/// as the user speaking.
const CONTEXT_OPEN: &str = "[Context: added automatically, not written by the user]\n";
const CONTEXT_CLOSE: &str = "\n[End of context]\n\n";
const TOKENS_PER_MESSAGE_OVERHEAD: u32 = 4;
/// Conservative per-image token weight for budget math. Real usage
/// depends on the vision model, but 1000 tokens errs on the side of
/// summarizing earlier rather than overflowing.
const TOKENS_PER_IMAGE: u32 = 1000;

/// Message role for conversation turns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    /// Output of a tool the assistant called, answering one entry of the
    /// preceding assistant message's `tool_calls`.
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
/// JSON-encoded string the model emitted, stored verbatim because some
/// servers compare the replayed text against their own serialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallRecord {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// An image held as the `data:` URI it goes out as. Encoding happens
/// once, when the image enters the conversation, rather than on every
/// request that replays it; the raw bytes are dropped at that point.
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
    /// Set only on [`Role::Tool`] messages: the id of the assistant tool
    /// call this message answers.
    pub tool_call_id: Option<String>,
    /// The reasoning behind an assistant message's `tool_calls`. Held
    /// only while the tool loop it belongs to is in progress: the next
    /// user turn clears it, because chat templates stop rendering it
    /// from that point on.
    pub reasoning: String,
    /// Set only on user messages, and rendered at the head of that
    /// message's text. Anchoring context here rather than beside the
    /// static system prompt keeps every earlier turn byte-identical
    /// across requests, so the server's prefix cache covers the whole
    /// history instead of only the system prompt.
    pub context: Option<String>,
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

/// Mutable conversation state.
///
/// Layout invariants:
/// - `pending_context`, if `Some`, is attached to the next user turn
///   pushed and renders inside that message, ahead of the user's own
///   text, for the rest of that turn (see [`Message::context`]).
/// - `as_wire_messages()` carries at most one system message, at its
///   head: the prompt with the summary appended. Chat templates do not
///   reliably render a second one or one further down.
/// - `transient_note`, if `Some`, renders as a final user message and
///   lives for one request; the caller clears it with
///   [`Self::consume_transient_note`] once that request commits. The
///   tail is where a mid-turn instruction has to go: a model deep in a
///   tool loop follows what it read last, not a system prompt thousands
///   of tokens upstream.
/// - `messages` never holds the system prompt itself; it holds the
///   turns plus at most one summary message (role `System`, content
///   prefixed with `SUMMARY_PREFIX`) at index 0.
#[derive(Debug)]
pub struct Conversation {
    system_prompt: String,
    pending_context: Option<String>,
    transient_note: Option<String>,
    messages: Vec<Message>,
}

impl Conversation {
    /// Creates an empty conversation with the given static system prompt.
    pub fn new(system_prompt: String) -> Self {
        Self {
            system_prompt,
            pending_context: None,
            transient_note: None,
            messages: Vec::new(),
        }
    }

    /// Stash the context block for the next user turn, replacing any
    /// pending one. It attaches when that turn is pushed and renders at
    /// the head of its text until the following user turn.
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

    /// Appends a plain-text user turn, closing the previous turn.
    pub fn push_user(&mut self, content: String) {
        self.push_user_with_attachments(content, Vec::new());
    }

    /// Append a user turn whose wire form is a multimodal `content`
    /// array: one `text` part followed by one `image_url` part per
    /// attachment. Closes the previous turn like [`Self::push_user`].
    pub fn push_user_with_attachments(&mut self, content: String, attachments: Vec<Attachment>) {
        self.close_previous_turn();
        self.messages.push(Message {
            role: Role::User,
            content,
            attachments: encode_all(attachments),
            tool_calls: Vec::new(),
            tool_call_id: None,
            reasoning: String::new(),
            context: self.pending_context.take(),
        });
    }

    /// Append an image-carrying tool result as a tagged user message
    /// (see [`TOOL_RESULT_PREFIX`]). Unlike a real user turn this does
    /// not close the tool loop it belongs to.
    pub fn push_tool_result_with_attachments(
        &mut self,
        name: &str,
        content: String,
        attachments: Vec<Attachment>,
    ) {
        self.messages.push(Message {
            role: Role::User,
            content: format!("{TOOL_RESULT_PREFIX}{name}]\n{content}"),
            attachments: encode_all(attachments),
            tool_calls: Vec::new(),
            tool_call_id: None,
            reasoning: String::new(),
            context: None,
        });
    }

    /// Appends a plain-text assistant turn.
    pub fn push_assistant(&mut self, content: String) {
        self.messages.push(Message {
            role: Role::Assistant,
            content,
            attachments: Vec::new(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            reasoning: String::new(),
            context: None,
        });
    }

    /// Append an assistant turn that requested tool calls. `content` is
    /// the narration streamed before the call, kept so the model does not
    /// repeat itself on the next step. `reasoning` is what the model
    /// thought before calling; a reasoning model that is shown its earlier
    /// steps with the thinking stripped out starts skipping the thinking
    /// itself, and then ends turns it had just said it would continue.
    /// `calls` must be non-empty.
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
            role: Role::Assistant,
            content: content.unwrap_or_default(),
            attachments: Vec::new(),
            tool_calls: calls,
            tool_call_id: None,
            reasoning,
            context: None,
        });
    }

    /// Append the output of one tool call as an OpenAI `role: "tool"`
    /// message. Routing results here rather than onto the user role is
    /// what keeps the model reading them as its own tool's output: a
    /// user turn reads as the person speaking again, and the model
    /// answers it by re-introducing what it is about to do.
    pub fn push_tool_result(&mut self, call_id: String, content: String) {
        self.messages.push(Message {
            role: Role::Tool,
            content,
            attachments: Vec::new(),
            tool_calls: Vec::new(),
            tool_call_id: Some(call_id),
            reasoning: String::new(),
            context: None,
        });
    }

    fn close_previous_turn(&mut self) {
        for message in &mut self.messages {
            message.reasoning.clear();
            message.context = None;
        }
    }

    /// Drop the most recent message if and only if it is a user message,
    /// keeping history consistent with what the model actually saw after
    /// a request fails before any output. Its context block, if any,
    /// goes back to pending so a retried turn still carries it.
    pub fn rollback_last_user(&mut self) {
        if matches!(self.messages.last().map(|m| m.role), Some(Role::User))
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
    }

    /// Drop everything from the latest real user message onward, where
    /// a tool result riding on the user role does not count as one.
    /// Also clears `pending_context` and `transient_note`. Returns the
    /// number of removed entries; 0 when no real user message exists.
    pub fn truncate_to_last_real_user(&mut self) -> usize {
        let mut last_real_user = None;
        for (i, m) in self.messages.iter().enumerate().rev() {
            if m.role == Role::User && !Self::is_tool_result(m) {
                last_real_user = Some(i);
                break;
            }
        }
        let Some(idx) = last_real_user else {
            return 0;
        };
        let removed = self.messages.len() - idx;
        self.messages.truncate(idx);
        self.pending_context = None;
        self.transient_note = None;
        removed
    }

    /// Estimates the total token count of the conversation using a bytes-per-token heuristic.
    pub fn approx_total_tokens(&self) -> u32 {
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

    /// Render the current state as wire messages. Text-only messages
    /// stay plain strings for compatibility with non-vision models;
    /// messages with attachments become multimodal `content` arrays.
    pub fn as_wire_messages(&self) -> Vec<wire::ChatMessage<'_>> {
        let mut out = Vec::with_capacity(self.messages.len() + 2);
        let turns_start = self.summary_insertion_index();
        let summary = self.messages[..turns_start]
            .first()
            .map(|m| m.content.as_str());
        let head = match (self.system_prompt.as_str(), summary) {
            ("", None) => None,
            ("", Some(summary)) => Some(Cow::Borrowed(summary)),
            (prompt, None) => Some(Cow::Borrowed(prompt)),
            (prompt, Some(summary)) => Some(Cow::Owned(format!("{prompt}\n\n{summary}"))),
        };
        if let Some(head) = head {
            out.push(wire::ChatMessage {
                role: Role::System.as_wire(),
                content: Some(wire::ContentBody::Text(head)),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            });
        }
        for message in &self.messages[turns_start..] {
            if !message.tool_calls.is_empty() {
                let specs: Vec<wire::ToolCallSpec<'_>> = message
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
                out.push(wire::ChatMessage {
                    role: message.role.as_wire(),
                    content: (!message.content.is_empty())
                        .then(|| wire::ContentBody::Text(Cow::Borrowed(&message.content))),
                    tool_calls: Some(specs),
                    tool_call_id: None,
                    reasoning_content: (!message.reasoning.is_empty())
                        .then_some(message.reasoning.as_str()),
                });
                continue;
            }
            if message.role == Role::Tool {
                out.push(wire::ChatMessage {
                    role: message.role.as_wire(),
                    content: Some(wire::ContentBody::Text(Cow::Borrowed(&message.content))),
                    tool_calls: None,
                    tool_call_id: message.tool_call_id.as_deref(),
                    reasoning_content: None,
                });
                continue;
            }
            let text = wire_text(message);
            let content =
                if message.attachments.is_empty() {
                    wire::ContentBody::Text(text)
                } else {
                    let mut parts = Vec::with_capacity(message.attachments.len() + 1);
                    parts.push(wire::ContentPart::Text { text });
                    parts.extend(message.attachments.iter().map(|image| {
                        wire::ContentPart::ImageUrl {
                            image_url: wire::ImageUrl {
                                url: image.as_str(),
                            },
                        }
                    }));
                    wire::ContentBody::Parts(parts)
                };
            out.push(wire::ChatMessage {
                role: message.role.as_wire(),
                content: Some(content),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            });
        }
        if let Some(note) = &self.transient_note {
            out.push(wire::ChatMessage {
                role: Role::User.as_wire(),
                content: Some(wire::ContentBody::Text(Cow::Borrowed(note))),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            });
        }
        out
    }

    /// Keep the approximate token total under budget, summarizing the
    /// oldest turns if needed. Returns an error, with history unchanged,
    /// if the summarizer fails or returns empty text;
    /// [`Self::truncate_to_budget`] is the infallible fallback.
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

        let dialogue = serialize_tail(&self.messages[tail_start..preserve_from]);
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

        let summary_msg = Message {
            role: Role::System,
            content: format!("{SUMMARY_PREFIX}{body}"),
            attachments: Vec::new(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            reasoning: String::new(),
            context: None,
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

    /// Drop the oldest messages after any summary until the conversation
    /// fits the budget or only the latest user turn remains. Tool-call and
    /// result pairs are dropped together, because most chat templates
    /// reject `tool_calls` without matching results (or vice versa).
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

    /// Remove `idx` and, when it is an assistant message with tool
    /// calls, the tool results that follow it. `first_droppable_index`
    /// always yields the assistant half first, so the reverse direction
    /// never needs handling.
    fn drop_with_pair(&mut self, idx: usize) {
        if idx >= self.messages.len() {
            return;
        }
        let drop_trailing_result = matches!(
            self.messages.get(idx),
            Some(m) if m.role == Role::Assistant && !m.tool_calls.is_empty()
        );
        self.messages.remove(idx);
        while drop_trailing_result
            && self
                .messages
                .get(idx)
                .map(Self::is_tool_result)
                .unwrap_or(false)
        {
            self.messages.remove(idx);
        }
    }

    /// Both shapes a tool result can take: the [`Role::Tool`] message
    /// text-only results use, and the prefixed user message an
    /// image-carrying result still rides in.
    fn is_tool_result(m: &Message) -> bool {
        m.role == Role::Tool || (m.role == Role::User && m.content.starts_with(TOOL_RESULT_PREFIX))
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
                    // An assistant reply is counted together with the
                    // user message it answers, so the walk steps over
                    // both at once.
                    if prev > start {
                        idx = prev - 1;
                    } else {
                        idx = prev;
                    }
                    pairs_seen += 1;
                }
                Role::System | Role::Tool => {
                    idx = prev;
                }
            }
        }
        // A boundary on a tool result would orphan its assistant
        // `tool_calls` half on summarize, so walk back to keep the pair
        // intact.
        while idx > start
            && self
                .messages
                .get(idx)
                .map(Self::is_tool_result)
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
    let tool_call_bytes: usize = m
        .tool_calls
        .iter()
        .map(|c| c.id.len() + c.name.len() + c.arguments.len() + 32)
        .sum();
    let tool_call_cost = approx_tokens_bytes(tool_call_bytes);
    let context_cost = m
        .context
        .as_deref()
        .map(|ctx| approx_tokens_bytes(CONTEXT_OPEN.len() + ctx.len() + CONTEXT_CLOSE.len()))
        .unwrap_or(0);
    TOKENS_PER_MESSAGE_OVERHEAD
        .saturating_add(approx_tokens(&m.content))
        .saturating_add(approx_tokens(&m.reasoning))
        .saturating_add(image_cost)
        .saturating_add(tool_call_cost)
        .saturating_add(context_cost)
}

fn approx_tokens_bytes(n: usize) -> u32 {
    ((n as u32).saturating_add(3)) / 4
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
mod tests;
