//! Multi-turn conversation state with token budgeting: a bytes-per-token
//! heuristic, corrected by the prompt size the server last measured.

use std::borrow::Cow;
use std::ops::Range;

use assistd_config::{ChatConfig, ModelConfig};
use assistd_tools::Attachment;
use async_trait::async_trait;
use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use tracing::{debug, warn};

use super::error::ChatClientError;
use super::wire;

mod step_log;

pub use step_log::StepRecord;

const SUMMARY_PREFIX: &str = "[Conversation summary] ";
/// Opens the instruction that asks the model to summarize its own context.
const SUMMARY_INSTRUCTION_HEAD: &str = "[Context compaction] Pause the task and make no tool \
    calls in this reply. Part of the conversation above is about to be replaced by your summary, \
    and you will continue from that summary alone.";
/// What every summary must keep, current state first so a reply cut off at
/// its length limit loses the oldest history rather than the next step.
const SUMMARY_INSTRUCTION_RULES: &str = "Start with where things stand now: what the user's \
    latest request needs, what has been established for it, what remains open, and the next \
    step. Then cover the earlier history, most recent first, preserving every factual claim, user \
    request, conclusion, and what each tool call found. Keep every approach that failed and why, \
    and every limit discovered in the tools or environment (commands or flags that are \
    unsupported, hosts that are unreachable, paths that do not exist), so none is retried. Write \
    with no preamble or commentary.";
/// Longest quote of a message the summary instruction uses to name it.
const SUMMARY_QUOTE_BYTES: usize = 120;
/// Marks an image-carrying tool result sent on the user role (templates
/// render images only on user turns); text-only results use [`Role::Tool`].
pub const TOOL_RESULT_PREFIX: &str = "[tool:";
/// Plain-text delimiters around a context block folded into its user turn;
/// templates mishandle system messages anywhere but the head of the list.
const CONTEXT_OPEN: &str = "[Context: added automatically, not written by the user]\n";
const CONTEXT_CLOSE: &str = "\n[End of context]\n\n";
/// Chat-template token text and its inert spelling; llama-server tokenizes
/// message text with special-token parsing on, so untrusted text could
/// otherwise forge a turn boundary or tool call.
const TEMPLATE_TOKEN_DEFANGS: [(&str, &str); 8] = [
    ("<|", "< |"),
    ("|>", "| >"),
    ("<think>", "< think>"),
    ("</think>", "< /think>"),
    ("<tool_call>", "< tool_call>"),
    ("</tool_call>", "< /tool_call>"),
    ("<tool_response>", "< tool_response>"),
    ("</tool_response>", "< /tool_response>"),
];
const TOKENS_PER_MESSAGE_OVERHEAD: u32 = 4;
/// Conservative per-image token weight; errs toward summarizing early.
const TOKENS_PER_IMAGE: u32 = 1000;
/// Share of the history budget, in percent, past which compaction is near.
const COMPACTION_WARNING_PERCENT: u64 = 80;
/// Share of the history budget, in percent, that compaction trims down to,
/// leaving room for several large tool steps before the next one.
const COMPACTION_TARGET_PERCENT: u64 = 40;
/// Share of `summary_target_tokens`, in percent, the instruction asks for, so
/// a summary that overruns it still ends before its length limit.
const SUMMARY_ASK_PERCENT: u32 = 75;

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
    /// On a real user message: tool calls already folded out of its turn,
    /// rendered after its text.
    pub earlier_steps: Vec<StepRecord>,
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
            earlier_steps: Vec::new(),
        }
    }
}

/// A summary to write by continuing the conversation itself, so the server
/// can reuse the prompt it already holds and read only the instruction.
#[derive(Debug)]
pub struct SummaryRequest<'a> {
    /// The conversation exactly as the next step would render it.
    pub messages: Vec<wire::ChatMessage<'a>>,
    /// The closing user message, naming the stretch to summarize.
    pub instruction: String,
}

/// Condenses a stretch of the conversation into a summary when it outgrows
/// its token budget.
#[async_trait]
pub trait Summarizer: Send + Sync {
    /// The model's reply to `request`: the summary text alone.
    async fn summarize(&self, request: SummaryRequest<'_>) -> Result<String, ChatClientError>;
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
    /// Tokens compaction has removed since the message list was last replaced.
    compacted_tokens: u32,
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
            compacted_tokens: 0,
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
        self.compacted_tokens = 0;
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

    /// Tokens the conversation has held in all: its current size plus
    /// everything compaction has removed from it.
    pub fn total_tokens(&self) -> u32 {
        self.approx_total_tokens()
            .saturating_add(self.compacted_tokens)
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
        total.saturating_add(messages_tokens(&self.messages))
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

    /// Keep the approximate token total under budget by folding older
    /// history into one summary: the oldest steps of the current turn when
    /// that turn alone outgrows the budget, else the turns before the
    /// preserved ones. Returns an error, with history unchanged, if the
    /// summarizer fails or returns empty text; [`Self::truncate_to_budget`]
    /// is the infallible fallback.
    pub async fn ensure_budget(
        &mut self,
        summarizer: &dyn Summarizer,
        chat: &ChatConfig,
        model: &ModelConfig,
    ) -> Result<(), ChatClientError> {
        if !self.exceeds_budget(chat, model) {
            return Ok(());
        }
        let fold_end = self.fold_end(chat, model);
        if fold_end <= self.summary_insertion_index() {
            debug!(
                target: "assistd::chat",
                "ensure_budget: nothing to summarize, falling through to truncation"
            );
            self.truncate_to_budget(chat, model);
            return Ok(());
        }

        let request = SummaryRequest {
            messages: self.as_wire_messages(),
            instruction: self.summary_instruction(fold_end, chat.summary_target_tokens.get()),
        };
        let summary = summarize_bounded(summarizer, request, chat).await?;
        self.fold_into_summary(fold_end, &summary);

        if self.exceeds_budget(chat, model) {
            debug!(
                target: "assistd::chat",
                "ensure_budget: still over budget after summarize, truncating"
            );
            self.truncate_to_budget(chat, model);
        }
        Ok(())
    }

    /// The instruction asking for a summary of everything before `fold_end`,
    /// naming the first message kept word for word by quoting it.
    fn summary_instruction(&self, fold_end: usize, target_tokens: u32) -> String {
        let scope = match self.messages.get(fold_end) {
            Some(step) if fold_end > self.current_turn_start() => format!(
                "Cover everything before your tool call {}, including any earlier conversation \
                 summary and the steps already taken for the user's current request. The current \
                 request, that call, and everything after it stay in context word for word.",
                quote_first_call(step)
            ),
            Some(kept) => format!(
                "Cover everything before the {} message containing \"{}\", including any earlier \
                 conversation summary. That message and everything after it stay in context word \
                 for word.",
                kept.role.as_wire(),
                quote(&kept.content)
            ),
            None => "Cover the whole conversation, including any earlier conversation summary."
                .to_string(),
        };
        format!(
            "{SUMMARY_INSTRUCTION_HEAD} {scope} {SUMMARY_INSTRUCTION_RULES} Keep it under about \
             {} tokens.",
            target_tokens.saturating_mul(SUMMARY_ASK_PERCENT) / 100
        )
    }

    /// End of the stretch to summarize. Past the current turn's question
    /// only when that turn outgrows the budget even with all earlier
    /// history summarized.
    fn fold_end(&self, chat: &ChatConfig, model: &ModelConfig) -> usize {
        let preserved = self.first_preserved_index(chat.preserve_recent_turns.get() as usize);
        let turn_start = self.current_turn_start();
        let turn_tokens = u64::from(messages_tokens(&self.messages[turn_start..]));
        let fixed = u64::from(self.fixed_tokens()) + u64::from(chat.summary_target_tokens.get());
        if fixed + turn_tokens <= u64::from(effective_budget(chat, model)) {
            return preserved;
        }
        self.first_kept_step(chat, model).unwrap_or(preserved)
    }

    /// Oldest tool step of the current turn that, with every newer one,
    /// fits under the compaction target beside the summary; `None` unless
    /// at least one older step would be folded. The newest step is kept.
    fn first_kept_step(&self, chat: &ChatConfig, model: &ModelConfig) -> Option<usize> {
        let turn_start = self.current_turn_start();
        let question = self.messages.get(turn_start)?;
        let reserved = u64::from(self.fixed_tokens())
            + u64::from(chat.summary_target_tokens.get())
            + u64::from(approx_message_tokens(question));
        let allowance =
            budget_share(chat, model, COMPACTION_TARGET_PERCENT).saturating_sub(reserved);
        let mut kept = None;
        for start in (turn_start + 1..self.messages.len()).rev() {
            if !is_tool_step(&self.messages[start]) {
                continue;
            }
            if kept.is_some() && u64::from(messages_tokens(&self.messages[start..])) > allowance {
                break;
            }
            kept = Some(start);
        }
        kept.filter(|&start| start > turn_start + 1)
    }

    /// Replace everything before `fold_end` with `summary`, keeping the
    /// current turn's question and logging any of its steps folded away.
    fn fold_into_summary(&mut self, fold_end: usize, summary: &str) {
        let before = self.approx_total_tokens();
        let turn_start = self.current_turn_start();
        if fold_end > turn_start + 1 {
            self.fold_steps_into_log(turn_start + 1..fold_end);
        }
        self.messages.splice(
            ..fold_end.min(turn_start),
            [Message::text(
                Role::System,
                format!("{SUMMARY_PREFIX}{summary}"),
            )],
        );
        self.record_compaction(before);
    }

    fn record_compaction(&mut self, tokens_before: u32) {
        let removed = tokens_before.saturating_sub(self.approx_total_tokens());
        self.compacted_tokens = self.compacted_tokens.saturating_add(removed);
    }

    /// Remove `steps` from the current turn, recording their calls on its
    /// question.
    fn fold_steps_into_log(&mut self, steps: Range<usize>) {
        let records = step_log::records(&self.messages[steps.clone()]);
        self.messages.drain(steps);
        let turn_start = self.current_turn_start();
        if let Some(question) = self.messages.get_mut(turn_start) {
            question.earlier_steps.extend(records);
            let excess = question
                .earlier_steps
                .len()
                .saturating_sub(step_log::MAX_LOGGED_CALLS);
            question.earlier_steps.drain(..excess);
        }
    }

    /// Tokens outside the message list: system prompt, transient text, and
    /// the measured calibration.
    fn fixed_tokens(&self) -> u32 {
        self.approx_total_tokens()
            .saturating_sub(messages_tokens(&self.messages))
    }

    pub fn exceeds_budget(&self, chat: &ChatConfig, model: &ModelConfig) -> bool {
        self.approx_total_tokens() > effective_budget(chat, model)
    }

    /// Tokens history can still grow before compaction is near; zero once
    /// it is.
    pub fn compaction_warning_headroom(&self, chat: &ChatConfig, model: &ModelConfig) -> u32 {
        let warning_at = budget_share(chat, model, COMPACTION_WARNING_PERCENT);
        let headroom = warning_at.saturating_sub(u64::from(self.approx_total_tokens()));
        u32::try_from(headroom).unwrap_or(u32::MAX)
    }

    /// Drop the oldest messages after any summary, then fold the oldest tool
    /// steps of the current turn into its call log, until the conversation is
    /// under a low-water mark well below the budget. The latest user message
    /// and the newest step always stay.
    pub fn truncate_to_budget(&mut self, chat: &ChatConfig, model: &ModelConfig) {
        let before = self.approx_total_tokens();
        let target = budget_share(chat, model, COMPACTION_TARGET_PERCENT);
        while u64::from(self.approx_total_tokens()) > target {
            let Some(idx) = self.first_droppable_index() else {
                warn!(
                    target: "assistd::chat",
                    "truncate_to_budget: cannot drop any more messages without losing the current turn"
                );
                break;
            };
            let pair = self.pair_range(idx);
            if idx > self.current_turn_start() {
                self.fold_steps_into_log(pair);
            } else {
                self.messages.drain(pair);
            }
        }
        self.record_compaction(before);
    }

    /// `idx` plus, when it is a tool step, the results that follow it.
    fn pair_range(&self, idx: usize) -> Range<usize> {
        let results = if is_tool_step(&self.messages[idx]) {
            self.messages[idx + 1..]
                .iter()
                .take_while(|m| Self::is_tool_result(m))
                .count()
        } else {
            0
        };
        idx..idx + 1 + results
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
        let oldest = turn_start + 1;
        let newest = self.messages.iter().rposition(is_tool_step)?;
        (self.messages.get(oldest).is_some_and(is_tool_step) && oldest < newest).then_some(oldest)
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

fn is_tool_step(m: &Message) -> bool {
    m.role == Role::Assistant && !m.tool_calls.is_empty()
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
    let step_log_cost = approx_tokens_bytes(step_log::rendered_len(&m.earlier_steps));
    TOKENS_PER_MESSAGE_OVERHEAD
        .saturating_add(approx_tokens(&m.content))
        .saturating_add(approx_tokens(&m.reasoning))
        .saturating_add(image_cost)
        .saturating_add(tool_call_cost)
        .saturating_add(context_cost)
        .saturating_add(step_log_cost)
}

fn messages_tokens(messages: &[Message]) -> u32 {
    messages
        .iter()
        .fold(0, |total, m| total.saturating_add(approx_message_tokens(m)))
}

pub(super) fn approx_tokens_bytes(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX).saturating_add(3) / 4
}

fn wire_message(message: &Message) -> wire::ChatMessage<'_> {
    if !message.tool_calls.is_empty() {
        return tool_calls_message(message);
    }
    if message.role == Role::Tool {
        let content = wire::ContentBody::Text(neutralise_template_tokens(&message.content));
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
    let text = text_with_context(message);
    if message.earlier_steps.is_empty() {
        return text;
    }
    let log = step_log::render(&message.earlier_steps);
    Cow::Owned(format!("{text}\n\n{}", neutralise_template_tokens(&log)))
}

fn text_with_context(message: &Message) -> Cow<'_, str> {
    match &message.context {
        Some(ctx) => {
            let ctx = neutralise_context_markers(ctx.trim_end());
            Cow::Owned(format!(
                "{CONTEXT_OPEN}{}{CONTEXT_CLOSE}{}",
                neutralise_template_tokens(&ctx),
                message.content
            ))
        }
        None if Conversation::is_tool_result(message) => {
            neutralise_template_tokens(&message.content)
        }
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

fn neutralise_template_tokens(text: &str) -> Cow<'_, str> {
    if !TEMPLATE_TOKEN_DEFANGS
        .iter()
        .any(|(token, _)| text.contains(token))
    {
        return Cow::Borrowed(text);
    }
    Cow::Owned(
        TEMPLATE_TOKEN_DEFANGS
            .iter()
            .fold(text.to_string(), |acc, (token, inert)| {
                acc.replace(token, inert)
            }),
    )
}

fn encode_all(attachments: Vec<Attachment>) -> Vec<ImageDataUri> {
    attachments.into_iter().map(ImageDataUri::encode).collect()
}

fn effective_budget(chat: &ChatConfig, model: &ModelConfig) -> u32 {
    chat.max_history_tokens.get().min(model.context_budget())
}

fn budget_share(chat: &ChatConfig, model: &ModelConfig, percent: u64) -> u64 {
    u64::from(effective_budget(chat, model)) * percent / 100
}

/// The opening of `text`'s first non-empty line, defanged, for naming a
/// message inside the summary instruction.
fn quote(text: &str) -> String {
    let line = text
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("");
    truncate_utf8(&neutralise_template_tokens(line), SUMMARY_QUOTE_BYTES)
}

/// `step`'s first tool call as the instruction names it.
fn quote_first_call(step: &Message) -> String {
    step.tool_calls.first().map_or_else(String::new, |call| {
        format!(
            "`{}` with arguments `{}`",
            call.name,
            quote(&call.arguments)
        )
    })
}

/// The summary `request` yields, cut to about `summary_target_tokens`.
async fn summarize_bounded(
    summarizer: &dyn Summarizer,
    request: SummaryRequest<'_>,
    chat: &ChatConfig,
) -> Result<String, ChatClientError> {
    let summary = summarizer.summarize(request).await?;
    let trimmed = summary.trim();
    if trimmed.is_empty() {
        return Err(ChatClientError::Summarize(
            "summarizer returned empty text".into(),
        ));
    }
    let max_summary_bytes = (chat.summary_target_tokens.get() as usize).saturating_mul(4);
    Ok(truncate_utf8(trimmed, max_summary_bytes))
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
