//! Multi-turn conversation state with token budgeting: a bytes-per-token
//! heuristic, corrected by the prompt size the server last measured.

use std::borrow::Cow;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;

use assistd_config::{ChatConfig, ModelConfig};
use assistd_tools::Attachment;
use async_trait::async_trait;
use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
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
    unsupported, hosts that are unreachable, paths that do not exist), so none is retried. If you \
    have the `reminisce` tool, the summarized messages stay searchable with it, so favour state \
    and decisions over verbatim detail. Write with no preamble or commentary.";
/// Longest quote of a message the summary instruction uses to name it.
const SUMMARY_QUOTE_BYTES: usize = 120;
/// Delimiters around the tool name that heads an image-carrying tool result.
const TOOL_RESULT_OPEN: &str = "[tool:";
const TOOL_RESULT_CLOSE: &str = "]\n";
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
/// Recent user/assistant exchanges kept verbatim when summarizing.
const PRESERVE_RECENT_TURNS: usize = 6;

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

/// An image encoded once, on entry, as the `data:` URI it is sent as;
/// clones share the encoded text.
#[derive(Debug, Clone)]
pub struct ImageDataUri(Arc<str>);

impl ImageDataUri {
    fn encode(attachment: Attachment) -> Self {
        match attachment {
            Attachment::Image { mime, bytes } => {
                let mut uri = String::with_capacity(
                    "data:;base64,".len() + mime.as_str().len() + bytes.len().div_ceil(3) * 4,
                );
                uri.push_str("data:");
                uri.push_str(mime.as_str());
                uri.push_str(";base64,");
                B64.encode_string(bytes, &mut uri);
                Self(uri.into())
            }
        }
    }

    /// The URI text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// One entry in the in-memory conversation, owned by [`Conversation`].
#[derive(Debug, Clone)]
pub enum Message {
    /// Compaction summary of the history before it, without its prefix.
    Summary(String),
    /// A turn the user wrote, which opens a new tool loop.
    User(UserMessage),
    Assistant(AssistantMessage),
    /// Text output answering the assistant tool call `call_id`.
    ToolResult {
        call_id: String,
        content: String,
    },
    /// Image-carrying tool output. Sent on the user role, the only one
    /// templates render images on, without closing its tool loop.
    ToolImageResult {
        tool_name: String,
        content: String,
        attachments: Vec<ImageDataUri>,
    },
}

impl Message {
    fn role(&self) -> Role {
        match self {
            Message::Summary(_) => Role::System,
            Message::User(_) | Message::ToolImageResult { .. } => Role::User,
            Message::Assistant(_) => Role::Assistant,
            Message::ToolResult { .. } => Role::Tool,
        }
    }

    fn content(&self) -> &str {
        match self {
            Message::Summary(content)
            | Message::ToolResult { content, .. }
            | Message::ToolImageResult { content, .. } => content,
            Message::User(user) => &user.content,
            Message::Assistant(assistant) => &assistant.content,
        }
    }

    fn tool_calls(&self) -> &[ToolCallRecord] {
        match self {
            Message::Assistant(assistant) => &assistant.tool_calls,
            _ => &[],
        }
    }

    fn is_tool_result(&self) -> bool {
        matches!(
            self,
            Message::ToolResult { .. } | Message::ToolImageResult { .. }
        )
    }
}

/// A user turn's text and what is sent along with it.
#[derive(Debug, Clone, Default)]
pub struct UserMessage {
    pub content: String,
    pub attachments: Vec<ImageDataUri>,
    /// Context block rendered at the head of the text, keeping earlier
    /// turns byte-identical for the prefix cache.
    pub context: Option<String>,
    /// Tool calls already folded out of this turn, rendered after its text.
    pub earlier_steps: Vec<StepRecord>,
}

/// An assistant reply, or a tool step when `tool_calls` is non-empty.
#[derive(Debug, Clone, Default)]
pub struct AssistantMessage {
    pub content: String,
    /// Reasoning behind `tool_calls`, sent back with them.
    pub reasoning: String,
    pub tool_calls: Vec<ToolCallRecord>,
}

/// What a compaction summary leaves of the conversation: its `kept_turns`
/// newest user turns, the oldest of them cut to its `kept_steps` newest tool
/// steps when its earlier steps were folded into its call log.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactionCut {
    pub kept_turns: NonZeroUsize,
    pub kept_steps: Option<NonZeroUsize>,
}

/// A summary to write by continuing the conversation itself, so the server
/// can reuse the prompt it already holds and read only the instruction.
#[derive(Debug)]
pub struct SummaryRequest<'a> {
    /// The conversation as the next step would render it, except that the
    /// turns being folded lose their context blocks.
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
/// only turns plus at most one [`Message::Summary`], at index 0; the
/// rendered form carries a single system message, at its head.
#[derive(Debug, Clone)]
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
    /// Bumped whenever existing messages are removed or rewritten; appends
    /// leave it alone, so a fold planned before them still applies.
    revision: u64,
}

/// A summary fold planned on a snapshot of the conversation, applied once
/// the summary is written if the history has not been rewritten meanwhile.
struct CompactionPlan {
    snapshot: Conversation,
    instruction: String,
    fold: Fold,
    revision: u64,
}

/// Everything before `end` folds into the summary, except the current
/// turn's question at `turn_start`.
#[derive(Debug, Clone, Copy)]
struct Fold {
    end: usize,
    turn_start: usize,
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
            revision: 0,
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
        self.messages.push(Message::User(UserMessage {
            content,
            attachments: encode_all(attachments),
            context: self.pending_context.take(),
            earlier_steps: Vec::new(),
        }));
    }

    /// Append the output of the tool `name` together with the images it
    /// produced, as a [`Message::ToolImageResult`].
    pub fn push_tool_result_with_attachments(
        &mut self,
        name: String,
        content: String,
        attachments: Vec<Attachment>,
    ) {
        self.messages.push(Message::ToolImageResult {
            tool_name: name,
            content,
            attachments: encode_all(attachments),
        });
    }

    /// Appends a plain-text assistant turn.
    pub fn push_assistant(&mut self, content: String) {
        self.messages.push(Message::Assistant(AssistantMessage {
            content,
            ..AssistantMessage::default()
        }));
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
        self.messages.push(Message::Assistant(AssistantMessage {
            content: content.unwrap_or_default(),
            reasoning,
            tool_calls: calls,
        }));
    }

    /// Append the output of one tool call as a `role: "tool"` message.
    pub fn push_tool_result(&mut self, call_id: String, content: String) {
        self.messages.push(Message::ToolResult { call_id, content });
    }

    /// Drop the most recent message if it is a real user message (not a
    /// tool result), returning its context block to pending so a retried
    /// turn still carries it.
    pub fn rollback_last_user(&mut self) {
        let last_user = self.messages.pop_if(|message| is_user_turn(message));
        if let Some(Message::User(user)) = last_user {
            self.pending_context = user.context;
            self.revision += 1;
        }
    }

    /// Replace the message list wholesale, headed by `summary` when given,
    /// and clear any pending context or note. `folded_steps`, tool steps
    /// already folded out of the first message's turn, become its call log.
    pub fn replace_messages(
        &mut self,
        summary: Option<&str>,
        folded_steps: &[Message],
        mut msgs: Vec<Message>,
    ) {
        if !folded_steps.is_empty()
            && let Some(Message::User(question)) = msgs.first_mut()
        {
            step_log::append(&mut question.earlier_steps, step_log::records(folded_steps));
        }
        self.messages = summary
            .map(|text| Message::Summary(text.to_owned()))
            .into_iter()
            .chain(msgs)
            .collect();
        self.pending_context = None;
        self.transient_note = None;
        self.calibration = 0;
        self.compacted_tokens = 0;
        self.revision += 1;
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
        self.revision += 1;
        removed
    }

    /// The text of the newest `limit` real user messages, oldest first.
    pub fn recent_user_messages(&self, limit: usize) -> Vec<String> {
        let mut texts: Vec<String> = self
            .messages
            .iter()
            .rev()
            .filter_map(|message| match message {
                Message::User(user) => Some(user.content.clone()),
                _ => None,
            })
            .take(limit)
            .collect();
        texts.reverse();
        texts
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

    /// The current summary text and what it leaves in place, or `None` when
    /// there is no summary or no turn after it.
    pub fn summary_snapshot(&self) -> Option<(&str, CompactionCut)> {
        let summary = self.summary()?;
        let kept = &self.messages[self.summary_insertion_index()..];
        let kept_turns = kept.iter().filter(|m| is_user_turn(m)).count();
        let cut = CompactionCut {
            kept_turns: NonZeroUsize::new(kept_turns)?,
            kept_steps: Self::kept_steps_of_folded_turn(kept),
        };
        Some((summary, cut))
    }

    /// Tool steps left in the turn opening `kept` when that turn's earlier
    /// steps were folded into its call log.
    fn kept_steps_of_folded_turn(kept: &[Message]) -> Option<NonZeroUsize> {
        let (Message::User(question), rest) = kept.split_first()? else {
            return None;
        };
        if question.earlier_steps.is_empty() {
            return None;
        }
        let steps = rest
            .iter()
            .take_while(|m| !is_user_turn(m))
            .filter(|m| is_tool_step(m))
            .count();
        NonZeroUsize::new(steps)
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
        out.extend(self.system_head_message());
        out.extend(self.messages[turns_start..].iter().map(wire_message));
        if let Some(note) = &self.transient_note {
            let content = wire::ContentBody::Text(Cow::Borrowed(note.as_str()));
            out.push(wire::ChatMessage::plain(Role::User.as_wire(), content));
        }
        out
    }

    /// The system prompt with any summary appended, as one message.
    fn system_head_message(&self) -> Option<wire::ChatMessage<'_>> {
        let summary = self.summary().map(summary_text);
        let head = match (self.system_prompt.as_str(), summary) {
            ("", None) => return None,
            ("", Some(summary)) => Cow::Owned(summary),
            (prompt, None) => Cow::Borrowed(prompt),
            (prompt, Some(summary)) => Cow::Owned(format!("{prompt}\n\n{summary}")),
        };
        Some(wire::ChatMessage::plain(
            Role::System.as_wire(),
            wire::ContentBody::Text(head),
        ))
    }

    /// Plan the summary fold that brings the conversation under budget, or
    /// truncate in place and return `None` when nothing can be summarized.
    /// `None` too when already under budget.
    fn plan_compaction(
        &mut self,
        chat: &ChatConfig,
        model: &ModelConfig,
    ) -> Option<CompactionPlan> {
        if !self.exceeds_budget(chat, model) {
            return None;
        }
        let fold = Fold {
            end: self.fold_end(chat, model),
            turn_start: self.current_turn_start(),
        };
        if fold.end <= self.summary_insertion_index() {
            debug!(
                target: "assistd::chat",
                "ensure_budget: nothing to summarize, falling through to truncation"
            );
            self.truncate_to_budget(chat, model);
            return None;
        }
        Some(CompactionPlan {
            instruction: self.summary_instruction(fold.end, chat.summary_target_tokens.get()),
            fold,
            revision: self.revision,
            snapshot: self.without_folded_context(fold),
        })
    }

    /// A copy whose turns covered by `fold` carry no context block, so what
    /// was injected for those moments is not summarized as conversation fact.
    fn without_folded_context(&self, fold: Fold) -> Self {
        let mut snapshot = self.clone();
        for message in &mut snapshot.messages[..fold.end.min(fold.turn_start)] {
            if let Message::User(user) = message {
                user.context = None;
            }
        }
        snapshot
    }

    /// Fold `summary` in as planned at `revision`, discarding it when the
    /// history was rewritten since, then truncate if still over budget.
    fn apply_compaction(
        &mut self,
        fold: Fold,
        revision: u64,
        summary: &str,
        chat: &ChatConfig,
        model: &ModelConfig,
    ) {
        if revision == self.revision {
            self.fold_into_summary(fold, summary);
        } else {
            warn!(
                target: "assistd::chat",
                "ensure_budget: history was rewritten while summarizing, discarding the summary"
            );
        }
        if self.exceeds_budget(chat, model) {
            debug!(
                target: "assistd::chat",
                "ensure_budget: still over budget after summarize, truncating"
            );
            self.truncate_to_budget(chat, model);
        }
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
                kept.role().as_wire(),
                quote(kept.content())
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
        let preserved = self.first_preserved_index(PRESERVE_RECENT_TURNS);
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

    /// Replace everything `fold` covers with `summary`, keeping the turn's
    /// question and logging any of its steps folded away.
    fn fold_into_summary(&mut self, fold: Fold, summary: &str) {
        let before = self.approx_total_tokens();
        if fold.end > fold.turn_start + 1 {
            self.fold_steps_into_log(fold.turn_start, fold.turn_start + 1..fold.end);
        }
        self.messages.splice(
            ..fold.end.min(fold.turn_start),
            [Message::Summary(summary.to_owned())],
        );
        self.record_compaction(before);
    }

    fn record_compaction(&mut self, tokens_before: u32) {
        let removed = tokens_before.saturating_sub(self.approx_total_tokens());
        self.compacted_tokens = self.compacted_tokens.saturating_add(removed);
        self.revision += 1;
    }

    /// Remove `steps` from the turn whose question is at `turn_start`,
    /// recording their calls on that question.
    fn fold_steps_into_log(&mut self, turn_start: usize, steps: Range<usize>) {
        let records = step_log::records(&self.messages[steps.clone()]);
        self.messages.drain(steps);
        if let Some(Message::User(question)) = self.messages.get_mut(turn_start) {
            step_log::append(&mut question.earlier_steps, records);
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

    /// Drop the oldest messages after any summary, then fold the current turn's oldest tool steps
    /// into its call log, until under a low-water mark below the budget. The latest user message
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
            let turn_start = self.current_turn_start();
            if idx > turn_start {
                self.fold_steps_into_log(turn_start, pair);
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
                .take_while(|m| m.is_tool_result())
                .count()
        } else {
            0
        };
        idx..idx + 1 + results
    }

    fn summary(&self) -> Option<&str> {
        match self.messages.first() {
            Some(Message::Summary(text)) => Some(text),
            _ => None,
        }
    }

    fn summary_insertion_index(&self) -> usize {
        usize::from(self.summary().is_some())
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
            match self.messages[prev] {
                Message::User(_) | Message::ToolImageResult { .. } => {
                    pairs_seen += 1;
                    idx = prev;
                }
                Message::Assistant(_) => {
                    idx = if prev > start { prev - 1 } else { prev };
                    pairs_seen += 1;
                }
                Message::Summary(_) | Message::ToolResult { .. } => {
                    idx = prev;
                }
            }
        }
        while idx > start && self.messages.get(idx).is_some_and(Message::is_tool_result) {
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
        self.messages.iter().rposition(is_user_turn)
    }

    fn current_turn_start(&self) -> usize {
        self.last_real_user_index().unwrap_or(self.messages.len())
    }
}

fn is_user_turn(m: &Message) -> bool {
    matches!(m, Message::User(_))
}

fn is_tool_step(m: &Message) -> bool {
    !m.tool_calls().is_empty()
}

fn approx_tokens(text: &str) -> u32 {
    u32::try_from(text.len()).unwrap_or(u32::MAX).div_ceil(4)
}

fn approx_message_tokens(message: &Message) -> u32 {
    let body = match message {
        Message::Summary(text) => approx_tokens_bytes(SUMMARY_PREFIX.len() + text.len()),
        Message::User(user) => approx_user_tokens(user),
        Message::Assistant(assistant) => approx_assistant_tokens(assistant),
        Message::ToolResult { content, .. } => approx_tokens(content),
        Message::ToolImageResult {
            tool_name,
            content,
            attachments,
        } => approx_tokens_bytes(
            TOOL_RESULT_OPEN.len() + tool_name.len() + TOOL_RESULT_CLOSE.len() + content.len(),
        )
        .saturating_add(approx_image_tokens(attachments)),
    };
    TOKENS_PER_MESSAGE_OVERHEAD.saturating_add(body)
}

fn approx_user_tokens(user: &UserMessage) -> u32 {
    let context_cost = user.context.as_deref().map_or(0, |ctx| {
        approx_tokens_bytes(CONTEXT_OPEN.len() + ctx.len() + CONTEXT_CLOSE.len())
    });
    let step_log_cost = approx_tokens_bytes(step_log::rendered_len(&user.earlier_steps));
    approx_tokens(&user.content)
        .saturating_add(approx_image_tokens(&user.attachments))
        .saturating_add(context_cost)
        .saturating_add(step_log_cost)
}

fn approx_assistant_tokens(assistant: &AssistantMessage) -> u32 {
    let tool_call_bytes: usize = assistant
        .tool_calls
        .iter()
        .map(|c| c.id.len() + c.name.len() + c.arguments.len() + 32)
        .sum();
    approx_tokens(&assistant.content)
        .saturating_add(approx_tokens(&assistant.reasoning))
        .saturating_add(approx_tokens_bytes(tool_call_bytes))
}

fn approx_image_tokens(attachments: &[ImageDataUri]) -> u32 {
    u32::try_from(attachments.len())
        .unwrap_or(u32::MAX)
        .saturating_mul(TOKENS_PER_IMAGE)
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
    let role = message.role().as_wire();
    match message {
        Message::Summary(text) => wire::ChatMessage::plain(
            role,
            wire::ContentBody::Text(Cow::Owned(summary_text(text))),
        ),
        Message::User(user) => {
            wire::ChatMessage::plain(role, content_body(wire_text(user), &user.attachments))
        }
        Message::Assistant(step) if !step.tool_calls.is_empty() => tool_calls_message(step),
        Message::Assistant(reply) => wire::ChatMessage::plain(role, defanged(&reply.content)),
        Message::ToolResult { call_id, content } => wire::ChatMessage {
            tool_call_id: Some(call_id),
            ..wire::ChatMessage::plain(role, defanged(content))
        },
        Message::ToolImageResult {
            tool_name,
            content,
            attachments,
        } => wire::ChatMessage::plain(
            role,
            content_body(
                Cow::Owned(image_result_text(tool_name, content)),
                attachments,
            ),
        ),
    }
}

fn defanged(text: &str) -> wire::ContentBody<'_> {
    wire::ContentBody::Text(neutralise_template_tokens(text))
}

fn summary_text(summary: &str) -> String {
    format!("{SUMMARY_PREFIX}{}", neutralise_template_tokens(summary))
}

fn image_result_text(tool_name: &str, content: &str) -> String {
    format!(
        "{TOOL_RESULT_OPEN}{}{TOOL_RESULT_CLOSE}{}",
        neutralise_template_tokens(tool_name),
        neutralise_template_tokens(content)
    )
}

fn tool_calls_message(message: &AssistantMessage) -> wire::ChatMessage<'_> {
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
        role: Role::Assistant.as_wire(),
        content: (!message.content.is_empty()).then(|| defanged(&message.content)),
        tool_calls: Some(specs),
        tool_call_id: None,
        reasoning_content: (!message.reasoning.is_empty())
            .then(|| neutralise_template_tokens(&message.reasoning)),
    }
}

fn content_body<'a>(text: Cow<'a, str>, attachments: &'a [ImageDataUri]) -> wire::ContentBody<'a> {
    if attachments.is_empty() {
        return wire::ContentBody::Text(text);
    }
    let mut parts = Vec::with_capacity(attachments.len() + 1);
    parts.push(wire::ContentPart::Text { text });
    parts.extend(attachments.iter().map(|image| wire::ContentPart::ImageUrl {
        image_url: wire::ImageUrl {
            url: image.as_str(),
        },
    }));
    wire::ContentBody::Parts(parts)
}

fn wire_text(user: &UserMessage) -> Cow<'_, str> {
    let text = text_with_context(user);
    if user.earlier_steps.is_empty() {
        return text;
    }
    let log = step_log::render(&user.earlier_steps);
    Cow::Owned(format!("{text}\n\n{}", neutralise_template_tokens(&log)))
}

fn text_with_context(user: &UserMessage) -> Cow<'_, str> {
    match &user.context {
        Some(ctx) => {
            let ctx = neutralise_context_markers(ctx.trim_end());
            Cow::Owned(format!(
                "{CONTEXT_OPEN}{}{CONTEXT_CLOSE}{}",
                neutralise_template_tokens(&ctx),
                user.content
            ))
        }
        None => Cow::Borrowed(&user.content),
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

pub(super) fn neutralise_template_tokens(text: &str) -> Cow<'_, str> {
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

pub(super) fn effective_budget(chat: &ChatConfig, model: &ModelConfig) -> u32 {
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
    step.tool_calls().first().map_or_else(String::new, |call| {
        format!(
            "`{}` with arguments `{}`",
            call.name,
            quote(&call.arguments)
        )
    })
}

/// Keep `conv` under budget by folding older history into one summary,
/// releasing the lock while the summarizer runs. Returns an error, with
/// history unchanged, if the summarizer fails or returns empty text.
pub async fn ensure_budget(
    conv: &Mutex<Conversation>,
    summarizer: &dyn Summarizer,
    chat: &ChatConfig,
    model: &ModelConfig,
) -> Result<(), ChatClientError> {
    let Some(plan) = conv.lock().await.plan_compaction(chat, model) else {
        return Ok(());
    };
    let CompactionPlan {
        snapshot,
        instruction,
        fold,
        revision,
    } = plan;
    let request = SummaryRequest {
        messages: snapshot.as_wire_messages(),
        instruction,
    };
    let summary = summarize_bounded(summarizer, request, chat).await?;
    conv.lock()
        .await
        .apply_compaction(fold, revision, &summary, chat, model);
    Ok(())
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
