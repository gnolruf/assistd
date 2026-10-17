//! Session transcripts: sessions, turns, branches, and their messages.

use std::fmt;
use std::num::NonZeroUsize;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::{BranchId, ChunkId, MessageId, Result, Seq, SessionId, TurnId};

/// Role of a persisted message, stored as the `messages.role` code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PersistedRole {
    User,
    Assistant,
    Tool,
}

impl PersistedRole {
    /// The lowercase role name.
    pub fn as_wire(self) -> &'static str {
        match self {
            PersistedRole::User => "user",
            PersistedRole::Assistant => "assistant",
            PersistedRole::Tool => "tool",
        }
    }
}

/// One message as stored in the `messages` table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistedMessage {
    pub content: String,
    pub kind: MessageKind,
}

/// The role of a [`PersistedMessage`] together with the fields only that role carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MessageKind {
    User,
    Assistant {
        /// The model's reasoning for this step; `None` when it gave none.
        reasoning: Option<Reasoning>,
        /// The tools this step called, in call order; empty for a plain reply.
        tool_calls: Vec<PersistedToolCall>,
    },
    Tool {
        call_id: String,
        tool_name: String,
        outcome: ToolOutcome,
    },
}

impl PersistedMessage {
    /// A user prompt.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            kind: MessageKind::User,
        }
    }

    /// An assistant reply with no tool calls.
    pub fn assistant_text(content: impl Into<String>, reasoning: Option<Reasoning>) -> Self {
        Self::assistant(content.into(), reasoning, Vec::new())
    }

    /// A model step that requested tools; `content` is any accompanying text,
    /// possibly empty.
    pub fn assistant_tool_calls(
        content: impl Into<String>,
        calls: Vec<PersistedToolCall>,
        reasoning: Option<Reasoning>,
    ) -> Self {
        Self::assistant(content.into(), reasoning, calls)
    }

    /// The output of tool call `call_id`, which invoked tool `name`.
    pub fn tool_result(
        content: impl Into<String>,
        call_id: impl Into<String>,
        name: impl Into<String>,
        outcome: ToolOutcome,
    ) -> Self {
        Self {
            content: content.into(),
            kind: MessageKind::Tool {
                call_id: call_id.into(),
                tool_name: name.into(),
                outcome,
            },
        }
    }

    /// The role this message is stored under.
    pub fn role(&self) -> PersistedRole {
        match self.kind {
            MessageKind::User => PersistedRole::User,
            MessageKind::Assistant { .. } => PersistedRole::Assistant,
            MessageKind::Tool { .. } => PersistedRole::Tool,
        }
    }

    fn assistant(
        content: String,
        reasoning: Option<Reasoning>,
        tool_calls: Vec<PersistedToolCall>,
    ) -> Self {
        Self {
            content,
            kind: MessageKind::Assistant {
                reasoning,
                tool_calls,
            },
        }
    }
}

/// The reasoning an assistant step streamed before answering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reasoning {
    pub text: String,
    /// Time from the first reasoning token to the last.
    pub duration_ms: u64,
}

/// One tool call requested by an assistant step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PersistedToolCall {
    /// Server-assigned call id, matched by the [`MessageKind::Tool`] result.
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// How one tool run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToolOutcome {
    pub exit_code: i64,
    pub duration_ms: u64,
}

/// Per-branch metadata; `is_current_in_session` marks the session's current branch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BranchInfo {
    pub branch_id: BranchId,
    pub session_id: SessionId,
    pub session_started_at: DateTime<Utc>,
    pub session_ended_at: Option<DateTime<Utc>>,
    pub session_title: Option<String>,
    pub name: String,
    pub parent_branch_id: Option<BranchId>,
    pub parent_branch_name: Option<String>,
    pub fork_point_seq: Option<Seq>,
    pub created_at: DateTime<Utc>,
    pub message_count: i64,
    pub is_current_in_session: bool,
    /// The later of the branch's creation and its newest message.
    pub last_active_at: DateTime<Utc>,
}

/// One persisted message on a branch, as replayed into the in-memory conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoryRow {
    pub message_id: MessageId,
    pub seq: Seq,
    pub message: PersistedMessage,
    /// Set on the last message of its turn on the branch once that turn finished.
    pub turn_stats: Option<TurnStats>,
}

/// How long a finished turn ran and the tokens it produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TurnStats {
    pub duration_ms: u64,
    /// Tokens the turn added to the conversation.
    pub generated_tokens: u64,
}

/// The latest compaction summary of a branch. It stands in for every message
/// before `kept_from_seq` and for those after it through `steps_folded_through`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BranchSummary {
    /// Seq of the user message that opens the history kept word for word.
    pub kept_from_seq: Seq,
    /// Last seq of that user message's turn whose tool steps were folded into
    /// its call log; `None` when its turn is kept whole.
    pub steps_folded_through: Option<Seq>,
    /// Seq of the newest message on the branch when the compaction ran.
    pub compacted_after_seq: Seq,
    pub content: String,
    /// Time the compaction that wrote it took.
    pub duration_ms: u64,
}

/// What a compaction summary leaves of a branch: its `kept_turns` newest user
/// turns, the oldest of them cut to its `kept_steps` newest tool steps when set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SummaryCut {
    pub kept_turns: NonZeroUsize,
    pub kept_steps: Option<NonZeroUsize>,
}

/// Result of [`TranscriptStore::append_message_to_branch`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppendedMessage {
    pub id: MessageId,
    /// The chunks stored for embedding, as `(id, chunk text)`; empty for tool results.
    pub chunks: Vec<(ChunkId, String)>,
}

/// Result of [`TranscriptStore::begin_turn`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenedTurn {
    pub turn: TurnId,
    /// The turn's first message.
    pub message: AppendedMessage,
}

/// Result of [`TranscriptStore::undo_last_turn`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UndoOutcome {
    /// Number of `branch_messages` rows dropped.
    pub removed_messages: usize,
    pub last_user_text: Option<String>,
}

/// Candidate returned by [`TranscriptStore::find_resumable_session`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResumeCandidate {
    pub session_id: SessionId,
    pub current_branch_id: BranchId,
    pub daemon_pid: u32,
    pub started_at: DateTime<Utc>,
}

/// Persistence for session transcripts.
#[async_trait]
pub trait TranscriptStore: fmt::Debug + Send + Sync + 'static {
    /// Atomically insert `session` and create its `main` branch.
    async fn begin_session_with_main_branch(
        &self,
        session: SessionId,
        daemon_pid: u32,
    ) -> Result<BranchId>;

    /// Mark `session` as ended by stamping `ended_at`. Errors with
    /// [`MemoryError::SessionNotFound`](crate::MemoryError::SessionNotFound) when
    /// `session` was never saved.
    async fn end_session(&self, session: SessionId) -> Result<()>;

    /// Open a new turn in `branch`'s session and append `prompt` to `branch` as its
    /// first message, as [`append_message_to_branch`](Self::append_message_to_branch)
    /// does, in one transaction.
    async fn begin_turn(&self, branch: BranchId, prompt: PersistedMessage) -> Result<OpenedTurn>;

    /// Make `branch` the current branch of `session`. Errors with
    /// [`MemoryError::SessionNotFound`](crate::MemoryError::SessionNotFound) or, when
    /// `branch` is not one of the session's,
    /// [`MemoryError::BranchNotFound`](crate::MemoryError::BranchNotFound).
    async fn set_current_branch(&self, session: SessionId, branch: BranchId) -> Result<()>;

    /// Read the current branch pointer for `session`, if any.
    async fn current_branch(&self, session: SessionId) -> Result<Option<BranchId>>;

    /// Append `msg` to `branch` and its session, chunking user and assistant text for
    /// embedding, all in one transaction. Errors with
    /// [`MemoryError::BranchNotFound`](crate::MemoryError::BranchNotFound) when `branch` is missing.
    async fn append_message_to_branch(
        &self,
        branch: BranchId,
        turn: TurnId,
        msg: PersistedMessage,
    ) -> Result<AppendedMessage>;

    /// Every branch of every session, most recently active first, then by
    /// branch id.
    async fn list_branches(&self) -> Result<Vec<BranchInfo>>;

    /// Metadata of `branch`, or `None` when it does not exist.
    async fn branch_info(&self, branch: BranchId) -> Result<Option<BranchInfo>>;

    /// Resolve `name` or `<session-id-prefix>/name` to a branch. An unqualified name
    /// prefers `prefer_session`; remaining ties go to the newest session.
    async fn resolve_branch(
        &self,
        target: &str,
        prefer_session: Option<SessionId>,
    ) -> Result<Option<(SessionId, BranchId)>>;

    /// Copy `src` into a new child branch `new_name` sharing all its messages and make it
    /// its session's current branch, in one transaction. Errors with
    /// [`MemoryError::BranchNotFound`](crate::MemoryError::BranchNotFound) or
    /// [`MemoryError::BranchNameTaken`](crate::MemoryError::BranchNameTaken).
    async fn fork_and_switch(&self, src: BranchId, new_name: &str) -> Result<BranchId>;

    /// Every message on `branch`, ordered by branch-local seq.
    async fn load_branch_history(&self, branch: BranchId) -> Result<Vec<HistoryRow>>;

    /// Make `summary`, which took `duration_ms` to write, the branch's summary of
    /// everything `cut` leaves out; a no-op when the branch has fewer user
    /// messages than `cut` keeps.
    async fn record_branch_summary(
        &self,
        branch: BranchId,
        summary: String,
        cut: SummaryCut,
        duration_ms: u64,
    ) -> Result<()>;

    /// The branch's latest compaction summary, if one is recorded.
    async fn load_branch_summary(&self, branch: BranchId) -> Result<Option<BranchSummary>>;

    /// Record how finished `turn` ran, replacing any earlier record.
    async fn record_turn_stats(&self, turn: TurnId, stats: TurnStats) -> Result<()>;

    /// When the newest message on `branch` was written, or `None` when the branch is empty.
    async fn latest_branch_activity(&self, branch: BranchId) -> Result<Option<DateTime<Utc>>>;

    /// Drop the latest turn from `branch`, deleting messages and the turn row once
    /// no other branch references them.
    async fn undo_last_turn(&self, branch: BranchId) -> Result<UndoOutcome>;

    /// The newest unended session with a current branch; its `daemon_pid` is not checked.
    async fn find_resumable_session(&self) -> Result<Option<ResumeCandidate>>;

    /// Current `sessions.title`, or `None` when none has been set.
    async fn session_title(&self, session: SessionId) -> Result<Option<String>>;

    /// Set `sessions.title` for `session`.
    async fn set_session_title(&self, session: SessionId, title: &str) -> Result<()>;
}

/// No-op store: writes are discarded and reads find nothing. Returned ids are
/// placeholders, meaningful only when passed back to this store.
#[derive(Debug)]
pub struct NoTranscriptStore;

#[async_trait]
impl TranscriptStore for NoTranscriptStore {
    async fn begin_session_with_main_branch(
        &self,
        _session: SessionId,
        _daemon_pid: u32,
    ) -> Result<BranchId> {
        Ok(BranchId(0))
    }

    async fn end_session(&self, _session: SessionId) -> Result<()> {
        Ok(())
    }

    async fn begin_turn(&self, _branch: BranchId, _prompt: PersistedMessage) -> Result<OpenedTurn> {
        Ok(OpenedTurn {
            turn: TurnId(0),
            message: AppendedMessage {
                id: MessageId(0),
                chunks: Vec::new(),
            },
        })
    }

    async fn set_current_branch(&self, _session: SessionId, _branch: BranchId) -> Result<()> {
        Ok(())
    }

    async fn current_branch(&self, _session: SessionId) -> Result<Option<BranchId>> {
        Ok(None)
    }

    async fn append_message_to_branch(
        &self,
        _branch: BranchId,
        _turn: TurnId,
        _msg: PersistedMessage,
    ) -> Result<AppendedMessage> {
        Ok(AppendedMessage {
            id: MessageId(0),
            chunks: Vec::new(),
        })
    }

    async fn list_branches(&self) -> Result<Vec<BranchInfo>> {
        Ok(Vec::new())
    }

    async fn branch_info(&self, _branch: BranchId) -> Result<Option<BranchInfo>> {
        Ok(None)
    }

    async fn resolve_branch(
        &self,
        _target: &str,
        _prefer_session: Option<SessionId>,
    ) -> Result<Option<(SessionId, BranchId)>> {
        Ok(None)
    }

    async fn fork_and_switch(&self, _src: BranchId, _new_name: &str) -> Result<BranchId> {
        Ok(BranchId(0))
    }

    async fn load_branch_history(&self, _branch: BranchId) -> Result<Vec<HistoryRow>> {
        Ok(Vec::new())
    }

    async fn record_branch_summary(
        &self,
        _branch: BranchId,
        _summary: String,
        _cut: SummaryCut,
        _duration_ms: u64,
    ) -> Result<()> {
        Ok(())
    }

    async fn load_branch_summary(&self, _branch: BranchId) -> Result<Option<BranchSummary>> {
        Ok(None)
    }

    async fn record_turn_stats(&self, _turn: TurnId, _stats: TurnStats) -> Result<()> {
        Ok(())
    }

    async fn latest_branch_activity(&self, _branch: BranchId) -> Result<Option<DateTime<Utc>>> {
        Ok(None)
    }

    async fn undo_last_turn(&self, _branch: BranchId) -> Result<UndoOutcome> {
        Ok(UndoOutcome::default())
    }

    async fn find_resumable_session(&self) -> Result<Option<ResumeCandidate>> {
        Ok(None)
    }

    async fn session_title(&self, _session: SessionId) -> Result<Option<String>> {
        Ok(None)
    }

    async fn set_session_title(&self, _session: SessionId, _title: &str) -> Result<()> {
        Ok(())
    }
}
