//! Session transcripts: sessions, turns, branches, and their messages.

use std::fmt;
use std::num::NonZeroUsize;

use async_trait::async_trait;
use chrono::{DateTime, Utc};

use crate::{BranchId, ChunkId, MessageId, Result, SessionId, TurnId};

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
        /// The model's reasoning for this step; never empty.
        reasoning: Option<String>,
        /// JSON array of `{id, name, arguments}`; only when the step called tools.
        tool_calls: Option<serde_json::Value>,
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

    /// An assistant reply with no tool calls; empty `reasoning` is not stored.
    pub fn assistant_text(content: impl Into<String>, reasoning: String) -> Self {
        Self::assistant(content.into(), reasoning, None)
    }

    /// A model step that requested tools; `content` is any accompanying text,
    /// possibly empty, and empty `reasoning` is not stored.
    pub fn assistant_tool_calls(
        content: impl Into<String>,
        calls: serde_json::Value,
        reasoning: String,
    ) -> Self {
        Self::assistant(content.into(), reasoning, Some(calls))
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
        reasoning: String,
        tool_calls: Option<serde_json::Value>,
    ) -> Self {
        Self {
            content,
            kind: MessageKind::Assistant {
                reasoning: Some(reasoning).filter(|r| !r.is_empty()),
                tool_calls,
            },
        }
    }
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
    pub fork_point_seq: Option<i64>,
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
    pub seq: i64,
    pub message: PersistedMessage,
}

/// The latest compaction summary of a branch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BranchSummary {
    /// Branch-local seq of the last message the summary stands in for.
    pub through_seq: i64,
    pub content: String,
}

/// Result of [`TranscriptStore::append_message_to_branch`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppendedMessage {
    pub id: MessageId,
    /// The chunks stored for embedding, as `(id, chunk text)`; empty for tool results.
    pub chunks: Vec<(ChunkId, String)>,
}

/// Result of [`TranscriptStore::undo_last_turn`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UndoOutcome {
    /// Number of `branch_messages` rows dropped.
    pub removed_messages: u32,
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

    /// Mark `session` as ended by stamping `ended_at`.
    async fn end_session(&self, session: SessionId) -> Result<()>;

    /// Open a new turn inside `session`; its messages are grouped by the id.
    async fn begin_turn(&self, session: SessionId) -> Result<TurnId>;

    /// Update `sessions.current_branch_id` to point at `branch`.
    async fn set_current_branch(&self, session: SessionId, branch: BranchId) -> Result<()>;

    /// Read the current branch pointer for `session`, if any.
    async fn current_branch(&self, session: SessionId) -> Result<Option<BranchId>>;

    /// Append `msg` to `branch` and its session, chunking user and assistant text for
    /// embedding, all in one transaction.
    async fn append_message_to_branch(
        &self,
        branch: BranchId,
        turn: Option<TurnId>,
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

    /// Copy `src` into a new child branch `new_name` sharing all its messages.
    /// Errors if `src` is missing or its session already has a branch `new_name`.
    async fn fork_branch(&self, src: BranchId, new_name: &str) -> Result<BranchId>;

    /// Every message on `branch`, ordered by branch-local seq.
    async fn load_branch_history(&self, branch: BranchId) -> Result<Vec<HistoryRow>>;

    /// Make `summary` the branch's summary of everything before its
    /// `kept_turns` most recent user messages; a no-op when the branch has
    /// fewer user messages than that.
    async fn record_branch_summary(
        &self,
        branch: BranchId,
        summary: String,
        kept_turns: NonZeroUsize,
    ) -> Result<()>;

    /// The branch's latest compaction summary, if one is recorded.
    async fn load_branch_summary(&self, branch: BranchId) -> Result<Option<BranchSummary>>;

    /// Branch-local seq of the newest message on `branch`, or `None`
    /// when the branch is empty.
    async fn branch_tail_seq(&self, branch: BranchId) -> Result<Option<i64>>;

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

    async fn begin_turn(&self, _session: SessionId) -> Result<TurnId> {
        Ok(TurnId(0))
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
        _turn: Option<TurnId>,
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

    async fn fork_branch(&self, _src: BranchId, _new_name: &str) -> Result<BranchId> {
        Ok(BranchId(0))
    }

    async fn load_branch_history(&self, _branch: BranchId) -> Result<Vec<HistoryRow>> {
        Ok(Vec::new())
    }

    async fn record_branch_summary(
        &self,
        _branch: BranchId,
        _summary: String,
        _kept_turns: NonZeroUsize,
    ) -> Result<()> {
        Ok(())
    }

    async fn load_branch_summary(&self, _branch: BranchId) -> Result<Option<BranchSummary>> {
        Ok(None)
    }

    async fn branch_tail_seq(&self, _branch: BranchId) -> Result<Option<i64>> {
        Ok(None)
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
