//! Session transcripts: sessions, turns, branches, and their messages.

use std::fmt;
use std::num::NonZeroUsize;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rusqlite::types::{FromSql, FromSqlError, FromSqlResult, ToSqlOutput, Type, ValueRef};
use rusqlite::{OptionalExtension, Row, ToSql};

use super::connection::SqliteHandle;
use super::ids::{BranchId, MessageId, SessionId, TurnId};
use super::timestamp::{optional_timestamp_at, timestamp_at};
use super::writer::{WriteOp, dispatch_write};
use crate::{MemoryError, Result};

/// Columns [`branch_info`] reads, over `branches b JOIN sessions s`.
const BRANCH_INFO_SELECT: &str = "
    SELECT  b.id,
            s.uuid,
            s.started_at,
            s.ended_at,
            s.title,
            b.name,
            b.parent_branch_id,
            (SELECT name FROM branches p WHERE p.id = b.parent_branch_id),
            b.fork_point_seq,
            b.created_at,
            (SELECT COUNT(*) FROM branch_messages bm WHERE bm.branch_id = b.id),
            (b.id = s.current_branch_id) AS is_current,
            MAX(b.created_at, COALESCE(
                (SELECT m.created_at
                 FROM branch_messages bm JOIN messages m ON m.id = bm.message_id
                 WHERE bm.branch_id = b.id ORDER BY bm.seq DESC LIMIT 1),
                b.created_at)) AS last_active_at
    FROM branches b JOIN sessions s ON s.id = b.session_id";

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

    fn code(self) -> i64 {
        match self {
            PersistedRole::User => 0,
            PersistedRole::Assistant => 1,
            PersistedRole::Tool => 2,
        }
    }
}

impl ToSql for PersistedRole {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        Ok(ToSqlOutput::from(self.code()))
    }
}

impl FromSql for PersistedRole {
    fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
        match value.as_i64()? {
            0 => Ok(PersistedRole::User),
            1 => Ok(PersistedRole::Assistant),
            2 => Ok(PersistedRole::Tool),
            code => Err(FromSqlError::OutOfRange(code)),
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

/// Result of [`TranscriptStore::undo_last_turn`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UndoOutcome {
    /// Number of `branch_messages` rows dropped.
    pub removed_messages: u32,
    pub last_user_text: Option<String>,
}

/// Persistence for session transcripts.
#[async_trait]
pub trait TranscriptStore: fmt::Debug + Send + Sync + 'static {
    /// Mark `id` as ended by stamping `ended_at`.
    async fn end_session(&self, id: &SessionId) -> Result<()>;
    /// Open a new turn inside `session`; its messages are grouped by the id.
    async fn begin_turn(&self, session: &SessionId) -> Result<TurnId>;

    /// Atomically insert `session` and create its `main` branch.
    async fn begin_session_with_main_branch(
        &self,
        session: &SessionId,
        daemon_pid: u32,
    ) -> Result<BranchId>;

    /// Update `sessions.current_branch_id` to point at `branch`.
    async fn set_current_branch(&self, session: &SessionId, branch: BranchId) -> Result<()>;

    /// Read the current branch pointer for `session`, if any.
    async fn get_current_branch(&self, session: &SessionId) -> Result<Option<BranchId>>;

    /// Append `msg` to `branch` in one transaction; returns its message id.
    async fn append_message_to_branch(
        &self,
        session: &SessionId,
        branch: BranchId,
        turn: Option<TurnId>,
        msg: PersistedMessage,
    ) -> Result<MessageId>;

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
        prefer_session: Option<&SessionId>,
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
    async fn get_session_title(&self, session: &SessionId) -> Result<Option<String>>;

    /// Set `sessions.title` for `session`.
    async fn set_session_title(&self, session: &SessionId, title: &str) -> Result<()>;
}

/// Candidate returned by [`TranscriptStore::find_resumable_session`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResumeCandidate {
    pub session_id: SessionId,
    pub current_branch_id: BranchId,
    pub daemon_pid: u32,
    pub started_at: DateTime<Utc>,
}

/// No-op store: writes are discarded and reads find nothing. Returned ids are
/// placeholders, meaningful only when passed back to this store.
#[derive(Debug)]
pub struct NoTranscriptStore;

#[async_trait]
impl TranscriptStore for NoTranscriptStore {
    async fn end_session(&self, _id: &SessionId) -> Result<()> {
        Ok(())
    }
    async fn begin_turn(&self, _session: &SessionId) -> Result<TurnId> {
        Ok(TurnId(0))
    }

    async fn begin_session_with_main_branch(
        &self,
        _session: &SessionId,
        _pid: u32,
    ) -> Result<BranchId> {
        Ok(BranchId(0))
    }

    async fn set_current_branch(&self, _session: &SessionId, _branch: BranchId) -> Result<()> {
        Ok(())
    }

    async fn get_current_branch(&self, _session: &SessionId) -> Result<Option<BranchId>> {
        Ok(None)
    }

    async fn append_message_to_branch(
        &self,
        _session: &SessionId,
        _branch: BranchId,
        _turn: Option<TurnId>,
        _msg: PersistedMessage,
    ) -> Result<MessageId> {
        Ok(MessageId(0))
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
        _prefer_session: Option<&SessionId>,
    ) -> Result<Option<(SessionId, BranchId)>> {
        Ok(None)
    }

    async fn fork_branch(&self, _src: BranchId, _name: &str) -> Result<BranchId> {
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

    async fn get_session_title(&self, _session: &SessionId) -> Result<Option<String>> {
        Ok(None)
    }

    async fn set_session_title(&self, _session: &SessionId, _title: &str) -> Result<()> {
        Ok(())
    }
}

/// SQLite-backed [`TranscriptStore`].
#[derive(Debug, Clone)]
pub struct SqliteTranscriptStore {
    handle: SqliteHandle,
}

impl SqliteTranscriptStore {
    /// Store over the shared database handle.
    pub fn new(handle: SqliteHandle) -> Self {
        Self { handle }
    }
}

#[async_trait]
impl TranscriptStore for SqliteTranscriptStore {
    async fn end_session(&self, id: &SessionId) -> Result<()> {
        let session_id = *id;
        dispatch_write(self.handle.writer(), |ack| WriteOp::EndSession {
            session_id,
            ack,
        })
        .await
    }

    async fn begin_turn(&self, session: &SessionId) -> Result<TurnId> {
        let session_id = *session;
        dispatch_write(self.handle.writer(), |ack| WriteOp::BeginTurn {
            session_id,
            ack,
        })
        .await
    }

    async fn begin_session_with_main_branch(
        &self,
        session: &SessionId,
        daemon_pid: u32,
    ) -> Result<BranchId> {
        let session_id = *session;
        dispatch_write(self.handle.writer(), |ack| {
            WriteOp::BeginSessionWithMainBranch {
                session_id,
                daemon_pid,
                ack,
            }
        })
        .await
    }

    async fn set_current_branch(&self, session: &SessionId, branch: BranchId) -> Result<()> {
        let session_id = *session;
        dispatch_write(self.handle.writer(), |ack| WriteOp::SetCurrentBranch {
            session_id,
            branch_id: branch,
            ack,
        })
        .await
    }

    async fn get_current_branch(&self, session: &SessionId) -> Result<Option<BranchId>> {
        let session_id = *session;
        self.handle
            .reader()
            .call(move |c| -> rusqlite::Result<_> {
                c.query_row(
                    "SELECT current_branch_id FROM sessions WHERE uuid = ?1",
                    rusqlite::params![session_id],
                    |r| r.get::<_, Option<BranchId>>(0),
                )
                .optional()
                .map(Option::flatten)
            })
            .await
            .map_err(MemoryError::sqlite("get_current_branch"))
    }

    async fn append_message_to_branch(
        &self,
        session: &SessionId,
        branch: BranchId,
        turn: Option<TurnId>,
        msg: PersistedMessage,
    ) -> Result<MessageId> {
        let session_id = *session;
        dispatch_write(self.handle.writer(), |ack| WriteOp::AppendMessageToBranch {
            session_id,
            branch_id: branch,
            turn_id: turn,
            msg,
            ack,
        })
        .await
    }

    async fn list_branches(&self) -> Result<Vec<BranchInfo>> {
        self.handle
            .reader()
            .call(|c| -> rusqlite::Result<_> {
                let sql = format!("{BRANCH_INFO_SELECT} ORDER BY last_active_at DESC, b.id ASC");
                let mut stmt = c.prepare(&sql)?;
                let rows: Vec<BranchInfo> = stmt
                    .query_map([], branch_info)?
                    .collect::<std::result::Result<_, _>>()?;
                Ok(rows)
            })
            .await
            .map_err(MemoryError::sqlite("list_branches"))
    }

    async fn branch_info(&self, branch: BranchId) -> Result<Option<BranchInfo>> {
        self.handle
            .reader()
            .call(move |c| -> rusqlite::Result<_> {
                let sql = format!("{BRANCH_INFO_SELECT} WHERE b.id = ?1");
                c.query_row(&sql, rusqlite::params![branch], branch_info)
                    .optional()
            })
            .await
            .map_err(MemoryError::sqlite("branch_info"))
    }

    async fn resolve_branch(
        &self,
        target: &str,
        prefer_session: Option<&SessionId>,
    ) -> Result<Option<(SessionId, BranchId)>> {
        let (session_prefix, name) = match target.split_once('/') {
            Some((prefix, name)) => (Some(prefix.to_string()), name.to_string()),
            None => (None, target.to_string()),
        };
        let prefer_session = prefer_session.copied();
        self.handle
            .reader()
            .call(move |c| -> rusqlite::Result<_> {
                if let Some(prefix) = session_prefix {
                    return c
                        .query_row(
                            "SELECT s.uuid, b.id
                             FROM branches b JOIN sessions s ON s.id = b.session_id
                             WHERE b.name = ?1 AND s.uuid LIKE ?2 ESCAPE '\\'
                             ORDER BY s.started_at DESC LIMIT 1",
                            rusqlite::params![name, like_prefix_pattern(&prefix)],
                            branch_ref,
                        )
                        .optional();
                }
                if let Some(pref) = prefer_session
                    && let Some(row) = c
                        .query_row(
                            "SELECT s.uuid, b.id
                             FROM branches b JOIN sessions s ON s.id = b.session_id
                             WHERE b.name = ?1 AND s.uuid = ?2",
                            rusqlite::params![name, pref],
                            branch_ref,
                        )
                        .optional()?
                {
                    return Ok(Some(row));
                }
                c.query_row(
                    "SELECT s.uuid, b.id
                     FROM branches b JOIN sessions s ON s.id = b.session_id
                     WHERE b.name = ?1
                     ORDER BY s.started_at DESC LIMIT 1",
                    rusqlite::params![name],
                    branch_ref,
                )
                .optional()
            })
            .await
            .map_err(MemoryError::sqlite("resolve_branch"))
    }

    async fn fork_branch(&self, src: BranchId, new_name: &str) -> Result<BranchId> {
        let new_name = new_name.to_string();
        dispatch_write(self.handle.writer(), |ack| WriteOp::ForkBranch {
            src_branch_id: src,
            new_name,
            ack,
        })
        .await
    }

    async fn load_branch_history(&self, branch: BranchId) -> Result<Vec<HistoryRow>> {
        self.handle
            .reader()
            .call(move |c| -> rusqlite::Result<_> {
                let sql = "
                    SELECT  m.id,
                            bm.seq,
                            m.role,
                            m.content,
                            m.tool_calls,
                            m.tool_call_id,
                            m.tool_name,
                            m.reasoning,
                            m.tool_exit_code,
                            m.tool_duration_ms
                    FROM branch_messages bm JOIN messages m ON m.id = bm.message_id
                    WHERE bm.branch_id = ?1
                    ORDER BY bm.seq ASC
                ";
                let mut stmt = c.prepare(sql)?;
                let rows = stmt
                    .query_map(rusqlite::params![branch], history_row)?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .await
            .map_err(MemoryError::sqlite("load_branch_history"))
    }

    async fn record_branch_summary(
        &self,
        branch: BranchId,
        summary: String,
        kept_turns: NonZeroUsize,
    ) -> Result<()> {
        dispatch_write(self.handle.writer(), |ack| WriteOp::RecordBranchSummary {
            branch_id: branch,
            summary,
            kept_turns,
            ack,
        })
        .await
    }

    async fn load_branch_summary(&self, branch: BranchId) -> Result<Option<BranchSummary>> {
        self.handle
            .reader()
            .call(move |c| -> rusqlite::Result<_> {
                c.query_row(
                    "SELECT through_seq, content FROM branch_summaries WHERE branch_id = ?1",
                    rusqlite::params![branch],
                    |r| {
                        Ok(BranchSummary {
                            through_seq: r.get(0)?,
                            content: r.get(1)?,
                        })
                    },
                )
                .optional()
            })
            .await
            .map_err(MemoryError::sqlite("load_branch_summary"))
    }

    async fn branch_tail_seq(&self, branch: BranchId) -> Result<Option<i64>> {
        self.handle
            .reader()
            .call(move |c| -> rusqlite::Result<_> {
                c.query_row(
                    "SELECT MAX(seq) FROM branch_messages WHERE branch_id = ?1",
                    rusqlite::params![branch],
                    |r| r.get::<_, Option<i64>>(0),
                )
            })
            .await
            .map_err(MemoryError::sqlite("branch_tail_seq"))
    }

    async fn latest_branch_activity(&self, branch: BranchId) -> Result<Option<DateTime<Utc>>> {
        self.handle
            .reader()
            .call(move |c| -> rusqlite::Result<_> {
                c.query_row(
                    "SELECT m.created_at
                     FROM branch_messages bm
                     JOIN messages m ON m.id = bm.message_id
                     WHERE bm.branch_id = ?1
                     ORDER BY bm.seq DESC LIMIT 1",
                    rusqlite::params![branch],
                    |r| timestamp_at(r, 0),
                )
                .optional()
            })
            .await
            .map_err(MemoryError::sqlite("latest_branch_activity"))
    }

    async fn undo_last_turn(&self, branch: BranchId) -> Result<UndoOutcome> {
        dispatch_write(self.handle.writer(), |ack| WriteOp::UndoLastTurn {
            branch_id: branch,
            ack,
        })
        .await
    }

    async fn get_session_title(&self, session: &SessionId) -> Result<Option<String>> {
        let session_id = *session;
        self.handle
            .reader()
            .call(move |c| -> rusqlite::Result<_> {
                c.query_row(
                    "SELECT title FROM sessions WHERE uuid = ?1",
                    rusqlite::params![session_id],
                    |r| r.get::<_, Option<String>>(0),
                )
                .optional()
                .map(Option::flatten)
            })
            .await
            .map_err(MemoryError::sqlite("get_session_title"))
    }

    async fn set_session_title(&self, session: &SessionId, title: &str) -> Result<()> {
        let session_id = *session;
        let title = title.to_string();
        dispatch_write(self.handle.writer(), |ack| WriteOp::SetSessionTitle {
            session_id,
            title,
            ack,
        })
        .await
    }

    async fn find_resumable_session(&self) -> Result<Option<ResumeCandidate>> {
        self.handle
            .reader()
            .call(|c| -> rusqlite::Result<_> {
                c.query_row(
                    "SELECT uuid, current_branch_id, daemon_pid, started_at
                     FROM sessions
                     WHERE ended_at IS NULL AND current_branch_id IS NOT NULL
                     ORDER BY started_at DESC LIMIT 1",
                    [],
                    |r| {
                        Ok(ResumeCandidate {
                            session_id: r.get(0)?,
                            current_branch_id: r.get(1)?,
                            daemon_pid: r.get(2)?,
                            started_at: timestamp_at(r, 3)?,
                        })
                    },
                )
                .optional()
            })
            .await
            .map_err(MemoryError::sqlite("find_resumable_session"))
    }
}

/// A `LIKE` pattern (escape char `\`) matching values that start with `prefix` literally,
/// ignoring ASCII case.
fn like_prefix_pattern(prefix: &str) -> String {
    let escaped = prefix
        .replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_");
    format!("{escaped}%")
}

fn branch_ref(row: &Row<'_>) -> rusqlite::Result<(SessionId, BranchId)> {
    Ok((row.get(0)?, row.get(1)?))
}

fn branch_info(row: &Row<'_>) -> rusqlite::Result<BranchInfo> {
    Ok(BranchInfo {
        branch_id: row.get(0)?,
        session_id: row.get(1)?,
        session_started_at: timestamp_at(row, 2)?,
        session_ended_at: optional_timestamp_at(row, 3)?,
        session_title: row.get(4)?,
        name: row.get(5)?,
        parent_branch_id: row.get(6)?,
        parent_branch_name: row.get(7)?,
        fork_point_seq: row.get(8)?,
        created_at: timestamp_at(row, 9)?,
        message_count: row.get(10)?,
        is_current_in_session: row.get(11)?,
        last_active_at: timestamp_at(row, 12)?,
    })
}

fn history_row(row: &Row<'_>) -> rusqlite::Result<HistoryRow> {
    let kind = match row.get(2)? {
        PersistedRole::User => MessageKind::User,
        PersistedRole::Assistant => MessageKind::Assistant {
            reasoning: row.get(7)?,
            tool_calls: tool_calls_at(row, 4)?,
        },
        PersistedRole::Tool => MessageKind::Tool {
            call_id: row.get(5)?,
            tool_name: row.get(6)?,
            outcome: ToolOutcome {
                exit_code: row.get(8)?,
                duration_ms: row.get(9)?,
            },
        },
    };
    Ok(HistoryRow {
        message_id: row.get(0)?,
        seq: row.get(1)?,
        message: PersistedMessage {
            content: row.get(3)?,
            kind,
        },
    })
}

fn tool_calls_at(row: &Row<'_>, idx: usize) -> rusqlite::Result<Option<serde_json::Value>> {
    row.get::<_, Option<String>>(idx)?
        .map(|json| {
            serde_json::from_str(&json).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(idx, Type::Text, Box::new(e))
            })
        })
        .transpose()
}

#[cfg(test)]
mod tests;
