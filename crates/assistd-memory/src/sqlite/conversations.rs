//! Conversation persistence: sessions, turns, branches, messages, and
//! FTS5 search.

use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::connection::SqliteHandle;
use super::writer::{WriteOp, dispatch_write};

/// Session identifier: a UUID string, stable across daemon restarts.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(pub String);

impl SessionId {
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for SessionId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Opaque turn identifier (rowid).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TurnId(pub i64);

/// Opaque branch identifier (rowid in `branches`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BranchId(pub i64);

/// Role of a persisted message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PersistedRole {
    System,
    User,
    Assistant,
    Tool,
}

impl PersistedRole {
    /// Lowercase form stored in the `conversations.role` column.
    pub fn as_wire(self) -> &'static str {
        match self {
            PersistedRole::System => "system",
            PersistedRole::User => "user",
            PersistedRole::Assistant => "assistant",
            PersistedRole::Tool => "tool",
        }
    }

    /// Inverse of [`Self::as_wire`]; `None` for unrecognised values.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "system" => Some(PersistedRole::System),
            "user" => Some(PersistedRole::User),
            "assistant" => Some(PersistedRole::Assistant),
            "tool" => Some(PersistedRole::Tool),
            _ => None,
        }
    }
}

/// One message ready to write to the `conversations` table.
#[derive(Debug, Clone)]
pub struct PersistedMessage {
    pub role: PersistedRole,
    pub content: String,
    /// JSON array of `{id, name, arguments}`. Set only on assistant
    /// rows that requested tool calls.
    pub tool_calls: Option<serde_json::Value>,
    /// Set only when `role == Tool`.
    pub tool_call_id: Option<String>,
    /// Set only when `role == Tool`.
    pub tool_name: Option<String>,
}

impl PersistedMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: PersistedRole::User,
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
            tool_name: None,
        }
    }

    pub fn assistant_text(content: impl Into<String>) -> Self {
        Self {
            role: PersistedRole::Assistant,
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
            tool_name: None,
        }
    }

    pub fn assistant_tool_calls(calls: serde_json::Value) -> Self {
        Self {
            role: PersistedRole::Assistant,
            content: String::new(),
            tool_calls: Some(calls),
            tool_call_id: None,
            tool_name: None,
        }
    }

    pub fn tool_result(
        content: impl Into<String>,
        call_id: impl Into<String>,
        name: impl Into<String>,
    ) -> Self {
        Self {
            role: PersistedRole::Tool,
            content: content.into(),
            tool_calls: None,
            tool_call_id: Some(call_id.into()),
            tool_name: Some(name.into()),
        }
    }
}

/// Coarse summary of one turn.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TurnSummary {
    pub turn_id: i64,
    pub session_id: String,
    pub started_at: String,
    pub ended_at: Option<String>,
    pub user_text: String,
    pub message_count: i64,
}

/// Per-branch metadata. `is_current_in_session` flags the branch that
/// `sessions.current_branch_id` points at.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BranchInfo {
    pub branch_id: BranchId,
    pub session_id: String,
    pub session_started_at: String,
    pub session_ended_at: Option<String>,
    pub session_title: Option<String>,
    pub name: String,
    pub parent_branch_id: Option<BranchId>,
    pub parent_branch_name: Option<String>,
    pub fork_point_seq: Option<i64>,
    pub created_at: String,
    pub message_count: i64,
    pub is_current_in_session: bool,
}

/// One persisted message reconstructed for replay into the in-memory
/// conversation, including tool-call and tool-result rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoryRow {
    pub conversation_id: i64,
    pub seq: i64,
    pub role: PersistedRole,
    pub content: String,
    /// JSON array of `{id, name, arguments}` when the row is an
    /// assistant-with-tool-calls; `None` for plain rows.
    pub tool_calls: Option<serde_json::Value>,
    pub tool_call_id: Option<String>,
    pub tool_name: Option<String>,
}

/// Result of [`ConversationStore::undo_last_turn`]: how many
/// `branch_messages` rows were dropped, the undone user prompt, and the
/// dropped turn id.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct UndoOutcome {
    pub removed_messages: u32,
    pub last_user_text: Option<String>,
    pub removed_turn_id: Option<i64>,
}

/// Conversation persistence.
#[async_trait]
pub trait ConversationStore: Send + Sync + 'static {
    /// Mark `id` as ended by stamping `ended_at`.
    async fn end_session(&self, id: &SessionId) -> Result<()>;
    /// Open a new turn row inside `session` labelled with `user_text`.
    async fn begin_turn(&self, session: &SessionId, user_text: &str) -> Result<TurnId>;
    /// Mark `turn` as ended by stamping `ended_at`.
    async fn end_turn(&self, turn: TurnId) -> Result<()>;
    /// Return the `limit` most-recent turns ordered by turn id descending.
    async fn recent_turns(&self, limit: usize) -> Result<Vec<TurnSummary>>;

    /// Atomically begin a session and create its `main` branch, so a
    /// crash between the two writes can't leave a session without one.
    async fn begin_session_with_main_branch(
        &self,
        daemon_pid: u32,
    ) -> Result<(SessionId, BranchId)>;

    /// Insert a branch row in `session`.
    async fn create_branch(
        &self,
        session: &SessionId,
        name: &str,
        parent: Option<BranchId>,
        fork_point_seq: Option<i64>,
    ) -> Result<BranchId>;

    /// Update `sessions.current_branch_id` to point at `branch`.
    async fn set_current_branch(&self, session: &SessionId, branch: BranchId) -> Result<()>;

    /// Read the current branch pointer for `session`, if any.
    async fn get_current_branch(&self, session: &SessionId) -> Result<Option<BranchId>>;

    /// Append `msg` and reference it from `branch_messages` under
    /// `branch`, in one transaction. Returns the `conversations.id`.
    async fn append_message_to_branch(
        &self,
        session: &SessionId,
        branch: BranchId,
        turn: Option<TurnId>,
        msg: PersistedMessage,
    ) -> Result<i64>;

    /// Every branch across every session, sorted by session start
    /// (newest first) then branch id.
    async fn list_branches(&self) -> Result<Vec<BranchInfo>>;

    /// Look up a branch by name, optionally qualified by an 8-char
    /// session id prefix. Returns the matching `(SessionId, BranchId)`
    /// pair or `None` when no match exists. Ambiguous matches
    /// (multiple branches with the same name across sessions) return
    /// the first hit ordered by session.started_at DESC; callers can
    /// detect ambiguity by passing the qualified form.
    async fn resolve_branch(
        &self,
        target: &str,
        prefer_session: Option<&SessionId>,
    ) -> Result<Option<(SessionId, BranchId)>>;

    /// Snapshot a branch by copying every `branch_messages` row from
    /// `src` into a freshly-created branch named `new_name` with
    /// `parent = src` and `fork_point_seq = max seq on src`. Returns
    /// the new BranchId. Atomic in a single transaction.
    async fn fork_branch(&self, src: BranchId, new_name: &str) -> Result<BranchId>;

    /// Every message on `branch`, ordered by branch-local seq.
    async fn load_branch_history(&self, branch: BranchId) -> Result<Vec<HistoryRow>>;

    /// RFC3339 timestamp of the newest message on `branch`, or `None`
    /// when the branch is empty.
    async fn latest_branch_activity(&self, branch: BranchId) -> Result<Option<String>>;

    /// Drop the latest turn from `branch`: its `branch_messages` rows,
    /// any `conversations` rows no branch references any more, and the
    /// `turns` row when no surviving message points at it.
    async fn undo_last_turn(&self, branch: BranchId) -> Result<UndoOutcome>;

    /// The most recent session with `ended_at IS NULL` and a current
    /// branch, or `None`. The caller checks whether `daemon_pid` is
    /// still alive before claiming it.
    async fn find_resumable_session(&self) -> Result<Option<ResumeCandidate>>;

    /// Current `sessions.title`, or `None` when none has been set.
    async fn get_session_title(&self, session: &SessionId) -> Result<Option<String>>;

    /// Set `sessions.title` for `session`.
    async fn set_session_title(&self, session: &SessionId, title: &str) -> Result<()>;
}

/// Candidate session returned by
/// [`ConversationStore::find_resumable_session`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResumeCandidate {
    pub session_id: SessionId,
    pub current_branch_id: BranchId,
    pub daemon_pid: u32,
    pub started_at: String,
}

/// No-op fallback used when memory is disabled.
pub struct NoConversationStore;

#[async_trait]
impl ConversationStore for NoConversationStore {
    async fn end_session(&self, _id: &SessionId) -> Result<()> {
        Ok(())
    }
    async fn begin_turn(&self, _s: &SessionId, _t: &str) -> Result<TurnId> {
        Ok(TurnId(0))
    }
    async fn end_turn(&self, _t: TurnId) -> Result<()> {
        Ok(())
    }
    async fn recent_turns(&self, _l: usize) -> Result<Vec<TurnSummary>> {
        Ok(Vec::new())
    }

    async fn begin_session_with_main_branch(&self, _pid: u32) -> Result<(SessionId, BranchId)> {
        Ok((SessionId::new(), BranchId(0)))
    }

    async fn create_branch(
        &self,
        _s: &SessionId,
        _name: &str,
        _parent: Option<BranchId>,
        _fp: Option<i64>,
    ) -> Result<BranchId> {
        Ok(BranchId(0))
    }

    async fn set_current_branch(&self, _s: &SessionId, _b: BranchId) -> Result<()> {
        Ok(())
    }

    async fn get_current_branch(&self, _s: &SessionId) -> Result<Option<BranchId>> {
        Ok(None)
    }

    async fn append_message_to_branch(
        &self,
        _s: &SessionId,
        _b: BranchId,
        _t: Option<TurnId>,
        _m: PersistedMessage,
    ) -> Result<i64> {
        Ok(0)
    }

    async fn list_branches(&self) -> Result<Vec<BranchInfo>> {
        Ok(Vec::new())
    }

    async fn resolve_branch(
        &self,
        _t: &str,
        _p: Option<&SessionId>,
    ) -> Result<Option<(SessionId, BranchId)>> {
        Ok(None)
    }

    async fn fork_branch(&self, _src: BranchId, _name: &str) -> Result<BranchId> {
        Ok(BranchId(0))
    }

    async fn load_branch_history(&self, _b: BranchId) -> Result<Vec<HistoryRow>> {
        Ok(Vec::new())
    }

    async fn latest_branch_activity(&self, _b: BranchId) -> Result<Option<String>> {
        Ok(None)
    }

    async fn undo_last_turn(&self, _b: BranchId) -> Result<UndoOutcome> {
        Ok(UndoOutcome::default())
    }

    async fn find_resumable_session(&self) -> Result<Option<ResumeCandidate>> {
        Ok(None)
    }

    async fn get_session_title(&self, _s: &SessionId) -> Result<Option<String>> {
        Ok(None)
    }

    async fn set_session_title(&self, _s: &SessionId, _t: &str) -> Result<()> {
        Ok(())
    }
}

/// SQLite-backed [`ConversationStore`].
#[derive(Clone)]
pub struct SqliteConversationStore {
    handle: Arc<SqliteHandle>,
}

impl SqliteConversationStore {
    pub fn new(handle: Arc<SqliteHandle>) -> Self {
        Self { handle }
    }
}

#[async_trait]
impl ConversationStore for SqliteConversationStore {
    async fn end_session(&self, id: &SessionId) -> Result<()> {
        let session_id = id.0.clone();
        dispatch_write(self.handle.writer(), |ack| WriteOp::EndSession {
            session_id,
            ack,
        })
        .await
    }

    async fn begin_turn(&self, session: &SessionId, user_text: &str) -> Result<TurnId> {
        let session_id = session.0.clone();
        let text = user_text.to_string();
        dispatch_write(self.handle.writer(), |ack| WriteOp::BeginTurn {
            session_id,
            user_text: text,
            ack,
        })
        .await
    }

    async fn end_turn(&self, turn: TurnId) -> Result<()> {
        dispatch_write(self.handle.writer(), |ack| WriteOp::EndTurn {
            turn_id: turn,
            ack,
        })
        .await
    }

    async fn recent_turns(&self, limit: usize) -> Result<Vec<TurnSummary>> {
        let limit = limit as i64;
        self.handle
            .conn()
            .call(move |c| -> rusqlite::Result<_> {
                let sql = "
                    SELECT  t.id,
                            t.session_id,
                            t.started_at,
                            t.ended_at,
                            t.user_text,
                            (SELECT count(*) FROM conversations WHERE turn_id = t.id)
                    FROM turns t
                    ORDER BY t.id DESC
                    LIMIT ?1
                ";
                let mut stmt = c.prepare(sql)?;
                let rows = stmt
                    .query_map(rusqlite::params![limit], |row| {
                        Ok(TurnSummary {
                            turn_id: row.get(0)?,
                            session_id: row.get(1)?,
                            started_at: row.get(2)?,
                            ended_at: row.get(3)?,
                            user_text: row.get(4)?,
                            message_count: row.get(5)?,
                        })
                    })?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .await
            .context("recent_turns")
    }

    async fn begin_session_with_main_branch(
        &self,
        daemon_pid: u32,
    ) -> Result<(SessionId, BranchId)> {
        let id = SessionId::new();
        let session_id = id.0.clone();
        let branch = dispatch_write(self.handle.writer(), |ack| {
            WriteOp::BeginSessionWithMainBranch {
                session_id,
                daemon_pid,
                ack,
            }
        })
        .await?;
        Ok((id, branch))
    }

    async fn create_branch(
        &self,
        session: &SessionId,
        name: &str,
        parent: Option<BranchId>,
        fork_point_seq: Option<i64>,
    ) -> Result<BranchId> {
        let session_id = session.0.clone();
        let name = name.to_string();
        dispatch_write(self.handle.writer(), |ack| WriteOp::CreateBranch {
            session_id,
            name,
            parent_branch_id: parent,
            fork_point_seq,
            ack,
        })
        .await
    }

    async fn set_current_branch(&self, session: &SessionId, branch: BranchId) -> Result<()> {
        let session_id = session.0.clone();
        dispatch_write(self.handle.writer(), |ack| WriteOp::SetCurrentBranch {
            session_id,
            branch_id: branch,
            ack,
        })
        .await
    }

    async fn get_current_branch(&self, session: &SessionId) -> Result<Option<BranchId>> {
        let session_id = session.0.clone();
        let id: Option<i64> = self
            .handle
            .conn()
            .call(move |c| -> rusqlite::Result<_> {
                Ok(c.query_row(
                    "SELECT current_branch_id FROM sessions WHERE id = ?1",
                    rusqlite::params![session_id],
                    |r| r.get::<_, Option<i64>>(0),
                )
                .ok()
                .flatten())
            })
            .await
            .context("get_current_branch")?;
        Ok(id.map(BranchId))
    }

    async fn append_message_to_branch(
        &self,
        session: &SessionId,
        branch: BranchId,
        turn: Option<TurnId>,
        msg: PersistedMessage,
    ) -> Result<i64> {
        let session_id = session.0.clone();
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
            .conn()
            .call(|c| -> rusqlite::Result<_> {
                let sql = "
                    SELECT  b.id,
                            b.session_id,
                            s.started_at,
                            s.ended_at,
                            s.title,
                            b.name,
                            b.parent_branch_id,
                            (SELECT name FROM branches p WHERE p.id = b.parent_branch_id),
                            b.fork_point_seq,
                            b.created_at,
                            (SELECT COUNT(*) FROM branch_messages bm WHERE bm.branch_id = b.id),
                            (b.id = s.current_branch_id) AS is_current
                    FROM branches b JOIN sessions s ON s.id = b.session_id
                    ORDER BY s.started_at DESC, b.id ASC
                ";
                let mut stmt = c.prepare(sql)?;
                let rows: Vec<BranchInfo> = stmt
                    .query_map([], |row| {
                        let parent_id: Option<i64> = row.get(6)?;
                        let is_current: i64 = row.get(11)?;
                        Ok(BranchInfo {
                            branch_id: BranchId(row.get(0)?),
                            session_id: row.get(1)?,
                            session_started_at: row.get(2)?,
                            session_ended_at: row.get(3)?,
                            session_title: row.get(4)?,
                            name: row.get(5)?,
                            parent_branch_id: parent_id.map(BranchId),
                            parent_branch_name: row.get(7)?,
                            fork_point_seq: row.get(8)?,
                            created_at: row.get(9)?,
                            message_count: row.get(10)?,
                            is_current_in_session: is_current != 0,
                        })
                    })?
                    .collect::<std::result::Result<_, _>>()?;
                Ok(rows)
            })
            .await
            .context("list_branches")
    }

    async fn resolve_branch(
        &self,
        target: &str,
        prefer_session: Option<&SessionId>,
    ) -> Result<Option<(SessionId, BranchId)>> {
        let (session_prefix, name) = match target.split_once('/') {
            Some((p, n)) => (Some(p.to_string()), n.to_string()),
            None => (None, target.to_string()),
        };
        let prefer_session = prefer_session.map(|s| s.0.clone());
        let row: Option<(String, i64)> = self
            .handle
            .conn()
            .call(move |c| -> rusqlite::Result<_> {
                if let Some(prefix) = session_prefix {
                    let pattern = format!("{prefix}%");
                    let row = c
                        .query_row(
                            "SELECT b.session_id, b.id
                             FROM branches b JOIN sessions s ON s.id = b.session_id
                             WHERE b.name = ?1 AND b.session_id LIKE ?2
                             ORDER BY s.started_at DESC LIMIT 1",
                            rusqlite::params![name, pattern],
                            |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)),
                        )
                        .ok();
                    Ok(row)
                } else if let Some(pref) = prefer_session {
                    if let Ok(row) = c.query_row(
                        "SELECT session_id, id FROM branches WHERE name = ?1 AND session_id = ?2",
                        rusqlite::params![name, pref],
                        |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)),
                    ) {
                        return Ok(Some(row));
                    }
                    Ok(c.query_row(
                        "SELECT b.session_id, b.id
                         FROM branches b JOIN sessions s ON s.id = b.session_id
                         WHERE b.name = ?1
                         ORDER BY s.started_at DESC LIMIT 1",
                        rusqlite::params![name],
                        |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)),
                    )
                    .ok())
                } else {
                    Ok(c.query_row(
                        "SELECT b.session_id, b.id
                         FROM branches b JOIN sessions s ON s.id = b.session_id
                         WHERE b.name = ?1
                         ORDER BY s.started_at DESC LIMIT 1",
                        rusqlite::params![name],
                        |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)),
                    )
                    .ok())
                }
            })
            .await
            .context("resolve_branch")?;
        Ok(row.map(|(s, b)| (SessionId(s), BranchId(b))))
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
            .conn()
            .call(move |c| -> rusqlite::Result<_> {
                let sql = "
                    SELECT  c.id,
                            bm.seq,
                            c.role,
                            c.content,
                            c.tool_calls,
                            c.tool_call_id,
                            c.tool_name
                    FROM branch_messages bm JOIN conversations c ON c.id = bm.conversation_id
                    WHERE bm.branch_id = ?1
                    ORDER BY bm.seq ASC
                ";
                let mut stmt = c.prepare(sql)?;
                let rows = stmt
                    .query_map(rusqlite::params![branch.0], |row| {
                        let role_str: String = row.get(2)?;
                        let tool_calls_str: Option<String> = row.get(4)?;
                        let role = PersistedRole::parse(&role_str).ok_or_else(|| {
                            rusqlite::Error::FromSqlConversionFailure(
                                2,
                                rusqlite::types::Type::Text,
                                Box::new(std::io::Error::other(format!(
                                    "unknown role in DB: {role_str}"
                                ))),
                            )
                        })?;
                        Ok(HistoryRow {
                            conversation_id: row.get(0)?,
                            seq: row.get(1)?,
                            role,
                            content: row.get(3)?,
                            tool_calls: tool_calls_str.map(|s| {
                                serde_json::from_str(&s).unwrap_or(serde_json::Value::Null)
                            }),
                            tool_call_id: row.get(5)?,
                            tool_name: row.get(6)?,
                        })
                    })?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .await
            .context("load_branch_history")
    }

    async fn latest_branch_activity(&self, branch: BranchId) -> Result<Option<String>> {
        self.handle
            .conn()
            .call(move |c| -> rusqlite::Result<_> {
                let ts: Option<String> = c
                    .query_row(
                        "SELECT c.timestamp
                         FROM branch_messages bm
                         JOIN conversations c ON c.id = bm.conversation_id
                         WHERE bm.branch_id = ?1
                         ORDER BY bm.seq DESC LIMIT 1",
                        rusqlite::params![branch.0],
                        |r| r.get::<_, String>(0),
                    )
                    .ok();
                Ok(ts)
            })
            .await
            .context("latest_branch_activity")
    }

    async fn undo_last_turn(&self, branch: BranchId) -> Result<UndoOutcome> {
        dispatch_write(self.handle.writer(), |ack| WriteOp::UndoLastTurn {
            branch_id: branch,
            ack,
        })
        .await
    }

    async fn get_session_title(&self, session: &SessionId) -> Result<Option<String>> {
        let session_id = session.0.clone();
        self.handle
            .conn()
            .call(move |c| -> rusqlite::Result<_> {
                let title: Option<String> = c
                    .query_row(
                        "SELECT title FROM sessions WHERE id = ?1",
                        rusqlite::params![session_id],
                        |r| r.get::<_, Option<String>>(0),
                    )
                    .ok()
                    .flatten();
                Ok(title)
            })
            .await
            .context("get_session_title")
    }

    async fn set_session_title(&self, session: &SessionId, title: &str) -> Result<()> {
        let session_id = session.0.clone();
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
            .conn()
            .call(|c| -> rusqlite::Result<_> {
                let row = c
                    .query_row(
                        "SELECT id, current_branch_id, daemon_pid, started_at
                         FROM sessions
                         WHERE ended_at IS NULL AND current_branch_id IS NOT NULL
                         ORDER BY started_at DESC LIMIT 1",
                        [],
                        |r| {
                            Ok((
                                r.get::<_, String>(0)?,
                                r.get::<_, i64>(1)?,
                                r.get::<_, u32>(2)?,
                                r.get::<_, String>(3)?,
                            ))
                        },
                    )
                    .ok();
                Ok(row)
            })
            .await
            .context("find_resumable_session")
            .map(|row| {
                row.map(|(id, branch, pid, started)| ResumeCandidate {
                    session_id: SessionId(id),
                    current_branch_id: BranchId(branch),
                    daemon_pid: pid,
                    started_at: started,
                })
            })
    }
}

#[cfg(test)]
mod tests;
