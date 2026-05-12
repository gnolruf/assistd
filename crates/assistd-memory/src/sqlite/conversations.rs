//! Conversation persistence: sessions, turns, messages, and FTS5 search.
//!
//! [`ConversationStore`] is a sibling trait to [`crate::MemoryStore`]:
//! the latter is a flat string-keyed KV; this one stores the richer
//! shape that the agent loop produces. Both implementations share a
//! single [`super::SqliteHandle`] so all writes go through the same
//! background writer.

use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;
use uuid::Uuid;

use super::connection::SqliteHandle;
use super::writer::{WriteCall, WriteOp};

/// Opaque session identifier. Stored as a uuid string so it stays
/// stable across daemon restarts (no risk of an in-memory rowid
/// recycling onto a stored ended_at).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(pub String);

impl SessionId {
    /// Generate a new random [`SessionId`] using UUIDv4.
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Borrow the inner UUID string.
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

/// Role of a persisted message. Mirrors `assistd_llm::chat::conversation::Role`
/// plus a `Tool` variant for tool-result rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PersistedRole {
    System,
    User,
    Assistant,
    Tool,
}

impl PersistedRole {
    /// Return the lowercase wire string stored in the `conversations.role` column.
    pub fn as_wire(self) -> &'static str {
        match self {
            PersistedRole::System => "system",
            PersistedRole::User => "user",
            PersistedRole::Assistant => "assistant",
            PersistedRole::Tool => "tool",
        }
    }

    /// Parse a wire string from the `conversations.role` column, returning
    /// `None` for unrecognised values.
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

/// One message ready to write to the `conversations` table. Built at
/// the persistence boundary in `state.rs::handle_query` from the
/// streaming `LlmEvent`s.
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
    /// Build a user-role message with no tool fields set.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: PersistedRole::User,
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
            tool_name: None,
        }
    }

    /// Build a plain assistant text message with no tool fields set.
    pub fn assistant_text(content: impl Into<String>) -> Self {
        Self {
            role: PersistedRole::Assistant,
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
            tool_name: None,
        }
    }

    /// Build an assistant message that carries tool-call JSON but no text content.
    pub fn assistant_tool_calls(calls: serde_json::Value) -> Self {
        Self {
            role: PersistedRole::Assistant,
            content: String::new(),
            tool_calls: Some(calls),
            tool_call_id: None,
            tool_name: None,
        }
    }

    /// Build a tool-result message with the given content, call id, and tool name.
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

/// One FTS5 hit returned by [`ConversationStore::search`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SearchHit {
    pub conversation_id: i64,
    pub session_id: String,
    pub timestamp: String,
    pub role: PersistedRole,
    /// FTS5 `snippet()` output: a short excerpt with match-highlighting
    /// markers around the search terms. Format is
    /// `…before <mark>match</mark> after…`.
    pub snippet: String,
}

/// Coarse summary of one turn. Used by `assistd memory sessions` and
/// future "show recent turns" UIs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TurnSummary {
    pub turn_id: i64,
    pub session_id: String,
    pub started_at: String,
    pub ended_at: Option<String>,
    pub user_text: String,
    pub message_count: i64,
}

/// Per-branch metadata returned by [`ConversationStore::list_branches`].
/// `is_current_in_session` flags the branch that `sessions.current_branch_id`
/// points at; the active session can be cross-referenced via the
/// daemon-held [`SessionId`].
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

/// One persisted message reconstructed from the DB for replay into the
/// LLM backend's in-memory conversation. Carries enough fields to
/// faithfully reproduce both plain user/assistant rows AND
/// assistant-with-tool-calls / tool-result rows.
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

/// Result of [`ConversationStore::undo_last_turn`]. `removed_messages`
/// is the count of `branch_messages` rows dropped from the branch (so
/// the TUI can render "removed N entries" feedback). `last_user_text`
/// echoes back the user prompt that was undone, used by callers that
/// want to surface "undone: <text>".
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct UndoOutcome {
    pub removed_messages: u32,
    pub last_user_text: Option<String>,
    pub removed_turn_id: Option<i64>,
}

/// Conversation persistence trait. Sibling to [`crate::MemoryStore`];
/// implementations may share underlying storage (the SQLite impls
/// below do; both hold an `Arc<SqliteHandle>`).
#[async_trait]
pub trait ConversationStore: Send + Sync + 'static {
    /// Open a new session row for the daemon process identified by `daemon_pid`.
    async fn begin_session(&self, daemon_pid: u32) -> Result<SessionId>;
    /// Mark `id` as ended by stamping `ended_at`.
    async fn end_session(&self, id: &SessionId) -> Result<()>;
    /// Open a new turn row inside `session` labelled with `user_text`.
    async fn begin_turn(&self, session: &SessionId, user_text: &str) -> Result<TurnId>;
    /// Mark `turn` as ended by stamping `ended_at`.
    async fn end_turn(&self, turn: TurnId) -> Result<()>;
    /// Append `msg` to the `conversations` table. Returns the new `conversations.id` rowid.
    async fn append_message(
        &self,
        session: &SessionId,
        turn: Option<TurnId>,
        msg: PersistedMessage,
    ) -> Result<i64>;
    /// Full-text search via the FTS5 index. Returns up to `limit` hits ordered by relevance.
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchHit>>;
    /// Return the `limit` most-recent turns ordered by turn id descending.
    async fn recent_turns(&self, limit: usize) -> Result<Vec<TurnSummary>>;

    /// Atomically begin a session and create its default `main` branch.
    /// Returns both ids; the caller stores them in `AppState`. Replaces
    /// the [`Self::begin_session`] call at daemon startup so a crash
    /// between the two writes can't orphan a session without a branch.
    async fn begin_session_with_main_branch(
        &self,
        daemon_pid: u32,
    ) -> Result<(SessionId, BranchId)>;

    /// Create a new branch in `session` with the given name. Used by
    /// `begin_session_with_main_branch` internally and by direct
    /// branch-creation paths during recovery / tests.
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

    /// Append `msg` to the conversations table AND to the
    /// `branch_messages` join under `branch`. Atomic via a single
    /// transaction. Returns the conversations.id rowid.
    async fn append_message_to_branch(
        &self,
        session: &SessionId,
        branch: BranchId,
        turn: Option<TurnId>,
        msg: PersistedMessage,
    ) -> Result<i64>;

    /// Enumerate every branch across every session. Sorted by
    /// session.started_at DESC, then branches.id ASC. The caller
    /// (handle_branches) re-orders so the active session shows first.
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

    /// Load every message row referenced by `branch`, ordered by
    /// `branch_messages.seq`. Used by `/switch` and by daemon-startup
    /// resume to repopulate the in-memory `Conversation`.
    async fn load_branch_history(&self, branch: BranchId) -> Result<Vec<HistoryRow>>;

    /// Return the RFC3339 timestamp of the most recent message on
    /// `branch`, or `None` when the branch has no messages. Used by
    /// [`Request::ResumeOrNew`] to decide whether to replay the current
    /// branch or start a fresh session.
    async fn latest_branch_activity(&self, branch: BranchId) -> Result<Option<String>>;

    /// Drop the latest turn from `branch`. Implementation:
    /// 1. find max(turn_id) for messages reachable from `branch`,
    /// 2. delete the matching `branch_messages` rows for `branch`,
    /// 3. orphan-sweep `conversations` rows now unreferenced by ANY
    ///    `branch_messages`,
    /// 4. delete the matching `turns` row when no other branch still
    ///    references it.
    /// Returns count + last user_text + the dropped turn id (when any).
    async fn undo_last_turn(&self, branch: BranchId) -> Result<UndoOutcome>;

    /// Find the most recent session with `ended_at IS NULL` whose
    /// `daemon_pid` is no longer alive. The caller then resumes by
    /// loading `current_branch_id`'s messages back into memory. Returns
    /// `None` when nothing is resumable (first-ever startup, or the
    /// only candidate is owned by a still-running daemon).
    async fn find_resumable_session(&self) -> Result<Option<ResumeCandidate>>;

    /// Return the current `sessions.title` for `session`, or `None`
    /// when no title has been generated yet. Used by the daemon's
    /// title-generation hook to decide whether to call the LLM.
    async fn get_session_title(&self, session: &SessionId) -> Result<Option<String>>;

    /// Persist `title` as the human-readable summary for `session`.
    /// Issued fire-and-forget by the title-generation task after the
    /// first agent response completes.
    async fn set_session_title(&self, session: &SessionId, title: &str) -> Result<()>;
}

/// Candidate session returned by [`ConversationStore::find_resumable_session`].
/// `daemon_pid` lets the caller decide whether the prior owner is still
/// alive (`kill -0`) before claiming the session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResumeCandidate {
    pub session_id: SessionId,
    pub current_branch_id: BranchId,
    pub daemon_pid: u32,
    pub started_at: String,
}

/// Successful-no-op fallback used when memory is disabled in config or in
/// tests that don't exercise persistence.
pub struct NoConversationStore;

#[async_trait]
impl ConversationStore for NoConversationStore {
    async fn begin_session(&self, _pid: u32) -> Result<SessionId> {
        Ok(SessionId::new())
    }
    async fn end_session(&self, _id: &SessionId) -> Result<()> {
        Ok(())
    }
    async fn begin_turn(&self, _s: &SessionId, _t: &str) -> Result<TurnId> {
        Ok(TurnId(0))
    }
    async fn end_turn(&self, _t: TurnId) -> Result<()> {
        Ok(())
    }
    async fn append_message(
        &self,
        _s: &SessionId,
        _t: Option<TurnId>,
        _m: PersistedMessage,
    ) -> Result<i64> {
        Ok(0)
    }
    async fn search(&self, _q: &str, _l: usize) -> Result<Vec<SearchHit>> {
        Ok(Vec::new())
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

/// SQLite-backed implementation. Holds an `Arc<SqliteHandle>` so it
/// shares the connection + writer with [`super::SqliteMemoryStore`].
#[derive(Clone)]
pub struct SqliteConversationStore {
    handle: Arc<SqliteHandle>,
}

impl SqliteConversationStore {
    /// Create a new store sharing `handle` with other store types.
    pub fn new(handle: Arc<SqliteHandle>) -> Self {
        Self { handle }
    }
}

#[async_trait]
impl ConversationStore for SqliteConversationStore {
    async fn begin_session(&self, daemon_pid: u32) -> Result<SessionId> {
        let id = SessionId::new();
        let session_id = id.0.clone();
        WriteCall::run(self.handle.writer(), |ack| WriteOp::BeginSession {
            session_id,
            daemon_pid,
            ack,
        })
        .await?;
        Ok(id)
    }

    async fn end_session(&self, id: &SessionId) -> Result<()> {
        let session_id = id.0.clone();
        WriteCall::run(self.handle.writer(), |ack| WriteOp::EndSession {
            session_id,
            ack,
        })
        .await
    }

    async fn begin_turn(&self, session: &SessionId, user_text: &str) -> Result<TurnId> {
        let session_id = session.0.clone();
        let text = user_text.to_string();
        WriteCall::run(self.handle.writer(), |ack| WriteOp::BeginTurn {
            session_id,
            user_text: text,
            ack,
        })
        .await
    }

    async fn end_turn(&self, turn: TurnId) -> Result<()> {
        WriteCall::run(self.handle.writer(), |ack| WriteOp::EndTurn {
            turn_id: turn,
            ack,
        })
        .await
    }

    async fn append_message(
        &self,
        session: &SessionId,
        turn: Option<TurnId>,
        msg: PersistedMessage,
    ) -> Result<i64> {
        let session_id = session.0.clone();
        WriteCall::run(self.handle.writer(), |ack| WriteOp::AppendMessage {
            session_id,
            turn_id: turn,
            msg,
            ack,
        })
        .await
    }

    async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchHit>> {
        // Reads bypass the writer channel; SQLite in WAL mode handles
        // concurrent readers fine, and routing them through the writer
        // would queue them behind any in-flight inserts.
        //
        // Wrap the user-supplied query in an FTS5 literal phrase so
        // grammar metacharacters (`*`, `+`, `-`, `OR`, quotes, …) are
        // matched as text rather than parsed as operators. Today only
        // tests call this method, but future callers (CLI re-exposure,
        // an LLM tool) inherit safe-by-default behaviour. A future
        // `search_raw` sibling can opt back into FTS5 grammar.
        let q = fts5_literal(query);
        let limit = limit as i64;
        self.handle
            .conn()
            .call(move |c| -> rusqlite::Result<_> {
                let sql = "
                    SELECT  conv.id,
                            conv.session_id,
                            conv.timestamp,
                            conv.role,
                            snippet(conversations_fts, 0, '<mark>', '</mark>', '…', 16)
                    FROM conversations_fts
                    JOIN conversations conv ON conv.id = conversations_fts.rowid
                    WHERE conversations_fts MATCH ?1
                    ORDER BY rank
                    LIMIT ?2
                ";
                let mut stmt = c.prepare(sql)?;
                let rows = stmt
                    .query_map(rusqlite::params![q, limit], |row| {
                        Ok((
                            row.get::<_, i64>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, String>(2)?,
                            row.get::<_, String>(3)?,
                            row.get::<_, String>(4)?,
                        ))
                    })?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .await
            .context("conversation search")?
            .into_iter()
            .map(|(id, session_id, ts, role, snippet)| {
                let role = PersistedRole::parse(&role)
                    .with_context(|| format!("unknown role in DB: {role}"))?;
                Ok(SearchHit {
                    conversation_id: id,
                    session_id,
                    timestamp: ts,
                    role,
                    snippet,
                })
            })
            .collect()
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
        let branch = WriteCall::run(self.handle.writer(), |ack| {
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
        WriteCall::run(self.handle.writer(), |ack| WriteOp::CreateBranch {
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
        WriteCall::run(self.handle.writer(), |ack| WriteOp::SetCurrentBranch {
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
        WriteCall::run(self.handle.writer(), |ack| WriteOp::AppendMessageToBranch {
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
        // Accept "session_prefix/name" (8-char hex prefix) or just
        // "name". Bare name disambiguates by preferring the active
        // session, then by most-recent session.
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
                    // Look first inside the preferred session, then fall
                    // back to "any session, most-recent first".
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
        WriteCall::run(self.handle.writer(), |ack| WriteOp::ForkBranch {
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
        WriteCall::run(self.handle.writer(), |ack| WriteOp::UndoLastTurn {
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
        WriteCall::run(self.handle.writer(), |ack| WriteOp::SetSessionTitle {
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

// `WriteCall::run` returns an `oneshot::Receiver<Result<T>>` style
// future via `dispatch`; one call site needs the `oneshot` import even
// though most uses are inside the helper itself. Suppress the unused
// import lint via direct reference.
#[allow(dead_code)]
fn _force_oneshot_referenced(_: oneshot::Receiver<Result<()>>) {}

/// Wrap a user-supplied query as an FTS5 literal phrase by surrounding
/// it with double-quotes and doubling any internal quotes (per FTS5's
/// quoted-string rules). The result is safe to paste into a `MATCH`
/// expression as a single phrase token.
fn fts5_literal(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        if ch == '"' {
            out.push('"');
            out.push('"');
        } else {
            out.push(ch);
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::watch;

    async fn fresh_store() -> (SqliteConversationStore, tokio::task::JoinHandle<()>) {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("memory.db");
        // Leak the tempdir for the duration of the test; `path` must
        // outlive the handle. Cleanup happens on test process exit.
        std::mem::forget(temp);
        let (_tx, rx) = watch::channel(false);
        let (handle, writer) = SqliteHandle::open(&path, rx).await.unwrap();
        (SqliteConversationStore::new(Arc::new(handle)), writer)
    }

    #[tokio::test]
    async fn round_trip_user_and_assistant_messages() {
        let (store, _w) = fresh_store().await;
        let session = store.begin_session(42).await.unwrap();
        let turn = store.begin_turn(&session, "what is 2+2?").await.unwrap();

        store
            .append_message(&session, Some(turn), PersistedMessage::user("what is 2+2?"))
            .await
            .unwrap();
        store
            .append_message(
                &session,
                Some(turn),
                PersistedMessage::assistant_text("four"),
            )
            .await
            .unwrap();
        store.end_turn(turn).await.unwrap();
        store.end_session(&session).await.unwrap();

        let hits = store.search("2+2", 10).await.unwrap();
        assert!(
            hits.iter().any(|h| matches!(h.role, PersistedRole::User)),
            "expected a user hit: {hits:#?}"
        );

        let recent = store.recent_turns(5).await.unwrap();
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].user_text, "what is 2+2?");
        assert_eq!(recent[0].message_count, 2);
    }

    #[tokio::test]
    async fn assistant_with_tool_calls_persists_json() {
        let (store, _w) = fresh_store().await;
        let session = store.begin_session(1).await.unwrap();
        let turn = store.begin_turn(&session, "list files").await.unwrap();

        let calls =
            serde_json::json!([{"id": "c-1", "name": "run", "arguments": {"command": "ls"}}]);
        let id = store
            .append_message(
                &session,
                Some(turn),
                PersistedMessage::assistant_tool_calls(calls.clone()),
            )
            .await
            .unwrap();
        let result_id = store
            .append_message(
                &session,
                Some(turn),
                PersistedMessage::tool_result("file1\nfile2", "c-1", "run"),
            )
            .await
            .unwrap();
        assert_ne!(id, result_id);

        // Read back: the assistant row should have non-NULL tool_calls
        // and the tool row should carry tool_call_id and tool_name.
        let conn = store.handle.conn();
        let (assistant_calls, tool_call_id, tool_name): (
            Option<String>,
            Option<String>,
            Option<String>,
        ) = conn
            .call(move |c| -> rusqlite::Result<_> {
                c.query_row(
                    "SELECT (SELECT tool_calls FROM conversations WHERE id = ?1),
                            (SELECT tool_call_id FROM conversations WHERE id = ?2),
                            (SELECT tool_name FROM conversations WHERE id = ?2)",
                    rusqlite::params![id, result_id],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
                )
            })
            .await
            .unwrap();
        assert!(assistant_calls.unwrap().contains("\"command\":\"ls\""));
        assert_eq!(tool_call_id.as_deref(), Some("c-1"));
        assert_eq!(tool_name.as_deref(), Some("run"));
    }

    #[tokio::test]
    async fn no_conversation_store_search_returns_empty() {
        let store = NoConversationStore;
        assert!(store.search("anything", 10).await.unwrap().is_empty());
        assert!(store.recent_turns(10).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn search_handles_fts5_grammar_safely() {
        // Inputs that would parse-error under raw FTS5 grammar must
        // round-trip through the literal-phrase escape without
        // surfacing as Err. We don't assert on hits; the default
        // tokenizer's behaviour on these inputs is implementation
        // detail; we only assert the call succeeds.
        let (store, _w) = fresh_store().await;
        let session = store.begin_session(1).await.unwrap();
        let turn = store.begin_turn(&session, "ignore").await.unwrap();
        store
            .append_message(&session, Some(turn), PersistedMessage::user("she said hi"))
            .await
            .unwrap();

        // Unmatched quote: would be a parse error pre-escape.
        store.search("say \"hi", 10).await.unwrap();
        // Operator-looking input: would be parsed as boolean OR.
        store.search("apple OR banana", 10).await.unwrap();
        // Wildcard star: would be a prefix match pre-escape.
        store.search("foo*", 10).await.unwrap();
    }

    #[test]
    fn fts5_literal_doubles_internal_quotes() {
        assert_eq!(fts5_literal("foo"), "\"foo\"");
        assert_eq!(fts5_literal("say \"hi\""), "\"say \"\"hi\"\"\"");
        assert_eq!(fts5_literal(""), "\"\"");
    }

    #[tokio::test]
    async fn begin_session_with_main_branch_inserts_session_and_main_branch() {
        let (store, _w) = fresh_store().await;
        let (session, branch) = store.begin_session_with_main_branch(123).await.unwrap();
        let current = store.get_current_branch(&session).await.unwrap();
        assert_eq!(current, Some(branch));
        let branches = store.list_branches().await.unwrap();
        assert_eq!(branches.len(), 1);
        assert_eq!(branches[0].name, "main");
        assert!(branches[0].is_current_in_session);
        assert_eq!(branches[0].parent_branch_id, None);
        assert_eq!(branches[0].message_count, 0);
    }

    #[tokio::test]
    async fn fork_creates_independent_branch_sharing_history() {
        let (store, _w) = fresh_store().await;
        let (session, main) = store.begin_session_with_main_branch(1).await.unwrap();
        let turn = store.begin_turn(&session, "hello").await.unwrap();
        store
            .append_message_to_branch(&session, main, Some(turn), PersistedMessage::user("hello"))
            .await
            .unwrap();
        store
            .append_message_to_branch(
                &session,
                main,
                Some(turn),
                PersistedMessage::assistant_text("hi"),
            )
            .await
            .unwrap();
        store.end_turn(turn).await.unwrap();

        let fork = store.fork_branch(main, "experiment").await.unwrap();
        let main_history = store.load_branch_history(main).await.unwrap();
        let fork_history = store.load_branch_history(fork).await.unwrap();
        assert_eq!(main_history.len(), 2);
        assert_eq!(fork_history.len(), 2);
        // Same conversation rows are referenced; no row duplication.
        assert_eq!(
            main_history
                .iter()
                .map(|r| r.conversation_id)
                .collect::<Vec<_>>(),
            fork_history
                .iter()
                .map(|r| r.conversation_id)
                .collect::<Vec<_>>()
        );
        let branches = store.list_branches().await.unwrap();
        let fork_info = branches.iter().find(|b| b.name == "experiment").unwrap();
        assert_eq!(fork_info.parent_branch_name.as_deref(), Some("main"));
        assert_eq!(fork_info.fork_point_seq, Some(1));
        assert_eq!(fork_info.message_count, 2);
    }

    #[tokio::test]
    async fn append_to_one_branch_does_not_show_on_the_other() {
        let (store, _w) = fresh_store().await;
        let (session, main) = store.begin_session_with_main_branch(1).await.unwrap();
        let turn = store.begin_turn(&session, "q").await.unwrap();
        store
            .append_message_to_branch(&session, main, Some(turn), PersistedMessage::user("q"))
            .await
            .unwrap();
        let fork = store.fork_branch(main, "alt").await.unwrap();
        // Append to fork only.
        let alt_turn = store.begin_turn(&session, "alt q").await.unwrap();
        store
            .append_message_to_branch(
                &session,
                fork,
                Some(alt_turn),
                PersistedMessage::user("alt q"),
            )
            .await
            .unwrap();
        let main_history = store.load_branch_history(main).await.unwrap();
        let fork_history = store.load_branch_history(fork).await.unwrap();
        assert_eq!(main_history.len(), 1);
        assert_eq!(fork_history.len(), 2);
    }

    #[tokio::test]
    async fn undo_removes_last_turn_from_branch_only() {
        let (store, _w) = fresh_store().await;
        let (session, main) = store.begin_session_with_main_branch(1).await.unwrap();
        // Two turns: "first", "second".
        let t1 = store.begin_turn(&session, "first").await.unwrap();
        store
            .append_message_to_branch(&session, main, Some(t1), PersistedMessage::user("first"))
            .await
            .unwrap();
        store
            .append_message_to_branch(
                &session,
                main,
                Some(t1),
                PersistedMessage::assistant_text("a"),
            )
            .await
            .unwrap();
        store.end_turn(t1).await.unwrap();

        let t2 = store.begin_turn(&session, "second").await.unwrap();
        store
            .append_message_to_branch(&session, main, Some(t2), PersistedMessage::user("second"))
            .await
            .unwrap();
        store
            .append_message_to_branch(
                &session,
                main,
                Some(t2),
                PersistedMessage::assistant_text("b"),
            )
            .await
            .unwrap();
        store.end_turn(t2).await.unwrap();

        let outcome = store.undo_last_turn(main).await.unwrap();
        assert_eq!(outcome.removed_messages, 2);
        assert_eq!(outcome.last_user_text.as_deref(), Some("second"));

        let history = store.load_branch_history(main).await.unwrap();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].content, "first");
        assert_eq!(history[1].content, "a");
    }

    #[tokio::test]
    async fn resolve_branch_qualified_form() {
        let (store, _w) = fresh_store().await;
        let (s1, _) = store.begin_session_with_main_branch(1).await.unwrap();
        let (s2, _) = store.begin_session_with_main_branch(2).await.unwrap();
        // Two sessions, both with a "main" branch.
        let bare = store.resolve_branch("main", Some(&s1)).await.unwrap();
        assert!(bare.is_some());
        assert_eq!(bare.unwrap().0, s1);
        let prefix = &s2.0[..8];
        let qualified = store
            .resolve_branch(&format!("{prefix}/main"), None)
            .await
            .unwrap();
        assert!(qualified.is_some());
        assert_eq!(qualified.unwrap().0, s2);
    }

    #[tokio::test]
    async fn find_resumable_session_returns_unended() {
        let (store, _w) = fresh_store().await;
        let (s, _) = store.begin_session_with_main_branch(99).await.unwrap();
        let resume = store.find_resumable_session().await.unwrap();
        assert!(resume.is_some());
        let cand = resume.unwrap();
        assert_eq!(cand.session_id, s);
        assert_eq!(cand.daemon_pid, 99);
        store.end_session(&s).await.unwrap();
        // Once ended, no longer resumable.
        let resume2 = store.find_resumable_session().await.unwrap();
        assert!(resume2.is_none());
    }
}
