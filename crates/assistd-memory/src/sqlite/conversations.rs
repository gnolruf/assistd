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
    pub fn as_wire(self) -> &'static str {
        match self {
            PersistedRole::System => "system",
            PersistedRole::User => "user",
            PersistedRole::Assistant => "assistant",
            PersistedRole::Tool => "tool",
        }
    }

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

/// Conversation persistence trait. Sibling to [`crate::MemoryStore`];
/// implementations may share underlying storage (the SQLite impls
/// below do — both hold an `Arc<SqliteHandle>`).
#[async_trait]
pub trait ConversationStore: Send + Sync + 'static {
    async fn begin_session(&self, daemon_pid: u32) -> Result<SessionId>;
    async fn end_session(&self, id: &SessionId) -> Result<()>;
    async fn begin_turn(&self, session: &SessionId, user_text: &str) -> Result<TurnId>;
    async fn end_turn(&self, turn: TurnId) -> Result<()>;
    async fn append_message(
        &self,
        session: &SessionId,
        turn: Option<TurnId>,
        msg: PersistedMessage,
    ) -> Result<i64>;
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchHit>>;
    async fn recent_turns(&self, limit: usize) -> Result<Vec<TurnSummary>>;
}

/// No-op fallback used when memory is disabled in config or in tests
/// that don't exercise persistence.
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
}

/// SQLite-backed implementation. Holds an `Arc<SqliteHandle>` so it
/// shares the connection + writer with [`super::SqliteMemoryStore`].
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
        // Reads bypass the writer channel — SQLite in WAL mode handles
        // concurrent readers fine, and routing them through the writer
        // would queue them behind any in-flight inserts.
        let q = query.to_string();
        let limit = limit as i64;
        self.handle
            .conn()
            .call(move |c| {
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
            .call(move |c| {
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
}

// `WriteCall::run` returns an `oneshot::Receiver<Result<T>>` style
// future via `dispatch`; one call site needs the `oneshot` import even
// though most uses are inside the helper itself. Suppress the unused
// import lint via direct reference.
#[allow(dead_code)]
fn _force_oneshot_referenced(_: oneshot::Receiver<Result<()>>) {}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::watch;

    async fn fresh_store() -> (SqliteConversationStore, tokio::task::JoinHandle<()>) {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("memory.db");
        // Leak the tempdir for the duration of the test — `path` must
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
            .call(move |c| {
                Ok(c.query_row(
                    "SELECT (SELECT tool_calls FROM conversations WHERE id = ?1),
                            (SELECT tool_call_id FROM conversations WHERE id = ?2),
                            (SELECT tool_name FROM conversations WHERE id = ?2)",
                    rusqlite::params![id, result_id],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
                )?)
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
}
