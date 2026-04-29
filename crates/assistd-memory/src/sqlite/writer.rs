//! Single-writer task for the SQLite store.
//!
//! Every mutating operation is sent over `mpsc` to one task that owns the
//! connection. SQLite serializes writes through its file lock anyway, so
//! pretending to do them in parallel gains nothing — funneling them
//! through one place gives us a single tracing target, lets the dispatch
//! loop fire-and-forget, and makes shutdown deterministic (we drain the
//! channel before the task exits).
//!
//! All branches return their result through a per-op `oneshot::Sender`.
//! Callers `await` that oneshot when they need the row id or to surface
//! errors; the chat-turn dispatch loop does NOT await it on the hot path
//! — it spawns a tiny logger task so DB latency never throttles token
//! streaming.

use std::sync::Arc;

use anyhow::{Context, Result};
use chrono::Utc;
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::JoinHandle;
use tokio_rusqlite::Connection;

use super::conversations::{PersistedMessage, PersistedRole, TurnId};

/// Typed write operations. Each carries a `oneshot` ack so callers can
/// either await the result (CLI / IPC handlers) or fire-and-forget plus
/// log on error (chat-turn persistence).
pub enum WriteOp {
    BeginSession {
        session_id: String,
        daemon_pid: u32,
        ack: oneshot::Sender<Result<()>>,
    },
    EndSession {
        session_id: String,
        ack: oneshot::Sender<Result<()>>,
    },
    BeginTurn {
        session_id: String,
        user_text: String,
        ack: oneshot::Sender<Result<TurnId>>,
    },
    EndTurn {
        turn_id: TurnId,
        ack: oneshot::Sender<Result<()>>,
    },
    AppendMessage {
        session_id: String,
        turn_id: Option<TurnId>,
        msg: PersistedMessage,
        ack: oneshot::Sender<Result<i64>>,
    },
    SaveMemory {
        key: String,
        value: String,
        source_conversation_id: Option<i64>,
        /// Returns the row id of the inserted/updated memory. Callers
        /// that fire-and-forget `save` (the IPC `MemorySave` handler)
        /// can ignore it; callers that want to enqueue an embed job for
        /// the value (the `RememberTool`) need it to FK the embedding.
        ack: oneshot::Sender<Result<i64>>,
    },
    DeleteMemory {
        key: String,
        ack: oneshot::Sender<Result<()>>,
    },
    /// Delete a memory by row id. Returns the key of the deleted row
    /// on hit so the IPC `MemoryForget` handler can echo it back to
    /// the CLI; returns `None` when no row with that id exists. The
    /// `memory_embeddings.memory_id ... ON DELETE CASCADE` FK cleans
    /// the embedding row in the same statement.
    DeleteMemoryById {
        id: i64,
        ack: oneshot::Sender<Result<Option<String>>>,
    },
    /// Persist one chunk of a conversation message. Returns the chunk
    /// rowid via the ack so the caller can dispatch an embed job.
    StoreChunk {
        conversation_id: i64,
        chunk_index: i64,
        content: String,
        token_count: Option<i64>,
        ack: oneshot::Sender<Result<i64>>,
    },
    /// Store an embedding vector for a `conversation_chunks` row.
    /// Idempotent on `(chunk_id)` via `ON CONFLICT(conversation_chunk_id)
    /// DO UPDATE` — the unique FK column lets a re-embed overwrite.
    StoreChunkEmbedding {
        chunk_id: i64,
        model: String,
        dim: i64,
        vector: Vec<u8>,
        ack: oneshot::Sender<Result<()>>,
    },
    /// Store an embedding vector for a `memories` row. Idempotent on
    /// `(memory_id)` for the same reason as above — when a memory is
    /// re-saved (UPSERT keeps the row id), the embedding refreshes in
    /// place rather than accumulating duplicates.
    StoreMemoryEmbedding {
        memory_id: i64,
        model: String,
        dim: i64,
        vector: Vec<u8>,
        ack: oneshot::Sender<Result<()>>,
    },
    /// Drop every chunk (and cascade-drop its embedding) for one
    /// conversation row. Currently unused — added for the future
    /// "re-chunk" workflow when a user edits or deletes a turn so the
    /// FK shape doesn't need a follow-up migration.
    DeleteChunksForConversation {
        conversation_id: i64,
        ack: oneshot::Sender<Result<()>>,
    },
}

/// Spawn the writer worker. Returns its `JoinHandle` so the daemon
/// shutdown path at `crates/assistd/src/daemon.rs:321-333` can await it
/// alongside the other background workers.
pub fn spawn_writer(
    conn: Connection,
    mut rx: mpsc::Receiver<WriteOp>,
    mut shutdown: watch::Receiver<bool>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            tokio::select! {
                biased;
                op = rx.recv() => {
                    match op {
                        Some(op) => handle_op(&conn, op).await,
                        None => {
                            // Sender dropped: handle nothing else and exit.
                            tracing::debug!(
                                target: "assistd::memory",
                                "writer channel closed; worker exiting"
                            );
                            break;
                        }
                    }
                }
                _ = shutdown.changed() => {
                    if *shutdown.borrow() {
                        // Drain remaining ops before exiting so an
                        // append_message issued just before SIGTERM
                        // still lands.
                        tracing::debug!(
                            target: "assistd::memory",
                            "shutdown received; draining writer queue"
                        );
                        while let Ok(op) = rx.try_recv() {
                            handle_op(&conn, op).await;
                        }
                        break;
                    }
                }
            }
        }
    })
}

async fn handle_op(conn: &Connection, op: WriteOp) {
    match op {
        WriteOp::BeginSession {
            session_id,
            daemon_pid,
            ack,
        } => {
            let res = begin_session(conn, session_id, daemon_pid).await;
            let _ = ack.send(res);
        }
        WriteOp::EndSession { session_id, ack } => {
            let res = end_session(conn, session_id).await;
            let _ = ack.send(res);
        }
        WriteOp::BeginTurn {
            session_id,
            user_text,
            ack,
        } => {
            let res = begin_turn(conn, session_id, user_text).await;
            let _ = ack.send(res);
        }
        WriteOp::EndTurn { turn_id, ack } => {
            let res = end_turn(conn, turn_id).await;
            let _ = ack.send(res);
        }
        WriteOp::AppendMessage {
            session_id,
            turn_id,
            msg,
            ack,
        } => {
            let res = append_message(conn, session_id, turn_id, msg).await;
            let _ = ack.send(res);
        }
        WriteOp::SaveMemory {
            key,
            value,
            source_conversation_id,
            ack,
        } => {
            let res = save_memory(conn, key, value, source_conversation_id).await;
            let _ = ack.send(res);
        }
        WriteOp::DeleteMemory { key, ack } => {
            let res = delete_memory(conn, key).await;
            let _ = ack.send(res);
        }
        WriteOp::DeleteMemoryById { id, ack } => {
            let res = delete_memory_by_id(conn, id).await;
            let _ = ack.send(res);
        }
        WriteOp::StoreChunk {
            conversation_id,
            chunk_index,
            content,
            token_count,
            ack,
        } => {
            let res = store_chunk(conn, conversation_id, chunk_index, content, token_count).await;
            let _ = ack.send(res);
        }
        WriteOp::StoreChunkEmbedding {
            chunk_id,
            model,
            dim,
            vector,
            ack,
        } => {
            let res = store_chunk_embedding(conn, chunk_id, model, dim, vector).await;
            let _ = ack.send(res);
        }
        WriteOp::StoreMemoryEmbedding {
            memory_id,
            model,
            dim,
            vector,
            ack,
        } => {
            let res = store_memory_embedding(conn, memory_id, model, dim, vector).await;
            let _ = ack.send(res);
        }
        WriteOp::DeleteChunksForConversation {
            conversation_id,
            ack,
        } => {
            let res = delete_chunks_for_conversation(conn, conversation_id).await;
            let _ = ack.send(res);
        }
    }
}

async fn begin_session(conn: &Connection, id: String, pid: u32) -> Result<()> {
    let started = Utc::now().to_rfc3339();
    conn.call(move |c| {
        c.execute(
            "INSERT INTO sessions (id, started_at, daemon_pid) VALUES (?1, ?2, ?3)",
            rusqlite::params![id, started, pid],
        )?;
        Ok(())
    })
    .await
    .context("begin_session")
}

async fn end_session(conn: &Connection, id: String) -> Result<()> {
    let ended = Utc::now().to_rfc3339();
    conn.call(move |c| {
        c.execute(
            "UPDATE sessions SET ended_at = ?1 WHERE id = ?2",
            rusqlite::params![ended, id],
        )?;
        Ok(())
    })
    .await
    .context("end_session")
}

async fn begin_turn(conn: &Connection, session: String, user_text: String) -> Result<TurnId> {
    let started = Utc::now().to_rfc3339();
    let id = conn
        .call(move |c| {
            c.execute(
                "INSERT INTO turns (session_id, started_at, user_text) VALUES (?1, ?2, ?3)",
                rusqlite::params![session, started, user_text],
            )?;
            Ok(c.last_insert_rowid())
        })
        .await
        .context("begin_turn")?;
    Ok(TurnId(id))
}

async fn end_turn(conn: &Connection, turn: TurnId) -> Result<()> {
    let ended = Utc::now().to_rfc3339();
    conn.call(move |c| {
        c.execute(
            "UPDATE turns SET ended_at = ?1 WHERE id = ?2",
            rusqlite::params![ended, turn.0],
        )?;
        Ok(())
    })
    .await
    .context("end_turn")
}

async fn append_message(
    conn: &Connection,
    session: String,
    turn: Option<TurnId>,
    msg: PersistedMessage,
) -> Result<i64> {
    let timestamp = Utc::now().to_rfc3339();
    let role = msg.role.as_wire().to_string();
    let tool_calls_json = match msg.tool_calls {
        Some(v) => Some(serde_json::to_string(&v).context("serialize tool_calls")?),
        None => None,
    };
    let turn_id = turn.map(|t| t.0);
    let id = conn
        .call(move |c| {
            // seq is per-session monotonic. Compute it inside the same
            // implicit transaction as the INSERT so concurrent writers
            // (none today, but cheap insurance) cannot collide.
            let seq: i64 = c
                .query_row(
                    "SELECT COALESCE(MAX(seq), -1) + 1 FROM conversations WHERE session_id = ?1",
                    rusqlite::params![session],
                    |r| r.get(0),
                )
                .unwrap_or(0);
            c.execute(
                "INSERT INTO conversations
                    (session_id, turn_id, seq, timestamp, role, content, tool_calls, tool_call_id, tool_name)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                rusqlite::params![
                    session,
                    turn_id,
                    seq,
                    timestamp,
                    role,
                    msg.content,
                    tool_calls_json,
                    msg.tool_call_id,
                    msg.tool_name,
                ],
            )?;
            Ok(c.last_insert_rowid())
        })
        .await
        .context("append_message")?;
    Ok(id)
}

async fn save_memory(
    conn: &Connection,
    key: String,
    value: String,
    source: Option<i64>,
) -> Result<i64> {
    let now = Utc::now().to_rfc3339();
    // `RETURNING id` (SQLite >= 3.35) gives us the row id of either the
    // freshly-inserted row or the row updated via ON CONFLICT. Saves an
    // extra `SELECT id FROM memories WHERE key = ?` round-trip — and the
    // caller (`RememberTool`) needs the id to FK an embedding row.
    let id = conn
        .call(move |c| {
            let id: i64 = c.query_row(
                "INSERT INTO memories (key, value, source_conversation_id, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?4)
                 ON CONFLICT(key) DO UPDATE SET
                     value = excluded.value,
                     source_conversation_id = excluded.source_conversation_id,
                     updated_at = excluded.updated_at
                 RETURNING id",
                rusqlite::params![key, value, source, now],
                |r| r.get(0),
            )?;
            Ok(id)
        })
        .await
        .context("save_memory")?;
    Ok(id)
}

async fn delete_memory(conn: &Connection, key: String) -> Result<()> {
    conn.call(move |c| {
        c.execute(
            "DELETE FROM memories WHERE key = ?1",
            rusqlite::params![key],
        )?;
        Ok(())
    })
    .await
    .context("delete_memory")
}

async fn delete_memory_by_id(conn: &Connection, id: i64) -> Result<Option<String>> {
    // `RETURNING key` (SQLite >= 3.35) gives us the deleted row's key
    // in the same round trip; `QueryReturnedNoRows` means the id
    // didn't exist. The `memory_embeddings.memory_id` FK has
    // `ON DELETE CASCADE`, so the embedding row drops with the memory
    // — no second statement needed.
    conn.call(move |c| {
        let result = c
            .query_row(
                "DELETE FROM memories WHERE id = ?1 RETURNING key",
                rusqlite::params![id],
                |r| r.get::<_, String>(0),
            )
            .map(Some)
            .or_else(|e| {
                if matches!(e, rusqlite::Error::QueryReturnedNoRows) {
                    Ok(None)
                } else {
                    Err(e)
                }
            })?;
        Ok(result)
    })
    .await
    .context("delete_memory_by_id")
}

async fn store_chunk(
    conn: &Connection,
    conversation_id: i64,
    chunk_index: i64,
    content: String,
    token_count: Option<i64>,
) -> Result<i64> {
    let id = conn
        .call(move |c| {
            // Idempotent on (conversation_id, chunk_index): re-running
            // chunking for the same row replaces in place.
            let id: i64 = c.query_row(
                "INSERT INTO conversation_chunks (conversation_id, chunk_index, content, token_count)
                 VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(conversation_id, chunk_index) DO UPDATE SET
                     content = excluded.content,
                     token_count = excluded.token_count
                 RETURNING id",
                rusqlite::params![conversation_id, chunk_index, content, token_count],
                |r| r.get(0),
            )?;
            Ok(id)
        })
        .await
        .context("store_chunk")?;
    Ok(id)
}

async fn store_chunk_embedding(
    conn: &Connection,
    chunk_id: i64,
    model: String,
    dim: i64,
    vector: Vec<u8>,
) -> Result<()> {
    let now = Utc::now().to_rfc3339();
    conn.call(move |c| {
        c.execute(
            "INSERT INTO embeddings (conversation_chunk_id, model, dim, vector, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(conversation_chunk_id) DO UPDATE SET
                 model = excluded.model,
                 dim = excluded.dim,
                 vector = excluded.vector,
                 created_at = excluded.created_at",
            rusqlite::params![chunk_id, model, dim, vector, now],
        )?;
        Ok(())
    })
    .await
    .context("store_chunk_embedding")
}

async fn store_memory_embedding(
    conn: &Connection,
    memory_id: i64,
    model: String,
    dim: i64,
    vector: Vec<u8>,
) -> Result<()> {
    let now = Utc::now().to_rfc3339();
    conn.call(move |c| {
        c.execute(
            "INSERT INTO memory_embeddings (memory_id, model, dim, vector, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(memory_id) DO UPDATE SET
                 model = excluded.model,
                 dim = excluded.dim,
                 vector = excluded.vector,
                 created_at = excluded.created_at",
            rusqlite::params![memory_id, model, dim, vector, now],
        )?;
        Ok(())
    })
    .await
    .context("store_memory_embedding")
}

async fn delete_chunks_for_conversation(conn: &Connection, conversation_id: i64) -> Result<()> {
    conn.call(move |c| {
        // ON DELETE CASCADE on the embeddings FK takes care of dropping
        // any matching `embeddings` rows.
        c.execute(
            "DELETE FROM conversation_chunks WHERE conversation_id = ?1",
            rusqlite::params![conversation_id],
        )?;
        Ok(())
    })
    .await
    .context("delete_chunks_for_conversation")
}

/// Re-export the role wire mapping used by [`append_message`] so callers
/// that build [`PersistedMessage`] from a `Role` enum elsewhere can do
/// the same translation in one place.
#[allow(dead_code)]
pub(super) fn role_wire(role: PersistedRole) -> &'static str {
    role.as_wire()
}

/// Helper used by other modules to send a [`WriteOp`] without unwrapping
/// the `Result<oneshot::Receiver<...>>` boilerplate at every call site.
pub(super) async fn dispatch<T>(
    tx: &mpsc::Sender<WriteOp>,
    op: WriteOp,
    ack_rx: oneshot::Receiver<Result<T>>,
) -> Result<T> {
    tx.send(op)
        .await
        .map_err(|_| anyhow::anyhow!("memory writer task is gone"))?;
    ack_rx
        .await
        .map_err(|_| anyhow::anyhow!("memory writer task dropped ack channel"))?
}

/// Convenience: every public method on the stores follows the same
/// shape — build oneshot, build op, call `dispatch`. Encoded once here
/// to keep the call sites readable.
pub(super) struct WriteCall;

#[allow(dead_code)]
impl WriteCall {
    pub(super) async fn run<T, F>(tx: &mpsc::Sender<WriteOp>, build: F) -> Result<T>
    where
        F: FnOnce(oneshot::Sender<Result<T>>) -> WriteOp,
    {
        let (ack_tx, ack_rx) = oneshot::channel();
        dispatch(tx, build(ack_tx), ack_rx).await
    }
}

/// Cheap clone alias used by the stores so they can share one
/// `mpsc::Sender<WriteOp>` without each owning a separate one.
pub type WriterSender = Arc<mpsc::Sender<WriteOp>>;
