//! Single-writer task for the SQLite store: every mutation is a [`WriteOp`]
//! carrying a `oneshot` ack, executed in order by the one task that owns writes.

use std::num::NonZeroUsize;
use std::ops::Range;
use std::time::Duration;

use rusqlite::{OptionalExtension, Transaction};
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::JoinHandle;
use tokio_rusqlite::Connection;

use super::embeddings::{Embedding, EmbeddingTarget, vector_to_blob};
use super::ids::{BranchId, ChunkId, MemoryId, MessageId, SessionId, TurnId};
use super::timestamp::now_millis;
use super::transcripts::{MessageKind, PersistedMessage, PersistedRole, UndoOutcome};
use crate::{MemoryError, Result};

const DRAIN_IDLE_TIMEOUT: Duration = Duration::from_secs(2);

/// Mutations the writer task executes, each with a `oneshot` ack.
#[derive(Debug)]
pub(crate) enum WriteOp {
    EndSession {
        session_id: SessionId,
        ack: oneshot::Sender<Result<()>>,
    },
    BeginTurn {
        session_id: SessionId,
        ack: oneshot::Sender<Result<TurnId>>,
    },
    /// Upsert a memory by key; acks the row id.
    SaveMemory {
        key: String,
        value: String,
        ack: oneshot::Sender<Result<MemoryId>>,
    },
    DeleteMemory {
        key: String,
        ack: oneshot::Sender<Result<()>>,
    },
    /// Delete a memory (and, by cascade, its embedding) by row id; acks the deleted key.
    DeleteMemoryById {
        id: MemoryId,
        ack: oneshot::Sender<Result<Option<String>>>,
    },
    /// Upsert the chunk at byte range `bytes` of a message; acks the chunk id.
    StoreChunk {
        message_id: MessageId,
        bytes: Range<usize>,
        ack: oneshot::Sender<Result<ChunkId>>,
    },
    /// Upsert a batch of embeddings under `model` in one transaction.
    StoreEmbeddings {
        model: String,
        embeddings: Vec<Embedding>,
        ack: oneshot::Sender<Result<()>>,
    },
    /// Begin a session and create its `main` branch in one transaction.
    BeginSessionWithMainBranch {
        session_id: SessionId,
        daemon_pid: u32,
        ack: oneshot::Sender<Result<BranchId>>,
    },
    SetCurrentBranch {
        session_id: SessionId,
        branch_id: BranchId,
        ack: oneshot::Sender<Result<()>>,
    },
    /// Append `msg` and reference it from `branch_messages` in one transaction.
    AppendMessageToBranch {
        session_id: SessionId,
        branch_id: BranchId,
        turn_id: Option<TurnId>,
        msg: PersistedMessage,
        ack: oneshot::Sender<Result<MessageId>>,
    },
    /// Create a branch referencing every message on `src`; acks the new id.
    ForkBranch {
        src_branch_id: BranchId,
        new_name: String,
        ack: oneshot::Sender<Result<BranchId>>,
    },
    /// Upsert the branch's summary of everything before its `kept_turns`
    /// most recent user messages.
    RecordBranchSummary {
        branch_id: BranchId,
        summary: String,
        kept_turns: NonZeroUsize,
        ack: oneshot::Sender<Result<()>>,
    },
    UndoLastTurn {
        branch_id: BranchId,
        ack: oneshot::Sender<Result<UndoOutcome>>,
    },
    SetSessionTitle {
        session_id: SessionId,
        title: String,
        ack: oneshot::Sender<Result<()>>,
    },
}

/// Spawn the writer task. It exits once every sender is dropped, or once `shutdown`
/// flips to `true` and the queue then stays idle for two seconds.
pub(super) fn spawn_writer(
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
                        Some(op) => execute(&conn, op).await,
                        None => {
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
                        drain_queue(&conn, &mut rx).await;
                        break;
                    }
                }
            }
        }
        optimize_before_close(&conn).await;
    })
}

/// Refresh planner statistics the session's writes may have skewed, as SQLite
/// advises before closing a long-lived connection.
async fn optimize_before_close(conn: &Connection) {
    if let Err(e) = conn.call(|c| c.execute_batch("PRAGMA optimize")).await {
        tracing::warn!(
            target: "assistd::memory",
            error = %e,
            "PRAGMA optimize failed at writer exit"
        );
    }
}

/// Execute queued ops until the channel closes or stays idle for [`DRAIN_IDLE_TIMEOUT`].
/// Uses a timed `recv` rather than `try_recv` so a sender about to enqueue still lands.
async fn drain_queue(conn: &Connection, rx: &mut mpsc::Receiver<WriteOp>) {
    tracing::debug!(
        target: "assistd::memory",
        "shutdown received; draining writer queue"
    );
    loop {
        match tokio::time::timeout(DRAIN_IDLE_TIMEOUT, rx.recv()).await {
            Ok(Some(op)) => execute(conn, op).await,
            Ok(None) => break,
            Err(_) => {
                tracing::debug!(
                    target: "assistd::memory",
                    "drain timed out with channel idle; exiting"
                );
                break;
            }
        }
    }
}

async fn execute(conn: &Connection, op: WriteOp) {
    match op {
        WriteOp::EndSession { session_id, ack } => reply(ack, end_session(conn, session_id).await),
        WriteOp::BeginTurn { session_id, ack } => reply(ack, begin_turn(conn, session_id).await),
        WriteOp::SaveMemory { key, value, ack } => reply(ack, save_memory(conn, key, value).await),
        WriteOp::DeleteMemory { key, ack } => reply(ack, delete_memory(conn, key).await),
        WriteOp::DeleteMemoryById { id, ack } => reply(ack, delete_memory_by_id(conn, id).await),
        WriteOp::StoreChunk {
            message_id,
            bytes,
            ack,
        } => reply(ack, store_chunk(conn, message_id, bytes).await),
        WriteOp::StoreEmbeddings {
            model,
            embeddings,
            ack,
        } => reply(ack, store_embeddings(conn, model, embeddings).await),
        WriteOp::BeginSessionWithMainBranch {
            session_id,
            daemon_pid,
            ack,
        } => reply(
            ack,
            begin_session_with_main_branch(conn, session_id, daemon_pid).await,
        ),
        WriteOp::SetCurrentBranch {
            session_id,
            branch_id,
            ack,
        } => reply(ack, set_current_branch(conn, session_id, branch_id).await),
        WriteOp::AppendMessageToBranch {
            session_id,
            branch_id,
            turn_id,
            msg,
            ack,
        } => reply(
            ack,
            append_message_to_branch(conn, session_id, branch_id, turn_id, msg).await,
        ),
        WriteOp::ForkBranch {
            src_branch_id,
            new_name,
            ack,
        } => reply(ack, fork_branch(conn, src_branch_id, new_name).await),
        WriteOp::RecordBranchSummary {
            branch_id,
            summary,
            kept_turns,
            ack,
        } => reply(
            ack,
            record_branch_summary(conn, branch_id, summary, kept_turns).await,
        ),
        WriteOp::UndoLastTurn { branch_id, ack } => {
            reply(ack, undo_last_turn(conn, branch_id).await);
        }
        WriteOp::SetSessionTitle {
            session_id,
            title,
            ack,
        } => reply(ack, set_session_title(conn, session_id, title).await),
    }
}

fn reply<T>(ack: oneshot::Sender<Result<T>>, result: Result<T>) {
    let _ = ack.send(result);
}

async fn set_session_title(conn: &Connection, session_id: SessionId, title: String) -> Result<()> {
    conn.call(move |c| -> rusqlite::Result<_> {
        c.execute(
            "UPDATE sessions SET title = ?1 WHERE uuid = ?2",
            rusqlite::params![title, session_id],
        )?;
        Ok(())
    })
    .await
    .map_err(MemoryError::sqlite("set_session_title"))
}

async fn end_session(conn: &Connection, id: SessionId) -> Result<()> {
    let ended = now_millis();
    conn.call(move |c| -> rusqlite::Result<_> {
        c.execute(
            "UPDATE sessions SET ended_at = ?1 WHERE uuid = ?2",
            rusqlite::params![ended, id],
        )?;
        Ok(())
    })
    .await
    .map_err(MemoryError::sqlite("end_session"))
}

async fn begin_turn(conn: &Connection, session: SessionId) -> Result<TurnId> {
    conn.call(move |c| -> rusqlite::Result<_> {
        c.query_row(
            "INSERT INTO turns (session_id)
             SELECT id FROM sessions WHERE uuid = ?1
             RETURNING id",
            rusqlite::params![session],
            |r| r.get(0),
        )
    })
    .await
    .map_err(MemoryError::sqlite("begin_turn"))
}

/// Upserts by key; a changed value bumps `updated_at` and drops the row's
/// embedding so the backfill re-indexes it instead of serving the old value's vector.
async fn save_memory(conn: &Connection, key: String, value: String) -> Result<MemoryId> {
    let now = now_millis();
    conn.call(move |c| -> rusqlite::Result<_> {
        let tx = c.transaction()?;
        let previous_value: Option<String> = tx
            .query_row(
                "SELECT value FROM memories WHERE key = ?1",
                rusqlite::params![key],
                |r| r.get(0),
            )
            .optional()?;
        let id: MemoryId = tx.query_row(
            "INSERT INTO memories (key, value, created_at, updated_at) VALUES (?1, ?2, ?3, ?3)
                 ON CONFLICT(key) DO UPDATE SET
                     value = excluded.value,
                     updated_at = CASE WHEN value = excluded.value
                                       THEN updated_at
                                       ELSE MAX(excluded.updated_at, created_at) END
                 RETURNING id",
            rusqlite::params![key, value, now],
            |r| r.get(0),
        )?;
        if previous_value.is_some_and(|previous| previous != value) {
            tx.execute(
                "DELETE FROM memory_embeddings WHERE memory_id = ?1",
                rusqlite::params![id],
            )?;
        }
        tx.commit()?;
        Ok(id)
    })
    .await
    .map_err(MemoryError::sqlite("save_memory"))
}

async fn delete_memory(conn: &Connection, key: String) -> Result<()> {
    conn.call(move |c| -> rusqlite::Result<_> {
        c.execute(
            "DELETE FROM memories WHERE key = ?1",
            rusqlite::params![key],
        )?;
        truncate_wal(c)
    })
    .await
    .map_err(MemoryError::sqlite("delete_memory"))
}

async fn delete_memory_by_id(conn: &Connection, id: MemoryId) -> Result<Option<String>> {
    conn.call(move |c| -> rusqlite::Result<_> {
        let key = c
            .query_row(
                "DELETE FROM memories WHERE id = ?1 RETURNING key",
                rusqlite::params![id],
                |r| r.get::<_, String>(0),
            )
            .optional()?;
        truncate_wal(c)?;
        Ok(key)
    })
    .await
    .map_err(MemoryError::sqlite("delete_memory_by_id"))
}

async fn store_chunk(
    conn: &Connection,
    message_id: MessageId,
    bytes: Range<usize>,
) -> Result<ChunkId> {
    let byte_start = i64::try_from(bytes.start).unwrap_or(i64::MAX);
    let byte_end = i64::try_from(bytes.end).unwrap_or(i64::MAX);
    conn.call(move |c| -> rusqlite::Result<_> {
        c.query_row(
            "INSERT INTO message_chunks (message_id, byte_start, byte_end)
             VALUES (?1, ?2, ?3)
             ON CONFLICT(message_id, byte_start) DO UPDATE SET byte_end = excluded.byte_end
             RETURNING id",
            rusqlite::params![message_id, byte_start, byte_end],
            |r| r.get(0),
        )
    })
    .await
    .map_err(MemoryError::sqlite("store_chunk"))
}

async fn store_embeddings(
    conn: &Connection,
    model: String,
    embeddings: Vec<Embedding>,
) -> Result<()> {
    conn.call(move |c| -> rusqlite::Result<_> {
        let tx = c.transaction()?;
        let model_id = embedding_model_id(&tx, &model)?;
        for Embedding { target, vector } in &embeddings {
            let (upsert, target_id) = upsert_embedding_sql(*target);
            tx.prepare_cached(upsert)?.execute(rusqlite::params![
                target_id,
                model_id,
                vector_to_blob(vector)
            ])?;
        }
        tx.commit()
    })
    .await
    .map_err(MemoryError::sqlite("store_embeddings"))
}

/// The upsert for `target`'s embedding table, and the row id it binds as `?1`;
/// the `SELECT` makes a since-deleted target a no-op rather than a foreign-key error.
fn upsert_embedding_sql(target: EmbeddingTarget) -> (&'static str, i64) {
    match target {
        EmbeddingTarget::Chunk(ChunkId(id)) => (
            "INSERT INTO chunk_embeddings (chunk_id, model_id, vector)
             SELECT id, ?2, ?3 FROM message_chunks WHERE id = ?1
             ON CONFLICT(chunk_id) DO UPDATE SET model_id = excluded.model_id, vector = excluded.vector",
            id,
        ),
        EmbeddingTarget::Memory(MemoryId(id)) => (
            "INSERT INTO memory_embeddings (memory_id, model_id, vector)
             SELECT id, ?2, ?3 FROM memories WHERE id = ?1
             ON CONFLICT(memory_id) DO UPDATE SET model_id = excluded.model_id, vector = excluded.vector",
            id,
        ),
    }
}

/// The `embedding_models` id of `name`, inserting the model on first use.
fn embedding_model_id(tx: &Transaction<'_>, name: &str) -> rusqlite::Result<i64> {
    tx.execute(
        "INSERT INTO embedding_models (name) VALUES (?1) ON CONFLICT(name) DO NOTHING",
        rusqlite::params![name],
    )?;
    tx.query_row(
        "SELECT id FROM embedding_models WHERE name = ?1",
        rusqlite::params![name],
        |r| r.get(0),
    )
}

async fn begin_session_with_main_branch(
    conn: &Connection,
    id: SessionId,
    pid: u32,
) -> Result<BranchId> {
    let started = now_millis();
    conn.call(move |c| -> rusqlite::Result<_> {
        let tx = c.transaction()?;
        let session_rowid: i64 = tx.query_row(
            "INSERT INTO sessions (uuid, started_at, daemon_pid) VALUES (?1, ?2, ?3)
                 RETURNING id",
            rusqlite::params![id, started, pid],
            |r| r.get(0),
        )?;
        tx.execute(
            "INSERT INTO branches (session_id, name, parent_branch_id, fork_point_seq, created_at)
                 VALUES (?1, 'main', NULL, NULL, ?2)",
            rusqlite::params![session_rowid, started],
        )?;
        let branch_id = BranchId(tx.last_insert_rowid());
        tx.execute(
            "UPDATE sessions SET current_branch_id = ?1 WHERE id = ?2",
            rusqlite::params![branch_id, session_rowid],
        )?;
        tx.commit()?;
        Ok(branch_id)
    })
    .await
    .map_err(MemoryError::sqlite("begin_session_with_main_branch"))
}

async fn set_current_branch(
    conn: &Connection,
    session_id: SessionId,
    branch_id: BranchId,
) -> Result<()> {
    conn.call(move |c| -> rusqlite::Result<_> {
        c.execute(
            "UPDATE sessions SET current_branch_id = ?1 WHERE uuid = ?2",
            rusqlite::params![branch_id, session_id],
        )?;
        Ok(())
    })
    .await
    .map_err(MemoryError::sqlite("set_current_branch"))
}

async fn append_message_to_branch(
    conn: &Connection,
    session: SessionId,
    branch_id: BranchId,
    turn_id: Option<TurnId>,
    msg: PersistedMessage,
) -> Result<MessageId> {
    let created_at = now_millis();
    let role = msg.role();
    let columns = MessageColumns::from_kind(msg.kind)?;
    conn.call(move |c| -> rusqlite::Result<_> {
        let tx = c.transaction()?;
        let message_id: MessageId = tx.query_row(
            "INSERT INTO messages
                (session_id, turn_id, created_at, role, content, tool_calls, tool_call_id,
                 tool_name, reasoning, tool_exit_code, tool_duration_ms)
             SELECT id, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11
             FROM sessions WHERE uuid = ?1
             RETURNING id",
            rusqlite::params![
                session,
                turn_id,
                created_at,
                role,
                msg.content,
                columns.tool_calls,
                columns.tool_call_id,
                columns.tool_name,
                columns.reasoning,
                columns.tool_exit_code,
                columns.tool_duration_ms,
            ],
            |r| r.get(0),
        )?;
        tx.execute(
            "INSERT INTO branch_messages (branch_id, seq, message_id)
             SELECT ?1, COALESCE(MAX(seq), -1) + 1, ?2 FROM branch_messages WHERE branch_id = ?1",
            rusqlite::params![branch_id, message_id],
        )?;
        tx.commit()?;
        Ok(message_id)
    })
    .await
    .map_err(MemoryError::sqlite("append_message_to_branch"))
}

/// The role-specific `messages` columns of a [`MessageKind`], `NULL` where the role has none.
#[derive(Default)]
struct MessageColumns {
    reasoning: Option<String>,
    tool_calls: Option<String>,
    tool_call_id: Option<String>,
    tool_name: Option<String>,
    tool_exit_code: Option<i64>,
    tool_duration_ms: Option<i64>,
}

impl MessageColumns {
    fn from_kind(kind: MessageKind) -> Result<Self> {
        Ok(match kind {
            MessageKind::User => Self::default(),
            MessageKind::Assistant {
                reasoning,
                tool_calls,
            } => Self {
                reasoning,
                tool_calls: tool_calls.as_ref().map(serde_json::to_string).transpose()?,
                ..Self::default()
            },
            MessageKind::Tool {
                call_id,
                tool_name,
                outcome,
            } => Self {
                tool_call_id: Some(call_id),
                tool_name: Some(tool_name),
                tool_exit_code: Some(outcome.exit_code),
                tool_duration_ms: Some(i64::try_from(outcome.duration_ms).unwrap_or(i64::MAX)),
                ..Self::default()
            },
        })
    }
}

async fn fork_branch(conn: &Connection, src: BranchId, new_name: String) -> Result<BranchId> {
    let created = now_millis();
    conn.call(move |c| -> rusqlite::Result<_> {
        let tx = c.transaction()?;
        let session_id: i64 = tx.query_row(
            "SELECT session_id FROM branches WHERE id = ?1",
            rusqlite::params![src],
            |r| r.get(0),
        )?;
        let fork_point_seq: Option<i64> = tx.query_row(
            "SELECT MAX(seq) FROM branch_messages WHERE branch_id = ?1",
            rusqlite::params![src],
            |r| r.get(0),
        )?;
        tx.execute(
            "INSERT INTO branches (session_id, name, parent_branch_id, fork_point_seq, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![session_id, new_name, src, fork_point_seq, created],
        )?;
        let new_branch_id = BranchId(tx.last_insert_rowid());
        tx.execute(
            "INSERT INTO branch_messages (branch_id, seq, message_id)
                 SELECT ?1, seq, message_id
                 FROM branch_messages WHERE branch_id = ?2",
            rusqlite::params![new_branch_id, src],
        )?;
        tx.execute(
            "INSERT INTO branch_summaries (branch_id, through_seq, content)
                 SELECT ?1, through_seq, content
                 FROM branch_summaries WHERE branch_id = ?2",
            rusqlite::params![new_branch_id, src],
        )?;
        tx.commit()?;
        Ok(new_branch_id)
    })
    .await
    .map_err(MemoryError::sqlite("fork_branch"))
}

async fn undo_last_turn(conn: &Connection, branch: BranchId) -> Result<UndoOutcome> {
    conn.call(move |c| -> rusqlite::Result<_> {
        let tx = c.transaction()?;
        let Some(turn_id) = latest_turn_on_branch(&tx, branch)? else {
            tx.commit()?;
            return Ok(UndoOutcome::default());
        };
        let last_user_text: Option<String> = tx
            .query_row(
                "SELECT content FROM messages WHERE turn_id = ?1 AND role = ?2
                 ORDER BY id LIMIT 1",
                rusqlite::params![turn_id, PersistedRole::User],
                |r| r.get(0),
            )
            .optional()?;
        drop_summary_covering_turn(&tx, branch, turn_id)?;
        let removed: usize = tx.execute(
            "DELETE FROM branch_messages
             WHERE branch_id = ?1
               AND message_id IN (
                   SELECT id FROM messages WHERE turn_id = ?2
               )",
            rusqlite::params![branch, turn_id],
        )?;
        delete_unreferenced_turn(&tx, turn_id)?;
        tx.commit()?;
        truncate_wal(c)?;
        Ok(UndoOutcome {
            removed_messages: u32::try_from(removed).unwrap_or(u32::MAX),
            last_user_text,
        })
    })
    .await
    .map_err(MemoryError::sqlite("undo_last_turn"))
}

async fn record_branch_summary(
    conn: &Connection,
    branch: BranchId,
    summary: String,
    kept_turns: NonZeroUsize,
) -> Result<()> {
    let offset = i64::try_from(kept_turns.get() - 1).unwrap_or(i64::MAX);
    conn.call(move |c| -> rusqlite::Result<_> {
        c.execute(
            "INSERT OR REPLACE INTO branch_summaries (branch_id, through_seq, content)
             SELECT ?1, bm.seq - 1, ?2
             FROM branch_messages bm JOIN messages m ON m.id = bm.message_id
             WHERE bm.branch_id = ?1 AND m.role = ?3
             ORDER BY bm.seq DESC LIMIT 1 OFFSET ?4",
            rusqlite::params![branch, summary, PersistedRole::User, offset],
        )?;
        Ok(())
    })
    .await
    .map_err(MemoryError::sqlite("record_branch_summary"))
}

/// Drop the branch's summary once it stands in for any message of `turn_id`.
fn drop_summary_covering_turn(
    tx: &Transaction<'_>,
    branch: BranchId,
    turn_id: TurnId,
) -> rusqlite::Result<()> {
    tx.execute(
        "DELETE FROM branch_summaries
         WHERE branch_id = ?1
           AND through_seq >= (
               SELECT MIN(bm.seq)
               FROM branch_messages bm JOIN messages m ON m.id = bm.message_id
               WHERE bm.branch_id = ?1 AND m.turn_id = ?2
           )",
        rusqlite::params![branch, turn_id],
    )?;
    Ok(())
}

fn latest_turn_on_branch(
    tx: &Transaction<'_>,
    branch: BranchId,
) -> rusqlite::Result<Option<TurnId>> {
    tx.query_row(
        "SELECT MAX(m.turn_id)
         FROM branch_messages bm JOIN messages m ON m.id = bm.message_id
         WHERE bm.branch_id = ?1 AND m.turn_id IS NOT NULL",
        rusqlite::params![branch],
        |r| r.get(0),
    )
}

/// Delete the turn's messages that no branch references any more, then the
/// turn itself once no message points at it.
fn delete_unreferenced_turn(tx: &Transaction<'_>, turn_id: TurnId) -> rusqlite::Result<()> {
    tx.execute(
        "DELETE FROM messages
         WHERE turn_id = ?1
           AND NOT EXISTS (SELECT 1 FROM branch_messages bm WHERE bm.message_id = messages.id)",
        rusqlite::params![turn_id],
    )?;
    tx.execute(
        "DELETE FROM turns
         WHERE id = ?1
           AND NOT EXISTS (SELECT 1 FROM messages WHERE turn_id = ?1)",
        rusqlite::params![turn_id],
    )?;
    Ok(())
}

/// Checkpoint and empty the WAL so frames holding deleted rows don't outlive the delete.
fn truncate_wal(c: &rusqlite::Connection) -> rusqlite::Result<()> {
    let busy: bool = c.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |r| r.get(0))?;
    if busy {
        tracing::warn!(
            target: "assistd::memory",
            "WAL checkpoint blocked by another reader; deleted rows stay in the WAL until the next checkpoint"
        );
    }
    Ok(())
}

pub(super) async fn dispatch_write<T, F>(tx: &mpsc::Sender<WriteOp>, build: F) -> Result<T>
where
    F: FnOnce(oneshot::Sender<Result<T>>) -> WriteOp,
{
    let (ack_tx, ack_rx) = oneshot::channel();
    tx.send(build(ack_tx))
        .await
        .map_err(|_| MemoryError::WriterGone)?;
    ack_rx.await.map_err(|_| MemoryError::AckDropped)?
}
