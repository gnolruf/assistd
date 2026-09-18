//! Single-writer task for the SQLite store. Every mutation is a
//! [`WriteOp`] sent to one task that owns the connection; each op
//! carries a `oneshot` ack the caller may await or ignore. On shutdown
//! the task drains its queue so a write issued just before SIGTERM
//! still lands.

use std::time::Duration;

use anyhow::{Context, Result};
use chrono::Utc;
use rusqlite::OptionalExtension;
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::JoinHandle;
use tokio_rusqlite::Connection;

use super::conversations::{BranchId, PersistedMessage, TurnId, UndoOutcome};

/// Mutations the writer task executes, each with a `oneshot` ack.
pub enum WriteOp {
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
    /// Upsert a memory by key; acks the row id.
    SaveMemory {
        key: String,
        value: String,
        source_conversation_id: Option<i64>,
        ack: oneshot::Sender<Result<i64>>,
    },
    DeleteMemory {
        key: String,
        ack: oneshot::Sender<Result<()>>,
    },
    /// Delete a memory by row id; acks the deleted key, or `None` on
    /// miss. The embedding row cascades.
    DeleteMemoryById {
        id: i64,
        ack: oneshot::Sender<Result<Option<String>>>,
    },
    /// Upsert one chunk of a conversation message; acks the chunk id.
    StoreChunk {
        conversation_id: i64,
        chunk_index: i64,
        content: String,
        token_count: Option<i64>,
        ack: oneshot::Sender<Result<i64>>,
    },
    /// Upsert the embedding for a `conversation_chunks` row.
    StoreChunkEmbedding {
        chunk_id: i64,
        model: String,
        dim: i64,
        vector: Vec<u8>,
        ack: oneshot::Sender<Result<()>>,
    },
    /// Upsert the embedding for a `memories` row.
    StoreMemoryEmbedding {
        memory_id: i64,
        model: String,
        dim: i64,
        vector: Vec<u8>,
        ack: oneshot::Sender<Result<()>>,
    },
    /// Begin a session and create its `main` branch in one transaction.
    BeginSessionWithMainBranch {
        session_id: String,
        daemon_pid: u32,
        ack: oneshot::Sender<Result<BranchId>>,
    },
    /// Insert one row into `branches`; acks the new id.
    CreateBranch {
        session_id: String,
        name: String,
        parent_branch_id: Option<BranchId>,
        fork_point_seq: Option<i64>,
        ack: oneshot::Sender<Result<BranchId>>,
    },
    /// Update `sessions.current_branch_id`.
    SetCurrentBranch {
        session_id: String,
        branch_id: BranchId,
        ack: oneshot::Sender<Result<()>>,
    },
    /// Append `msg` and reference it from `branch_messages`, in one
    /// transaction.
    AppendMessageToBranch {
        session_id: String,
        branch_id: BranchId,
        turn_id: Option<TurnId>,
        msg: PersistedMessage,
        ack: oneshot::Sender<Result<i64>>,
    },
    /// Create a branch that references every message on `src`,
    /// preserving seq; acks the new id.
    ForkBranch {
        src_branch_id: BranchId,
        new_name: String,
        ack: oneshot::Sender<Result<BranchId>>,
    },
    /// Drop the most recent turn from `branch`.
    UndoLastTurn {
        branch_id: BranchId,
        ack: oneshot::Sender<Result<UndoOutcome>>,
    },
    /// Set `sessions.title`.
    SetSessionTitle {
        session_id: String,
        title: String,
        ack: oneshot::Sender<Result<()>>,
    },
}

/// Spawn the writer task. The caller awaits the returned handle on
/// shutdown, after flipping `shutdown`.
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
                        // `recv` with a timeout rather than `try_recv`, so a
                        // sender that is about to enqueue still wins.
                        tracing::debug!(
                            target: "assistd::memory",
                            "shutdown received; draining writer queue"
                        );
                        let drain_deadline = Duration::from_secs(2);
                        loop {
                            match tokio::time::timeout(drain_deadline, rx.recv()).await {
                                Ok(Some(op)) => handle_op(&conn, op).await,
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
                        break;
                    }
                }
            }
        }
    })
}

async fn handle_op(conn: &Connection, op: WriteOp) {
    match op {
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
        WriteOp::BeginSessionWithMainBranch {
            session_id,
            daemon_pid,
            ack,
        } => {
            let res = begin_session_with_main_branch(conn, session_id, daemon_pid).await;
            let _ = ack.send(res);
        }
        WriteOp::CreateBranch {
            session_id,
            name,
            parent_branch_id,
            fork_point_seq,
            ack,
        } => {
            let res = create_branch(conn, session_id, name, parent_branch_id, fork_point_seq).await;
            let _ = ack.send(res);
        }
        WriteOp::SetCurrentBranch {
            session_id,
            branch_id,
            ack,
        } => {
            let res = set_current_branch(conn, session_id, branch_id).await;
            let _ = ack.send(res);
        }
        WriteOp::AppendMessageToBranch {
            session_id,
            branch_id,
            turn_id,
            msg,
            ack,
        } => {
            let res = append_message_to_branch(conn, session_id, branch_id, turn_id, msg).await;
            let _ = ack.send(res);
        }
        WriteOp::ForkBranch {
            src_branch_id,
            new_name,
            ack,
        } => {
            let res = fork_branch(conn, src_branch_id, new_name).await;
            let _ = ack.send(res);
        }
        WriteOp::UndoLastTurn { branch_id, ack } => {
            let res = undo_last_turn(conn, branch_id).await;
            let _ = ack.send(res);
        }
        WriteOp::SetSessionTitle {
            session_id,
            title,
            ack,
        } => {
            let res = set_session_title(conn, session_id, title).await;
            let _ = ack.send(res);
        }
    }
}

async fn set_session_title(conn: &Connection, session_id: String, title: String) -> Result<()> {
    conn.call(move |c| -> rusqlite::Result<_> {
        c.execute(
            "UPDATE sessions SET title = ?1 WHERE id = ?2",
            rusqlite::params![title, session_id],
        )?;
        Ok(())
    })
    .await
    .context("set_session_title")
}

async fn end_session(conn: &Connection, id: String) -> Result<()> {
    let ended = Utc::now().to_rfc3339();
    conn.call(move |c| -> rusqlite::Result<_> {
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
        .call(move |c| -> rusqlite::Result<_> {
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
    conn.call(move |c| -> rusqlite::Result<_> {
        c.execute(
            "UPDATE turns SET ended_at = ?1 WHERE id = ?2",
            rusqlite::params![ended, turn.0],
        )?;
        Ok(())
    })
    .await
    .context("end_turn")
}

async fn save_memory(
    conn: &Connection,
    key: String,
    value: String,
    source: Option<i64>,
) -> Result<i64> {
    let now = Utc::now().to_rfc3339();
    let id = conn
        .call(move |c| -> rusqlite::Result<_> {
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
    conn.call(move |c| -> rusqlite::Result<_> {
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
    conn.call(move |c| -> rusqlite::Result<_> {
        c.query_row(
            "DELETE FROM memories WHERE id = ?1 RETURNING key",
            rusqlite::params![id],
            |r| r.get::<_, String>(0),
        )
        .optional()
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
        .call(move |c| -> rusqlite::Result<_> {
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
    conn.call(move |c| -> rusqlite::Result<_> {
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
    conn.call(move |c| -> rusqlite::Result<_> {
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

async fn begin_session_with_main_branch(
    conn: &Connection,
    id: String,
    pid: u32,
) -> Result<BranchId> {
    let started = Utc::now().to_rfc3339();
    let created = started.clone();
    let branch_rowid = conn
        .call(move |c| -> rusqlite::Result<_> {
            // One transaction: startup treats a session without a main
            // branch as corrupt.
            let tx = c.transaction()?;
            tx.execute(
                "INSERT INTO sessions (id, started_at, daemon_pid) VALUES (?1, ?2, ?3)",
                rusqlite::params![id, started, pid],
            )?;
            tx.execute(
                "INSERT INTO branches (session_id, name, parent_branch_id, fork_point_seq, created_at)
                 VALUES (?1, 'main', NULL, NULL, ?2)",
                rusqlite::params![id, created],
            )?;
            let branch_id = tx.last_insert_rowid();
            tx.execute(
                "UPDATE sessions SET current_branch_id = ?1 WHERE id = ?2",
                rusqlite::params![branch_id, id],
            )?;
            tx.commit()?;
            Ok(branch_id)
        })
        .await
        .context("begin_session_with_main_branch")?;
    Ok(BranchId(branch_rowid))
}

async fn create_branch(
    conn: &Connection,
    session_id: String,
    name: String,
    parent: Option<BranchId>,
    fork_point_seq: Option<i64>,
) -> Result<BranchId> {
    let created = Utc::now().to_rfc3339();
    let parent_id = parent.map(|b| b.0);
    let id = conn
        .call(move |c| -> rusqlite::Result<_> {
            c.execute(
                "INSERT INTO branches (session_id, name, parent_branch_id, fork_point_seq, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![session_id, name, parent_id, fork_point_seq, created],
            )?;
            Ok(c.last_insert_rowid())
        })
        .await
        .context("create_branch")?;
    Ok(BranchId(id))
}

async fn set_current_branch(
    conn: &Connection,
    session_id: String,
    branch_id: BranchId,
) -> Result<()> {
    conn.call(move |c| -> rusqlite::Result<_> {
        c.execute(
            "UPDATE sessions SET current_branch_id = ?1 WHERE id = ?2",
            rusqlite::params![branch_id.0, session_id],
        )?;
        Ok(())
    })
    .await
    .context("set_current_branch")
}

async fn append_message_to_branch(
    conn: &Connection,
    session: String,
    branch_id: BranchId,
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
        .call(move |c| -> rusqlite::Result<_> {
            let tx = c.transaction()?;
            let seq: i64 = tx.query_row(
                "SELECT COALESCE(MAX(seq), -1) + 1 FROM conversations WHERE session_id = ?1",
                rusqlite::params![session],
                |r| r.get(0),
            )?;
            tx.execute(
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
            let conv_id = tx.last_insert_rowid();
            let branch_seq: i64 = tx.query_row(
                "SELECT COALESCE(MAX(seq), -1) + 1 FROM branch_messages WHERE branch_id = ?1",
                rusqlite::params![branch_id.0],
                |r| r.get(0),
            )?;
            tx.execute(
                "INSERT INTO branch_messages (branch_id, seq, conversation_id) VALUES (?1, ?2, ?3)",
                rusqlite::params![branch_id.0, branch_seq, conv_id],
            )?;
            tx.commit()?;
            Ok(conv_id)
        })
        .await
        .context("append_message_to_branch")?;
    Ok(id)
}

async fn fork_branch(conn: &Connection, src: BranchId, new_name: String) -> Result<BranchId> {
    let created = Utc::now().to_rfc3339();
    let id = conn
        .call(move |c| -> rusqlite::Result<_> {
            let tx = c.transaction()?;
            let session_id: String = tx.query_row(
                "SELECT session_id FROM branches WHERE id = ?1",
                rusqlite::params![src.0],
                |r| r.get(0),
            )?;
            let fork_point_seq: Option<i64> = tx.query_row(
                "SELECT MAX(seq) FROM branch_messages WHERE branch_id = ?1",
                rusqlite::params![src.0],
                |r| r.get(0),
            )?;
            tx.execute(
                "INSERT INTO branches (session_id, name, parent_branch_id, fork_point_seq, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![session_id, new_name, src.0, fork_point_seq, created],
            )?;
            let new_branch_id = tx.last_insert_rowid();
            tx.execute(
                "INSERT INTO branch_messages (branch_id, seq, conversation_id)
                 SELECT ?1, seq, conversation_id
                 FROM branch_messages WHERE branch_id = ?2",
                rusqlite::params![new_branch_id, src.0],
            )?;
            tx.commit()?;
            Ok(new_branch_id)
        })
        .await
        .context("fork_branch")?;
    Ok(BranchId(id))
}

async fn undo_last_turn(conn: &Connection, branch: BranchId) -> Result<UndoOutcome> {
    conn.call(move |c| -> rusqlite::Result<_> {
        let tx = c.transaction()?;
        let last_turn: Option<i64> = tx.query_row(
            "SELECT MAX(c.turn_id)
             FROM branch_messages bm JOIN conversations c ON c.id = bm.conversation_id
             WHERE bm.branch_id = ?1 AND c.turn_id IS NOT NULL",
            rusqlite::params![branch.0],
            |r| r.get(0),
        )?;
        let Some(turn_id) = last_turn else {
            tx.commit()?;
            return Ok(UndoOutcome::default());
        };
        let last_user_text: Option<String> = tx
            .query_row(
                "SELECT user_text FROM turns WHERE id = ?1",
                rusqlite::params![turn_id],
                |r| r.get(0),
            )
            .optional()?;

        // Captured before the delete: these are the orphan candidates.
        let target_conv_ids: Vec<i64> = {
            let mut stmt = tx.prepare(
                "SELECT bm.conversation_id
                 FROM branch_messages bm JOIN conversations c ON c.id = bm.conversation_id
                 WHERE bm.branch_id = ?1 AND c.turn_id = ?2",
            )?;
            let rows: Vec<i64> = stmt
                .query_map(rusqlite::params![branch.0, turn_id], |r| r.get(0))?
                .collect::<std::result::Result<_, _>>()?;
            rows
        };

        let removed: usize = tx.execute(
            "DELETE FROM branch_messages
             WHERE branch_id = ?1
               AND conversation_id IN (
                   SELECT id FROM conversations WHERE turn_id = ?2
               )",
            rusqlite::params![branch.0, turn_id],
        )?;

        for cid in &target_conv_ids {
            let still_referenced: i64 = tx.query_row(
                "SELECT COUNT(*) FROM branch_messages WHERE conversation_id = ?1",
                rusqlite::params![cid],
                |r| r.get(0),
            )?;
            if still_referenced == 0 {
                tx.execute(
                    "DELETE FROM conversations WHERE id = ?1",
                    rusqlite::params![cid],
                )?;
            }
        }

        // A forked sibling may still reference the turn; keep it then.
        let turn_still_used: i64 = tx.query_row(
            "SELECT COUNT(*) FROM conversations WHERE turn_id = ?1",
            rusqlite::params![turn_id],
            |r| r.get(0),
        )?;
        if turn_still_used == 0 {
            tx.execute(
                "DELETE FROM turns WHERE id = ?1",
                rusqlite::params![turn_id],
            )?;
        }

        tx.commit()?;
        Ok(UndoOutcome {
            removed_messages: removed as u32,
            last_user_text,
            removed_turn_id: Some(turn_id),
        })
    })
    .await
    .context("undo_last_turn")
}

pub(super) async fn dispatch_write<T, F>(tx: &mpsc::Sender<WriteOp>, build: F) -> Result<T>
where
    F: FnOnce(oneshot::Sender<Result<T>>) -> WriteOp,
{
    let (ack_tx, ack_rx) = oneshot::channel();
    tx.send(build(ack_tx))
        .await
        .map_err(|_| anyhow::anyhow!("memory writer task is gone"))?;
    ack_rx
        .await
        .map_err(|_| anyhow::anyhow!("memory writer task dropped ack channel"))?
}
