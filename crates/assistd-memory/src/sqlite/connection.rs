//! Open the SQLite database, apply pragmas, run migrations, and stand
//! up the writer task. Hands callers a [`SqliteHandle`] both stores
//! clone freely so they share one connection + one writer.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tokio_rusqlite::Connection;

use crate::migrations;

use super::writer::{WriteOp, spawn_writer};

/// Bounded channel size for the writer queue. Big enough to absorb a
/// bursty turn (a single agent step can fan out ~10 ToolCall/ToolResult
/// pairs) without backpressuring the dispatch loop. If the queue ever
/// fills, callers `await` on `tx.send()`, which briefly stalls the
/// caller (a `tokio::spawn` logger task in the chat-turn case, so the
/// LLM stream itself never stalls).
const WRITER_QUEUE_DEPTH: usize = 256;

/// Cheaply-cloneable handle held by [`super::SqliteMemoryStore`] and
/// [`super::SqliteConversationStore`]. The connection serves reads
/// directly via `conn.call(...)`; writes go through `writer_tx`.
#[derive(Clone)]
pub struct SqliteHandle {
    pub(super) conn: Connection,
    pub(super) writer_tx: Arc<mpsc::Sender<WriteOp>>,
}

impl SqliteHandle {
    /// Open `path`, apply pragmas, run migrations, and spawn the writer.
    /// Returns the handle plus the writer's `JoinHandle`; the daemon
    /// keeps it so it can `.await` the writer on shutdown alongside the
    /// other background tasks at `crates/assistd/src/daemon.rs:321-333`.
    ///
    /// `path` is created with all parent directories if they don't
    /// exist (mirrors the AC #1 expectation of an auto-created DB at
    /// `~/.local/share/assistd/memory.db`).
    pub async fn open(
        path: &Path,
        shutdown: watch::Receiver<bool>,
    ) -> Result<(Self, JoinHandle<()>)> {
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .with_context(|| format!("create memory.db parent dir {}", parent.display()))?;
        }

        let path_owned = path.to_path_buf();
        let conn = Connection::open(path_owned.clone())
            .await
            .with_context(|| format!("open SQLite at {}", path_owned.display()))?;

        // Pragmas first, then migrations. WAL gives us concurrent reads
        // alongside the single writer; foreign_keys must be ON before
        // any FK is exercised by migrations.
        conn.call(|c| -> rusqlite::Result<_> {
            c.pragma_update(None, "journal_mode", "WAL")?;
            c.pragma_update(None, "synchronous", "NORMAL")?;
            c.pragma_update(None, "foreign_keys", "ON")?;
            Ok(())
        })
        .await
        .context("apply SQLite pragmas")?;

        conn.call(|c| migrations::run(c))
            .await
            .context("run migrations")?;

        let (writer_tx, writer_rx) = mpsc::channel(WRITER_QUEUE_DEPTH);
        let writer_handle = spawn_writer(conn.clone(), writer_rx, shutdown);

        Ok((
            Self {
                conn,
                writer_tx: Arc::new(writer_tx),
            },
            writer_handle,
        ))
    }

    /// Direct access to the read-side `Connection` for query paths
    /// (search, list, recent_turns). Reads bypass the writer channel so
    /// they don't queue behind in-flight inserts.
    pub(super) fn conn(&self) -> &Connection {
        &self.conn
    }

    /// Sender shared by both stores so a single writer serves all writes.
    pub(super) fn writer(&self) -> &mpsc::Sender<WriteOp> {
        &self.writer_tx
    }

    /// Cheap clone of the writer-task `Arc<Sender>`. Exposed for
    /// `assistd-core` (chunking on the persistence path) and
    /// `assistd-embed` (the embedder task ack'ing back into the same
    /// writer queue); both live outside this crate so the
    /// `pub(super) fn writer(&self)` accessor isn't visible to them.
    pub fn writer_tx(&self) -> Arc<mpsc::Sender<WriteOp>> {
        self.writer_tx.clone()
    }

    /// Convenience: persist a chunk and return its rowid. Used by
    /// `state.rs::persist_message_fire_and_forget` after `append_message`
    /// so the chunk row can be embedded in the background.
    pub async fn store_chunk(
        &self,
        conversation_id: i64,
        chunk_index: i64,
        content: String,
        token_count: Option<i64>,
    ) -> anyhow::Result<i64> {
        use super::writer::WriteCall;
        WriteCall::run(self.writer(), |ack| WriteOp::StoreChunk {
            conversation_id,
            chunk_index,
            content,
            token_count,
            ack,
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shutdown_pair() -> (watch::Sender<bool>, watch::Receiver<bool>) {
        watch::channel(false)
    }

    #[tokio::test]
    async fn open_creates_parent_dirs_and_runs_migrations() {
        // Point at a path inside a non-existent directory; open() must
        // mkdir -p it. This is exactly what AC #1 ("created automatically
        // on first run") demands.
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("nested/dir/memory.db");

        let (_tx, rx) = shutdown_pair();
        let (handle, writer) = SqliteHandle::open(&path, rx).await.unwrap();

        // Migrations applied: schema_migrations has a row.
        let n: i64 = handle
            .conn()
            .call(|c| -> rusqlite::Result<_> {
                c.query_row(
                    "SELECT count(*) FROM sqlite_master WHERE name='conversations'",
                    [],
                    |r| r.get(0),
                )
            })
            .await
            .unwrap();
        assert_eq!(n, 1);

        drop(handle);
        // No graceful shutdown signal here; we just dropped the handle,
        // so the writer's mpsc closes naturally and the worker exits.
        writer.await.unwrap();
    }

    #[tokio::test]
    async fn shutdown_signal_drains_writer() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("memory.db");
        let (tx, rx) = shutdown_pair();
        let (handle, writer) = SqliteHandle::open(&path, rx).await.unwrap();

        // Send shutdown; the writer must observe it and exit promptly.
        tx.send(true).unwrap();
        // Drop our handle so the channel closes too; writer should
        // return either way.
        drop(handle);
        let res = tokio::time::timeout(std::time::Duration::from_secs(2), writer).await;
        res.expect("writer exited within 2s").unwrap();
    }
}
