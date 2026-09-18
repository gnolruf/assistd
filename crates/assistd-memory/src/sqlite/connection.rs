//! Opens the database and hands out the shared [`SqliteHandle`].

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tokio_rusqlite::Connection;

use crate::migrations;

use super::writer::{WriteOp, spawn_writer};

/// Writer queue depth: enough to absorb one bursty agent step (roughly
/// ten tool call/result pairs) without backpressuring the sender.
const WRITER_QUEUE_DEPTH: usize = 256;

/// Cheaply cloneable handle shared by every store: reads use `conn`
/// directly, writes go through the writer task.
#[derive(Clone)]
pub struct SqliteHandle {
    pub(super) conn: Connection,
    pub(super) writer_tx: Arc<mpsc::Sender<WriteOp>>,
}

impl SqliteHandle {
    /// Open `path` (creating parent directories), apply pragmas, run
    /// migrations, and spawn the writer task. Returns the handle and
    /// the writer's `JoinHandle`, which the caller awaits on shutdown.
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

        conn.call(|c| -> rusqlite::Result<_> {
            c.pragma_update(None, "journal_mode", "WAL")?;
            c.pragma_update(None, "synchronous", "NORMAL")?;
            c.pragma_update(None, "foreign_keys", "ON")?;
            Ok(())
        })
        .await
        .context("apply SQLite pragmas")?;

        conn.call(migrations::run).await.context("run migrations")?;

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

    pub(super) fn conn(&self) -> &Connection {
        &self.conn
    }

    pub(super) fn writer(&self) -> &mpsc::Sender<WriteOp> {
        &self.writer_tx
    }

    /// Clone of the writer sender, for producers that enqueue
    /// [`WriteOp`]s directly.
    pub fn writer_tx(&self) -> Arc<mpsc::Sender<WriteOp>> {
        self.writer_tx.clone()
    }

    /// Persist one chunk of a conversation message and return its row id.
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
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("nested/dir/memory.db");

        let (_tx, rx) = shutdown_pair();
        let (handle, writer) = SqliteHandle::open(&path, rx).await.unwrap();

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
