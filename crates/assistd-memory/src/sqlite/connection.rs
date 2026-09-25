//! Opens the database and hands out the shared [`SqliteHandle`].

use std::path::Path;
use std::sync::Arc;

use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tokio_rusqlite::Connection;

use crate::{MemoryError, Result, migrations};

use super::writer::{WriteOp, dispatch_write, spawn_writer};

/// Deep enough to absorb one bursty agent step without backpressuring the sender.
const WRITER_QUEUE_DEPTH: usize = 256;

/// Cheaply cloneable handle shared by every store: reads hit the connection directly,
/// writes go through the writer task.
#[derive(Clone)]
pub struct SqliteHandle {
    pub(super) conn: Connection,
    pub(super) writer_tx: Arc<mpsc::Sender<WriteOp>>,
}

impl SqliteHandle {
    /// Open `path` (creating parent directories), apply pragmas, run migrations, and
    /// spawn the writer task, returning it alongside the handle.
    pub async fn open(
        path: &Path,
        shutdown: watch::Receiver<bool>,
    ) -> Result<(Self, JoinHandle<()>)> {
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|source| MemoryError::CreateDir {
                    path: parent.to_path_buf(),
                    source,
                })?;
        }

        let conn = Connection::open(path)
            .await
            .map_err(|source| MemoryError::Open {
                path: path.to_path_buf(),
                source,
            })?;

        conn.call(|c| -> rusqlite::Result<_> {
            c.pragma_update(None, "journal_mode", "WAL")?;
            c.pragma_update(None, "synchronous", "NORMAL")?;
            c.pragma_update(None, "foreign_keys", "ON")?;
            Ok(())
        })
        .await
        .map_err(MemoryError::sqlite("apply SQLite pragmas"))?;

        conn.call(migrations::run)
            .await
            .map_err(MemoryError::Migration)?;

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

    /// Clone of the writer sender, for enqueueing [`WriteOp`]s directly.
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
    ) -> Result<i64> {
        dispatch_write(self.writer(), |ack| WriteOp::StoreChunk {
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

    #[tokio::test]
    async fn open_creates_parent_dirs_and_runs_migrations() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("nested/dir/memory.db");

        let (_tx, rx) = watch::channel(false);
        let (handle, writer) = SqliteHandle::open(&path, rx).await.unwrap();

        let table_count: i64 = handle
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
        assert_eq!(table_count, 1);

        drop(handle);
        writer.await.unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn shutdown_signal_stops_writer_while_handle_is_alive() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("memory.db");
        let (tx, rx) = watch::channel(false);
        let (_handle, writer) = SqliteHandle::open(&path, rx).await.unwrap();

        tx.send(true).unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(5), writer)
            .await
            .expect("writer exits once its idle drain window passes")
            .unwrap();
    }
}
