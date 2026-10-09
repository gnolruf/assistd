//! Opens the database and hands out the shared [`SqliteHandle`].

use std::ffi::OsString;
use std::io;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tokio_rusqlite::Connection;

use crate::{MemoryError, Result, migrations};

use super::writer::{WriteOp, dispatch_write, spawn_writer};

/// Deep enough to absorb one bursty agent step without backpressuring the sender.
const WRITER_QUEUE_DEPTH: usize = 256;

/// Mode for directories created to hold the database.
const PRIVATE_DIR_MODE: u32 = 0o700;

/// Mode for the database and its sidecars; SQLite copies the database's mode
/// onto the WAL and SHM files it creates.
const PRIVATE_FILE_MODE: u32 = 0o600;

/// Suffixes SQLite appends to the database path for its sidecar files.
const SIDECAR_SUFFIXES: [&str; 3] = ["-wal", "-shm", "-journal"];

/// Cheaply cloneable handle shared by every store: reads hit the connection directly,
/// writes go through the writer task.
#[derive(Debug, Clone)]
pub struct SqliteHandle {
    pub(super) conn: Connection,
    pub(super) writer_tx: Arc<mpsc::Sender<WriteOp>>,
}

impl SqliteHandle {
    /// Open `path` (creating parent directories), restrict it to its owner, apply
    /// pragmas, run migrations, and spawn the writer task, returning it alongside
    /// the handle.
    pub async fn open(
        path: &Path,
        shutdown: watch::Receiver<bool>,
    ) -> Result<(Self, JoinHandle<()>)> {
        prepare_private_db_file(path).await?;

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

async fn prepare_private_db_file(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::DirBuilder::new()
            .recursive(true)
            .mode(PRIVATE_DIR_MODE)
            .create(parent)
            .await
            .map_err(|source| MemoryError::CreateDir {
                path: parent.to_path_buf(),
                source,
            })?;
    }

    tokio::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .mode(PRIVATE_FILE_MODE)
        .open(path)
        .await
        .map_err(|source| MemoryError::Restrict {
            path: path.to_path_buf(),
            source,
        })?;

    for file in std::iter::once(path.to_path_buf()).chain(sidecar_paths(path)) {
        restrict_if_present(&file)
            .await
            .map_err(|source| MemoryError::Restrict { path: file, source })?;
    }
    Ok(())
}

fn sidecar_paths(path: &Path) -> impl Iterator<Item = PathBuf> {
    SIDECAR_SUFFIXES.iter().map(|suffix| {
        let mut name = OsString::from(path.as_os_str());
        name.push(suffix);
        PathBuf::from(name)
    })
}

async fn restrict_if_present(path: &Path) -> io::Result<()> {
    let permissions = std::fs::Permissions::from_mode(PRIVATE_FILE_MODE);
    match tokio::fs::set_permissions(path, permissions).await {
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        result => result,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mode_of(path: &Path) -> u32 {
        std::fs::metadata(path).unwrap().permissions().mode() & 0o777
    }

    #[tokio::test]
    async fn open_creates_owner_only_database_and_dirs() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("nested/memory.db");

        let (_tx, rx) = watch::channel(false);
        let (handle, writer) = SqliteHandle::open(&path, rx).await.unwrap();

        assert_eq!(mode_of(&path), PRIVATE_FILE_MODE);
        assert_eq!(
            mode_of(&path.with_file_name("memory.db-wal")),
            PRIVATE_FILE_MODE
        );
        assert_eq!(mode_of(path.parent().unwrap()), PRIVATE_DIR_MODE);

        drop(handle);
        writer.await.unwrap();
    }

    #[tokio::test]
    async fn open_tightens_existing_world_readable_files() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("memory.db");
        let wal = temp.path().join("memory.db-wal");
        for file in [&path, &wal] {
            std::fs::write(file, b"").unwrap();
            std::fs::set_permissions(file, std::fs::Permissions::from_mode(0o644)).unwrap();
        }

        let (_tx, rx) = watch::channel(false);
        let (handle, writer) = SqliteHandle::open(&path, rx).await.unwrap();

        assert_eq!(mode_of(&path), PRIVATE_FILE_MODE);
        assert_eq!(mode_of(&wal), PRIVATE_FILE_MODE);

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
