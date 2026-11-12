//! Opens the database and hands out the shared [`SqliteHandle`].

use std::ffi::OsString;
use std::io;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use rusqlite::{OpenFlags, Transaction};
use tokio_rusqlite::Connection;

use crate::{MemoryError, Result, migrations};

/// How long a statement waits on a lock held by another process (e.g. a
/// `sqlite3` shell) before failing with `SQLITE_BUSY`.
const BUSY_TIMEOUT: Duration = Duration::from_secs(5);

/// Page size of a newly created database; packs four 768-dimension or three
/// 1024-dimension `f32` vectors per leaf page, where 8 KiB fits only one 1024-dimension vector.
const PAGE_SIZE_BYTES: i64 = 16384;

/// How much of the database file each read-only connection may memory-map.
const READER_MMAP_BYTES: i64 = 256 * 1024 * 1024;

/// Mode for directories created to hold the database.
const PRIVATE_DIR_MODE: u32 = 0o700;

/// Mode for the database and its sidecars; SQLite copies the database's mode
/// onto the WAL and SHM files it creates.
const PRIVATE_FILE_MODE: u32 = 0o600;

/// Suffixes SQLite appends to the database path for its sidecar files.
const SIDECAR_SUFFIXES: [&str; 3] = ["-wal", "-shm", "-journal"];

/// Cheaply cloneable handle shared by every store: document vector scans, other
/// vector scans and other reads each go to their own read-only connection and writes
/// to the only writable one, each run in submission order on its connection's own thread.
#[derive(Debug, Clone)]
pub struct SqliteHandle {
    reader: Connection,
    search_reader: Connection,
    document_search_reader: Connection,
    writer: Connection,
}

impl SqliteHandle {
    /// Open `path` (creating parent directories), restrict it to its owner, apply
    /// pragmas, and run migrations.
    pub async fn open(path: &Path) -> Result<Self> {
        prepare_private_db_file(path).await?;
        let writer = open_writer(path).await?;
        let reader = open_reader(path).await?;
        let search_reader = open_reader(path).await?;
        let document_search_reader = open_reader(path).await?;
        Ok(Self {
            reader,
            search_reader,
            document_search_reader,
            writer,
        })
    }

    /// Refresh planner statistics and close every connection once every call
    /// already queued has run; later calls through any clone fail. The writer
    /// closes last so it checkpoints the WAL into the database and removes it.
    pub async fn close(self) -> Result<()> {
        if let Err(e) = self
            .writer
            .call(|c| c.execute_batch("PRAGMA optimize"))
            .await
        {
            tracing::warn!(
                target: "assistd::memory",
                error = %e,
                "PRAGMA optimize failed before close"
            );
        }
        self.reader
            .close()
            .await
            .map_err(MemoryError::sqlite("close reader connection"))?;
        self.search_reader
            .close()
            .await
            .map_err(MemoryError::sqlite("close search reader connection"))?;
        self.document_search_reader
            .close()
            .await
            .map_err(MemoryError::sqlite(
                "close document search reader connection",
            ))?;
        self.writer
            .close()
            .await
            .map_err(MemoryError::sqlite("close writer connection"))
    }

    /// Run `query` on the general read-only connection; a failure is reported as `op`.
    pub(super) async fn read<T, F>(&self, op: &'static str, query: F) -> Result<T>
    where
        F: FnOnce(&rusqlite::Connection) -> rusqlite::Result<T> + Send + 'static,
        T: Send + 'static,
    {
        read_on(&self.reader, op, query).await
    }

    /// Like [`Self::read`], with every statement of `query` seeing one snapshot.
    pub(super) async fn read_snapshot<T, F>(&self, op: &'static str, query: F) -> Result<T>
    where
        F: FnOnce(&Transaction<'_>) -> rusqlite::Result<T> + Send + 'static,
        T: Send + 'static,
    {
        self.reader
            .call(move |c| in_transaction(c, query))
            .await
            .map_err(MemoryError::sqlite(op))
    }

    /// Like [`Self::read`], on the connection reserved for full scans of message chunk
    /// and memory vectors.
    pub(super) async fn search<T, F>(&self, op: &'static str, query: F) -> Result<T>
    where
        F: FnOnce(&rusqlite::Connection) -> rusqlite::Result<T> + Send + 'static,
        T: Send + 'static,
    {
        read_on(&self.search_reader, op, query).await
    }

    /// Like [`Self::read`], on the connection reserved for full scans of document chunk vectors.
    pub(super) async fn search_documents<T, F>(&self, op: &'static str, query: F) -> Result<T>
    where
        F: FnOnce(&rusqlite::Connection) -> rusqlite::Result<T> + Send + 'static,
        T: Send + 'static,
    {
        read_on(&self.document_search_reader, op, query).await
    }

    /// Run `statements` in one transaction on the writable connection and commit it;
    /// a failure is reported as `op`.
    pub(super) async fn write<T, F>(&self, op: &'static str, statements: F) -> Result<T>
    where
        F: FnOnce(&Transaction<'_>) -> rusqlite::Result<T> + Send + 'static,
        T: Send + 'static,
    {
        self.try_write(statements)
            .await
            .map_err(MemoryError::sqlite(op))
    }

    /// Like [`Self::write`], leaving the failure for the caller to classify.
    pub(super) async fn try_write<T, F>(&self, statements: F) -> Result<T, tokio_rusqlite::Error>
    where
        F: FnOnce(&Transaction<'_>) -> rusqlite::Result<T> + Send + 'static,
        T: Send + 'static,
    {
        self.writer
            .call(move |c| in_transaction(c, statements))
            .await
    }

    /// Like [`Self::write`], then truncate the WAL when the committed write erased rows.
    pub(super) async fn write_erasing<T, F>(&self, op: &'static str, statements: F) -> Result<T>
    where
        F: FnOnce(&Transaction<'_>) -> rusqlite::Result<Erasing<T>> + Send + 'static,
        T: Send + 'static,
    {
        self.writer
            .call(move |c| {
                let Erasing { value, erased } = in_transaction(c, statements)?;
                if erased {
                    truncate_wal(c);
                }
                Ok(value)
            })
            .await
            .map_err(MemoryError::sqlite(op))
    }
}

/// A write's result, and whether it deleted or overwrote rows whose old contents
/// must not outlive it in the WAL.
pub(super) struct Erasing<T> {
    pub value: T,
    pub erased: bool,
}

async fn read_on<T, F>(reader: &Connection, op: &'static str, query: F) -> Result<T>
where
    F: FnOnce(&rusqlite::Connection) -> rusqlite::Result<T> + Send + 'static,
    T: Send + 'static,
{
    reader
        .call(move |c| query(c))
        .await
        .map_err(MemoryError::sqlite(op))
}

fn in_transaction<T>(
    c: &mut rusqlite::Connection,
    statements: impl FnOnce(&Transaction<'_>) -> rusqlite::Result<T>,
) -> rusqlite::Result<T> {
    let tx = c.transaction()?;
    let value = statements(&tx)?;
    tx.commit()?;
    Ok(value)
}

/// The writable connection: applies pragmas, migrates, and refreshes planner statistics.
async fn open_writer(path: &Path) -> Result<Connection> {
    let conn = Connection::open(path)
        .await
        .map_err(|source| MemoryError::Open {
            path: path.to_path_buf(),
            source: source.into(),
        })?;

    conn.call(|c| -> rusqlite::Result<_> {
        c.pragma_update(None, "page_size", PAGE_SIZE_BYTES)?;
        c.pragma_update(None, "journal_mode", "WAL")?;
        c.pragma_update(None, "synchronous", "NORMAL")?;
        c.pragma_update(None, "foreign_keys", "ON")?;
        c.pragma_update(None, "secure_delete", "ON")?;
        c.busy_timeout(BUSY_TIMEOUT)
    })
    .await
    .map_err(MemoryError::sqlite("apply SQLite pragmas"))?;

    conn.call(migrations::run)
        .await
        .map_err(|e| MemoryError::Migration(e.into()))?;

    conn.call(|c| c.execute_batch("PRAGMA optimize=0x10002"))
        .await
        .map_err(MemoryError::sqlite("optimize query planner statistics"))?;
    Ok(conn)
}

/// A read-only connection; opened after the writer has switched the database to WAL.
async fn open_reader(path: &Path) -> Result<Connection> {
    let flags = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX;
    let conn = Connection::open_with_flags(path, flags)
        .await
        .map_err(|source| MemoryError::Open {
            path: path.to_path_buf(),
            source: source.into(),
        })?;
    conn.call(|c| -> rusqlite::Result<_> {
        c.pragma_update(None, "mmap_size", READER_MMAP_BYTES)?;
        c.busy_timeout(BUSY_TIMEOUT)
    })
    .await
    .map_err(MemoryError::sqlite("apply SQLite reader pragmas"))?;
    Ok(conn)
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

/// Checkpoint and empty the WAL so frames holding deleted or overwritten rows
/// don't outlive the write. Never waits on readers: a blocked or failed checkpoint
/// is logged, and the frames go at the next checkpoint or at close.
fn truncate_wal(c: &rusqlite::Connection) {
    match checkpoint_without_waiting(c) {
        Ok(true) => tracing::warn!(
            target: "assistd::memory",
            "WAL checkpoint blocked by a reader; old rows stay in the WAL until the next checkpoint"
        ),
        Ok(false) => {}
        Err(e) => tracing::warn!(
            target: "assistd::memory",
            error = %e,
            "WAL checkpoint failed; old rows stay in the WAL until the next checkpoint"
        ),
    }
}

/// Run a truncating checkpoint with the busy handler off, returning whether a
/// reader blocked it.
fn checkpoint_without_waiting(c: &rusqlite::Connection) -> rusqlite::Result<bool> {
    c.busy_timeout(Duration::ZERO)?;
    let blocked = c.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |r| r.get(0));
    c.busy_timeout(BUSY_TIMEOUT)?;
    blocked
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

        let handle = SqliteHandle::open(&path).await.unwrap();

        assert_eq!(mode_of(&path), PRIVATE_FILE_MODE);
        assert_eq!(
            mode_of(&path.with_file_name("memory.db-wal")),
            PRIVATE_FILE_MODE
        );
        assert_eq!(mode_of(path.parent().unwrap()), PRIVATE_DIR_MODE);

        handle.close().await.unwrap();
        assert!(
            !path.with_file_name("memory.db-wal").exists(),
            "close must checkpoint and remove the WAL"
        );
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

        let handle = SqliteHandle::open(&path).await.unwrap();

        assert_eq!(mode_of(&path), PRIVATE_FILE_MODE);
        assert_eq!(mode_of(&wal), PRIVATE_FILE_MODE);

        handle.close().await.unwrap();
    }

    #[tokio::test]
    async fn open_refuses_database_newer_than_the_binary() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("memory.db");
        SqliteHandle::open(&path)
            .await
            .unwrap()
            .close()
            .await
            .unwrap();
        rusqlite::Connection::open(&path)
            .unwrap()
            .pragma_update(None, "user_version", 9999)
            .unwrap();

        let error = SqliteHandle::open(&path).await.unwrap_err();

        assert!(matches!(error, MemoryError::Migration(_)), "{error}");
    }
}
