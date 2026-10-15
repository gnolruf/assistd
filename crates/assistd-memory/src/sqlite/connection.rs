//! Opens the database and hands out the shared [`SqliteHandle`].

use std::ffi::OsString;
use std::io;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use rusqlite::OpenFlags;
use tokio_rusqlite::Connection;

use crate::{MemoryError, Result, migrations};

/// How long a statement waits on a lock held by another process (e.g. a
/// `sqlite3` shell) before failing with `SQLITE_BUSY`.
const BUSY_TIMEOUT: Duration = Duration::from_secs(5);

/// Mode for directories created to hold the database.
const PRIVATE_DIR_MODE: u32 = 0o700;

/// Mode for the database and its sidecars; SQLite copies the database's mode
/// onto the WAL and SHM files it creates.
const PRIVATE_FILE_MODE: u32 = 0o600;

/// Suffixes SQLite appends to the database path for its sidecar files.
const SIDECAR_SUFFIXES: [&str; 3] = ["-wal", "-shm", "-journal"];

/// Cheaply cloneable handle shared by every store: reads go to a read-only
/// connection and writes to the only writable one, each run in submission
/// order on its connection's own thread.
#[derive(Debug, Clone)]
pub struct SqliteHandle {
    reader: Connection,
    writer: Connection,
}

impl SqliteHandle {
    /// Open `path` (creating parent directories), restrict it to its owner, apply
    /// pragmas, and run migrations.
    pub async fn open(path: &Path) -> Result<Self> {
        prepare_private_db_file(path).await?;
        let writer = open_writer(path).await?;
        let reader = open_reader(path).await?;
        Ok(Self { reader, writer })
    }

    /// Refresh planner statistics and close both connections once every call
    /// already queued has run; later calls through any clone fail.
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
        self.writer
            .close()
            .await
            .map_err(MemoryError::sqlite("close writer connection"))?;
        self.reader
            .close()
            .await
            .map_err(MemoryError::sqlite("close reader connection"))
    }

    pub(super) fn reader(&self) -> &Connection {
        &self.reader
    }

    pub(super) fn writer(&self) -> &Connection {
        &self.writer
    }
}

/// The writable connection: applies pragmas, migrates, and refreshes planner statistics.
async fn open_writer(path: &Path) -> Result<Connection> {
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
        c.pragma_update(None, "secure_delete", "ON")?;
        c.busy_timeout(BUSY_TIMEOUT)
    })
    .await
    .map_err(MemoryError::sqlite("apply SQLite pragmas"))?;

    conn.call(migrations::run)
        .await
        .map_err(MemoryError::Migration)?;

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
            source,
        })?;
    conn.call(|c| c.busy_timeout(BUSY_TIMEOUT))
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
/// don't outlive the write.
pub(super) fn truncate_wal(c: &rusqlite::Connection) -> rusqlite::Result<()> {
    let busy: bool = c.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |r| r.get(0))?;
    if busy {
        tracing::warn!(
            target: "assistd::memory",
            "WAL checkpoint blocked by another reader; old rows stay in the WAL until the next checkpoint"
        );
    }
    Ok(())
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
}
