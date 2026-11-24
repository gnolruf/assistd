//! Opens the three database files and hands out the shared [`SqliteHandle`].

use std::ffi::OsString;
use std::io;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use rusqlite::{ErrorCode, OpenFlags, Transaction};
use tokio::sync::Notify;
use tokio_rusqlite::Connection;

use crate::{MemoryError, MemoryId, Result, index_schema, migrations, vectors_schema};

use super::chunks::delete_vectors_of_removed_texts;
use super::page_readings::delete_readings_of_removed_files;
use super::reconcile;

/// The file beside `memory.db` that holds what can be built again from it and from
/// the files on disk.
const INDEX_FILE_NAME: &str = "index.db";

/// The file beside `memory.db` that holds the vectors of the texts `index.db` chunks
/// and the readings of the pages a model transcribed.
const VECTORS_FILE_NAME: &str = "vectors.db";

/// The schema name `memory.db` has on the index writer.
const ATTACHED_MEMORY: &str = "mem";

/// The schema name `index.db` has on the read-only connections.
const ATTACHED_INDEX: &str = "idx";

/// The schema name `vectors.db` has on the index writer and the read-only connections.
const ATTACHED_VECTORS: &str = "vec";

/// The databases whose WAL `memory.db`'s writer truncates after erasing rows.
const MEMORY_WRITER_FILES: &[&str] = &["main"];

/// The databases whose WAL the index writer truncates after erasing rows.
const INDEX_WRITER_FILES: &[&str] = &["main", ATTACHED_VECTORS];

/// `index.db` as the daemon replaces it: chunks and document text it cuts again.
const INDEX_FILE: ReplaceableFile = ReplaceableFile {
    name: "index.db",
    version: index_schema::VERSION,
};

/// `vectors.db` as the daemon replaces it: vectors and page readings a model computes
/// again.
const VECTORS_FILE: ReplaceableFile = ReplaceableFile {
    name: "vectors.db",
    version: vectors_schema::VERSION,
};

/// How long a statement waits on a lock held by another process (e.g. a
/// `sqlite3` shell) before failing with `SQLITE_BUSY`.
const BUSY_TIMEOUT: Duration = Duration::from_secs(5);

/// Page size of a newly created database; packs four 768-dimension or three
/// 1024-dimension `f32` vectors per leaf page, where 8 KiB fits only one 1024-dimension vector.
const PAGE_SIZE_BYTES: i64 = 16384;

/// How much of each database file a read-only connection may memory-map.
const READER_MMAP_BYTES: i64 = 256 * 1024 * 1024;

/// Mode for directories created to hold the databases.
const PRIVATE_DIR_MODE: u32 = 0o700;

/// Mode for a database and its sidecars; SQLite copies the database's mode
/// onto the WAL and SHM files it creates.
const PRIVATE_FILE_MODE: u32 = 0o600;

/// Suffixes SQLite appends to a database path for its sidecar files.
const SIDECAR_SUFFIXES: [&str; 3] = ["-wal", "-shm", "-journal"];

/// Cheaply cloneable handle shared by every store, over `memory.db` and the `index.db`
/// and `vectors.db` beside it. `memory.db` has one writable connection and the other
/// two share one, each running its writes in submission order on its own thread;
/// document searches, other vector scans and other reads each go to their own
/// read-only connection onto all three files, each file mapped apart from the others.
/// It also carries the wake signal of each backlog the databases hold.
#[derive(Debug, Clone)]
pub struct SqliteHandle {
    reader: Connection,
    search_reader: Connection,
    document_search_reader: Connection,
    writer: Connection,
    /// Writes `index.db` and `vectors.db`, with `memory.db` attached to read.
    index_writer: Connection,
    rows_to_embed: Arc<Notify>,
    pages_to_transcribe: Arc<Notify>,
    messages_to_chunk: Arc<Notify>,
}

impl SqliteHandle {
    /// Open `memory.db` at `path` with `index.db` and `vectors.db` beside it (creating
    /// parent directories), restrict them to their owner, apply pragmas, and migrate
    /// `memory.db`. Either of the other two that is not a readable database or holds
    /// another schema version is replaced by an empty one, and the message, memory and
    /// access chunks of `index.db` are brought in line with `memory.db`.
    pub async fn open(path: &Path) -> Result<Self> {
        let files = DatabaseFiles {
            memory: path,
            index: &path.with_file_name(INDEX_FILE_NAME),
            vectors: &path.with_file_name(VECTORS_FILE_NAME),
        };
        prepare_private_db_file(path).await?;
        let writer = open_writer(path).await?;
        let vectors_created = create_vectors_file(files.vectors).await?;
        let index_writer = open_index_writer(&files).await?;
        let reader = open_reader(&files).await?;
        let search_reader = open_reader(&files).await?;
        let document_search_reader = open_reader(&files).await?;
        let handle = Self {
            reader,
            search_reader,
            document_search_reader,
            writer,
            index_writer,
            rows_to_embed: Arc::default(),
            pages_to_transcribe: Arc::default(),
            messages_to_chunk: Arc::default(),
        };
        handle.reconcile_index().await?;
        if vectors_created {
            handle.log_texts_without_vector().await?;
        }
        Ok(handle)
    }

    /// Signalled once a committed write has left a row to embed.
    pub(super) fn rows_to_embed(&self) -> &Arc<Notify> {
        &self.rows_to_embed
    }

    /// Signalled once a committed write has left a page waiting for transcription.
    pub(super) fn pages_to_transcribe(&self) -> &Arc<Notify> {
        &self.pages_to_transcribe
    }

    /// Signalled once a committed write has saved a message that has chunks to cut.
    pub(super) fn messages_to_chunk(&self) -> &Arc<Notify> {
        &self.messages_to_chunk
    }

    /// Refresh planner statistics and close every connection once every call
    /// already queued has run; later calls through any clone fail. The writers close
    /// after the readers so each checkpoints the WAL of its files into them and
    /// removes it.
    pub async fn close(self) -> Result<()> {
        for writer in [&self.index_writer, &self.writer] {
            if let Err(e) = writer
                .call(|c| c.execute_batch("PRAGMA main.optimize"))
                .await
            {
                tracing::warn!(
                    target: "assistd::memory",
                    error = %e,
                    "PRAGMA optimize failed before close"
                );
            }
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
        self.index_writer
            .close()
            .await
            .map_err(MemoryError::sqlite("close index writer connection"))?;
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

    /// Like [`Self::read`], on the connection reserved for full scans of message and
    /// memory chunk vectors.
    pub(super) async fn search<T, F>(&self, op: &'static str, query: F) -> Result<T>
    where
        F: FnOnce(&rusqlite::Connection) -> rusqlite::Result<T> + Send + 'static,
        T: Send + 'static,
    {
        read_on(&self.search_reader, op, query).await
    }

    /// Like [`Self::read`], on the connection reserved for document searches, which
    /// scan every document chunk vector.
    pub(super) async fn search_documents<T, F>(&self, op: &'static str, query: F) -> Result<T>
    where
        F: FnOnce(&rusqlite::Connection) -> rusqlite::Result<T> + Send + 'static,
        T: Send + 'static,
    {
        read_on(&self.document_search_reader, op, query).await
    }

    /// Run `statements` in one transaction on `memory.db`'s writable connection and
    /// commit it; a failure is reported as `op`.
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

    /// Like [`Self::write`]. When the committed write erased rows it truncates the
    /// WAL, then removes from `index.db` the chunks of the messages, memories and accesses that
    /// are gone.
    pub(super) async fn write_erasing<T, F>(&self, op: &'static str, statements: F) -> Result<T>
    where
        F: FnOnce(&Transaction<'_>) -> rusqlite::Result<Erasing<T>> + Send + 'static,
        T: Send + 'static,
    {
        let commit = move |c: &mut rusqlite::Connection| in_transaction(c, statements);
        let (value, erased) = commit_erasing(&self.writer, MEMORY_WRITER_FILES, op, commit).await?;
        if erased {
            self.remove_orphaned_chunks().await;
        }
        Ok(value)
    }

    /// Run `statements` in one transaction on the writable connection onto `index.db`
    /// and `vectors.db`, drop the vectors of the texts it removed the last chunk of
    /// and the readings of the files it removed the last document of, and commit it;
    /// a failure is reported as `op`. A crash can keep one file's half of the commit
    /// without the other's.
    pub(super) async fn write_index<T, F>(&self, op: &'static str, statements: F) -> Result<T>
    where
        F: FnOnce(&Transaction<'_>) -> rusqlite::Result<T> + Send + 'static,
        T: Send + 'static,
    {
        self.index_writer
            .call(move |c| in_index_transaction(c, statements))
            .await
            .map_err(MemoryError::sqlite(op))
    }

    /// Like [`Self::write_index`], then truncate the WAL of both files when the
    /// committed write erased rows.
    pub(super) async fn write_index_erasing<T, F>(
        &self,
        op: &'static str,
        statements: F,
    ) -> Result<T>
    where
        F: FnOnce(&Transaction<'_>) -> rusqlite::Result<Erasing<T>> + Send + 'static,
        T: Send + 'static,
    {
        let commit = move |c: &mut rusqlite::Connection| in_index_transaction(c, statements);
        let (value, _) = commit_erasing(&self.index_writer, INDEX_WRITER_FILES, op, commit).await?;
        Ok(value)
    }

    /// Cut again the chunk of `memory` when it does not cover the value `memory.db`
    /// now holds, and wake the embedding worker for a new one. A failure is logged:
    /// the chunk is then cut at the next start.
    pub(super) async fn recut_memory_chunk(&self, memory: MemoryId) {
        let recut = self
            .write_index_erasing("recut memory chunk", move |tx| {
                let recut = reconcile::recut_changed_memories(tx, Some(memory))?;
                Ok(Erasing {
                    erased: recut.replaced > 0,
                    value: recut,
                })
            })
            .await;
        match recut {
            Ok(recut) if recut.cut > 0 => self.rows_to_embed.notify_one(),
            Ok(_) => {}
            Err(e) => tracing::warn!(
                target: "assistd::memory",
                error = %e,
                memory = memory.0,
                "cannot cut the saved memory's chunk; search has its old value until the next start"
            ),
        }
    }

    /// Remove the chunks whose message, memory or access `memory.db` no longer holds. A
    /// failure is logged: searches skip those chunks, and the next start removes them.
    async fn remove_orphaned_chunks(&self) {
        let removed = self
            .write_index_erasing("remove orphaned chunks", |tx| {
                let removed = reconcile::remove_orphaned_chunks(tx)?;
                Ok(Erasing {
                    value: (),
                    erased: removed > 0,
                })
            })
            .await;
        if let Err(e) = removed {
            tracing::warn!(
                target: "assistd::memory",
                error = %e,
                "cannot remove the chunks of deleted rows; they stay in index.db until the next start"
            );
        }
    }

    async fn reconcile_index(&self) -> Result<()> {
        let reconciled = self
            .write_index_erasing("reconcile index.db with memory.db", |tx| {
                let reconciled = reconcile::reconcile(tx)?;
                Ok(Erasing {
                    erased: reconciled.erased_rows(),
                    value: reconciled,
                })
            })
            .await?;
        if reconciled.cut_from_other_messages {
            tracing::warn!(
                target: "assistd::memory",
                "index.db was built from other messages than memory.db holds; its message and \
                 access chunks are dropped and cut again"
            );
        }
        tracing::debug!(
            target: "assistd::memory",
            orphans_removed = reconciled.orphans_removed,
            memories_recut = reconciled.memories.cut,
            memory_vectors_removed = reconciled.unheld_vectors_removed,
            "index.db reconciled with memory.db"
        );
        Ok(())
    }

    /// Remove the message and access vectors whose text no chunk of their kind holds,
    /// unless saved messages still wait to be chunked; returns whether it could.
    pub(super) async fn remove_unheld_transcript_vectors(&self) -> Result<bool> {
        let removed = self
            .write_index_erasing("remove unheld message and access vectors", |tx| {
                let removed = reconcile::remove_unheld_transcript_vectors(tx)?;
                Ok(Erasing {
                    erased: removed.is_some_and(|vectors| vectors > 0),
                    value: removed,
                })
            })
            .await?;
        if let Some(vectors) = removed {
            tracing::debug!(
                target: "assistd::memory",
                vectors,
                "removed the message and access vectors no chunk holds the text of"
            );
        }
        Ok(removed.is_some())
    }

    async fn log_texts_without_vector(&self) -> Result<()> {
        let texts: u64 = self
            .read("count chunk texts without a vector", |c| {
                c.query_row(
                    "SELECT count(*) FROM (SELECT DISTINCT kind, hash FROM chunks)",
                    [],
                    |row| row.get(0),
                )
            })
            .await?;
        tracing::info!(
            target: "assistd::memory",
            texts,
            "vectors.db was created empty; every chunk text waits to be embedded"
        );
        Ok(())
    }
}

/// Where the three database files are.
struct DatabaseFiles<'a> {
    memory: &'a Path,
    index: &'a Path,
    vectors: &'a Path,
}

/// A database file holding only what the daemon can produce again, stamped with the
/// version of its schema.
struct ReplaceableFile {
    name: &'static str,
    version: i64,
}

/// How a connection may use a database it attaches.
#[derive(Clone, Copy)]
enum Access {
    ReadOnly,
    ReadWrite,
}

/// A write's result, and whether it deleted or overwrote rows whose old contents
/// must not outlive it in the WAL.
pub(super) struct Erasing<T> {
    pub value: T,
    pub erased: bool,
}

/// What a probe of a [`ReplaceableFile`] on disk found.
enum StoredFile {
    /// No file, or an empty one.
    Absent,
    /// A database, with the schema version it is stamped with.
    Version(i64),
    /// A file SQLite cannot read as a database.
    Unreadable,
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

/// Run `commit` on `writer` and truncate the WAL of each of `files`, its schema names
/// for them, when the write it committed erased rows. Returns the write's value and
/// whether it erased any.
async fn commit_erasing<T, F>(
    writer: &Connection,
    files: &'static [&'static str],
    op: &'static str,
    commit: F,
) -> Result<(T, bool)>
where
    F: FnOnce(&mut rusqlite::Connection) -> rusqlite::Result<Erasing<T>> + Send + 'static,
    T: Send + 'static,
{
    writer
        .call(move |c| {
            let Erasing { value, erased } = commit(c)?;
            if erased {
                files.iter().for_each(|file| truncate_wal(c, file));
            }
            Ok((value, erased))
        })
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

fn in_index_transaction<T>(
    c: &mut rusqlite::Connection,
    statements: impl FnOnce(&Transaction<'_>) -> rusqlite::Result<T>,
) -> rusqlite::Result<T> {
    let tx = c.transaction()?;
    let value = statements(&tx)?;
    delete_vectors_of_removed_texts(&tx)?;
    delete_readings_of_removed_files(&tx)?;
    tx.commit()?;
    Ok(value)
}

/// The writable connection onto `memory.db`: applies pragmas, migrates, and refreshes
/// planner statistics.
async fn open_writer(path: &Path) -> Result<Connection> {
    let conn = open_writable(path).await?;
    conn.call(migrations::run)
        .await
        .map_err(|e| MemoryError::Migration(e.into()))?;
    refresh_planner_statistics(&conn).await?;
    Ok(conn)
}

/// The writable connection onto `index.db`, with `memory.db` attached to read and
/// `vectors.db` to write. An `index.db` that is not a readable database or holds
/// another schema version is deleted first, and an empty one is given the schema.
async fn open_index_writer(files: &DatabaseFiles<'_>) -> Result<Connection> {
    discard_unusable(files.index, &INDEX_FILE).await?;
    prepare_private_db_file(files.index).await?;
    let conn = open_writable(files.index).await?;
    conn.call(index_schema::create)
        .await
        .map_err(MemoryError::sqlite("create the index.db schema"))?;
    attach(&conn, files.memory, ATTACHED_MEMORY, Access::ReadOnly).await?;
    attach(&conn, files.vectors, ATTACHED_VECTORS, Access::ReadWrite).await?;
    conn.call(|c| -> rusqlite::Result<_> {
        c.pragma_update(Some(ATTACHED_VECTORS), "synchronous", "NORMAL")?;
        c.pragma_update(Some(ATTACHED_VECTORS), "secure_delete", "ON")
    })
    .await
    .map_err(MemoryError::sqlite("apply SQLite pragmas to vectors.db"))?;
    refresh_planner_statistics(&conn).await?;
    Ok(conn)
}

/// Give the `vectors.db` at `path` its schema, after deleting one that is not a
/// readable database or holds another schema version. Returns whether the file is new.
async fn create_vectors_file(path: &Path) -> Result<bool> {
    let created = discard_unusable(path, &VECTORS_FILE).await?;
    prepare_private_db_file(path).await?;
    let conn = open_writable(path).await?;
    conn.call(vectors_schema::create)
        .await
        .map_err(MemoryError::sqlite("create the vectors.db schema"))?;
    conn.close().await.map_err(|source| MemoryError::Open {
        path: path.to_path_buf(),
        source: source.into(),
    })?;
    Ok(created)
}

/// A writable connection onto the database at `path` with the pragmas every writer runs under.
async fn open_writable(path: &Path) -> Result<Connection> {
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
    Ok(conn)
}

async fn refresh_planner_statistics(writer: &Connection) -> Result<()> {
    writer
        .call(|c| c.execute_batch("PRAGMA main.optimize=0x10002"))
        .await
        .map_err(MemoryError::sqlite("optimize query planner statistics"))
}

/// A read-only connection onto `memory.db` with `index.db` and `vectors.db` attached,
/// each mapped up to [`READER_MMAP_BYTES`]; opened after the writers have switched
/// every database to WAL.
async fn open_reader(files: &DatabaseFiles<'_>) -> Result<Connection> {
    let path = files.memory;
    let flags = OpenFlags::SQLITE_OPEN_READ_ONLY
        | OpenFlags::SQLITE_OPEN_NO_MUTEX
        | OpenFlags::SQLITE_OPEN_URI;
    let conn = Connection::open_with_flags(path, flags)
        .await
        .map_err(|source| MemoryError::Open {
            path: path.to_path_buf(),
            source: source.into(),
        })?;
    attach(&conn, files.index, ATTACHED_INDEX, Access::ReadOnly).await?;
    attach(&conn, files.vectors, ATTACHED_VECTORS, Access::ReadOnly).await?;
    conn.call(|c| -> rusqlite::Result<_> {
        c.pragma_update(None, "mmap_size", READER_MMAP_BYTES)?;
        c.busy_timeout(BUSY_TIMEOUT)
    })
    .await
    .map_err(MemoryError::sqlite("apply SQLite reader pragmas"))?;
    Ok(conn)
}

/// Attach the database at `path` to `conn` under `schema`.
async fn attach(
    conn: &Connection,
    path: &Path,
    schema: &'static str,
    access: Access,
) -> Result<()> {
    let uri = database_uri(path, access);
    conn.call(move |c| c.execute(&format!("ATTACH DATABASE ?1 AS {schema}"), [uri]))
        .await
        .map_err(|source| MemoryError::Open {
            path: path.to_path_buf(),
            source: source.into(),
        })?;
    Ok(())
}

/// The SQLite URI that opens the existing file at `path` for `access`, with every byte
/// a URI gives a meaning to percent-encoded.
fn database_uri(path: &Path, access: Access) -> String {
    let encoded: String = path
        .as_os_str()
        .as_bytes()
        .iter()
        .map(|&byte| {
            if byte.is_ascii_alphanumeric() || b"/-._~".contains(&byte) {
                char::from(byte).to_string()
            } else {
                format!("%{byte:02X}")
            }
        })
        .collect();
    let mode = match access {
        Access::ReadOnly => "ro",
        Access::ReadWrite => "rw",
    };
    format!("file:{encoded}?mode={mode}")
}

/// Delete `file` at `path` with its sidecars unless it is absent or holds this
/// build's schema version. Returns whether no usable file is left there.
async fn discard_unusable(path: &Path, file: &ReplaceableFile) -> Result<bool> {
    match stored_file(path).await? {
        StoredFile::Absent => return Ok(true),
        StoredFile::Version(version) if version == file.version => return Ok(false),
        StoredFile::Version(version) => tracing::warn!(
            target: "assistd::memory",
            path = %path.display(),
            version,
            expected = file.version,
            "{} holds another schema version; building it again",
            file.name
        ),
        StoredFile::Unreadable => tracing::warn!(
            target: "assistd::memory",
            path = %path.display(),
            "{} is not a readable database; building it again",
            file.name
        ),
    }
    for file in std::iter::once(path.to_path_buf()).chain(sidecar_paths(path)) {
        match tokio::fs::remove_file(&file).await {
            Err(source) if source.kind() != io::ErrorKind::NotFound => {
                return Err(MemoryError::DiscardUnusable { path: file, source });
            }
            _ => {}
        }
    }
    Ok(true)
}

/// What the replaceable file at `path` is, read without changing it.
async fn stored_file(path: &Path) -> Result<StoredFile> {
    let open_error = |source: tokio_rusqlite::Error| MemoryError::Open {
        path: path.to_path_buf(),
        source: source.into(),
    };
    match tokio::fs::metadata(path).await {
        Ok(file) if file.len() > 0 => {}
        Ok(_) => return Ok(StoredFile::Absent),
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(StoredFile::Absent),
        Err(source) => {
            return Err(MemoryError::Open {
                path: path.to_path_buf(),
                source: source.into(),
            });
        }
    }
    let conn = Connection::open(path)
        .await
        .map_err(|source| open_error(source.into()))?;
    let version = conn
        .call(|c| {
            c.busy_timeout(BUSY_TIMEOUT)?;
            let version = c.pragma_query_value(Some("main"), "user_version", |row| row.get(0))?;
            c.query_row("SELECT count(*) FROM sqlite_schema", [], |row| {
                row.get::<_, i64>(0)
            })?;
            Ok(version)
        })
        .await;
    conn.close().await.map_err(open_error)?;
    match version {
        Ok(version) => Ok(StoredFile::Version(version)),
        Err(e) if is_unreadable_database(&e) => Ok(StoredFile::Unreadable),
        Err(e) => Err(open_error(e)),
    }
}

/// Whether `error` is SQLite finding that a file is not a database or is a damaged one.
fn is_unreadable_database(error: &tokio_rusqlite::Error) -> bool {
    matches!(
        error,
        tokio_rusqlite::Error::Error(rusqlite::Error::SqliteFailure(failure, _))
            if matches!(failure.code, ErrorCode::NotADatabase | ErrorCode::DatabaseCorrupt)
    )
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

/// Checkpoint and empty the WAL of the database `c` names `file` so frames holding
/// deleted or overwritten rows don't outlive the write. Never waits on readers: a
/// blocked or failed checkpoint is logged, and the frames go at the next checkpoint
/// or at close.
fn truncate_wal(c: &rusqlite::Connection, file: &str) {
    match checkpoint_without_waiting(c, file) {
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
fn checkpoint_without_waiting(c: &rusqlite::Connection, file: &str) -> rusqlite::Result<bool> {
    c.busy_timeout(Duration::ZERO)?;
    let checkpoint = format!("PRAGMA {file}.wal_checkpoint(TRUNCATE)");
    let blocked = c.query_row(&checkpoint, [], |r| r.get(0));
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
    async fn open_replaces_an_index_or_vectors_file_that_is_unreadable_or_from_another_version() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("memory.db");
        for (name, version) in [
            (INDEX_FILE_NAME, index_schema::VERSION),
            (VECTORS_FILE_NAME, vectors_schema::VERSION),
        ] {
            let file = temp.path().join(name);
            std::fs::write(&file, b"not a database, and longer than a header").unwrap();

            let handle = SqliteHandle::open(&path).await.unwrap();
            handle.close().await.unwrap();
            rusqlite::Connection::open(&file)
                .unwrap()
                .pragma_update(None, "user_version", version + 1)
                .unwrap();

            let handle = SqliteHandle::open(&path).await.unwrap();
            handle.close().await.unwrap();
            let stored: i64 = rusqlite::Connection::open(&file)
                .unwrap()
                .pragma_query_value(None, "user_version", |row| row.get(0))
                .unwrap();
            assert_eq!(stored, version, "{name}");
            assert_eq!(mode_of(&file), PRIVATE_FILE_MODE, "{name}");
        }
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
