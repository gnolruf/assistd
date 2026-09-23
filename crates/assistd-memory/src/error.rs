use std::path::PathBuf;

use thiserror::Error;

/// Failures from opening or operating on the memory database.
#[derive(Debug, Error)]
pub enum MemoryError {
    #[error("create memory.db parent dir {}: {source}", path.display())]
    CreateDir {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("open SQLite at {}: {source}", path.display())]
    Open {
        path: PathBuf,
        #[source]
        source: rusqlite::Error,
    },

    #[error("run migrations: {0}")]
    Migration(#[source] tokio_rusqlite::Error<rusqlite_migration::Error>),

    /// A SQLite statement issued by `op` failed.
    #[error("{op}: {source}")]
    Sqlite {
        op: &'static str,
        #[source]
        source: tokio_rusqlite::Error,
    },

    #[error("serialize tool_calls: {0}")]
    SerializeToolCalls(#[from] serde_json::Error),

    /// A stored `role` column held a value outside the known set.
    #[error("unknown role in DB: {0}")]
    UnknownRole(String),

    #[error("memory writer task is gone")]
    WriterGone,

    #[error("memory writer task dropped ack channel")]
    AckDropped,
}

impl MemoryError {
    pub(crate) fn sqlite(op: &'static str) -> impl FnOnce(tokio_rusqlite::Error) -> Self {
        move |source| Self::Sqlite { op, source }
    }
}

/// Result alias for this crate's fallible operations.
pub type Result<T, E = MemoryError> = std::result::Result<T, E>;
