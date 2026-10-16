//! Error type for the memory database.

use std::path::PathBuf;

use thiserror::Error;

use crate::{BranchId, SessionId};

/// Failures from opening or operating on the memory database.
#[derive(Debug, Error)]
pub enum MemoryError {
    #[error("create memory.db parent dir {}: {source}", path.display())]
    CreateDir {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("restrict {} to owner-only access: {source}", path.display())]
    Restrict {
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

    #[error("no branch with id {0}")]
    BranchNotFound(BranchId),

    #[error("no saved session {0}")]
    SessionNotFound(SessionId),

    #[error("this session already has a branch named {0:?}")]
    BranchNameTaken(String),

    /// A SQLite statement issued by `op` failed.
    #[error("{op}: {source}")]
    Sqlite {
        op: &'static str,
        #[source]
        source: tokio_rusqlite::Error,
    },
}

impl MemoryError {
    pub(crate) fn sqlite(op: &'static str) -> impl FnOnce(tokio_rusqlite::Error) -> Self {
        move |source| Self::Sqlite { op, source }
    }

    /// Like [`MemoryError::sqlite`], but a query that matched no row becomes `missing`.
    pub(crate) fn sqlite_or_missing(
        op: &'static str,
        missing: Self,
    ) -> impl FnOnce(tokio_rusqlite::Error) -> Self {
        move |source| match source {
            tokio_rusqlite::Error::Error(rusqlite::Error::QueryReturnedNoRows) => missing,
            source => Self::Sqlite { op, source },
        }
    }
}

/// Result alias for this crate's fallible operations.
pub type Result<T, E = MemoryError> = std::result::Result<T, E>;
