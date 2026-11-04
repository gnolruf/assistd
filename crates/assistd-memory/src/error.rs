//! Error type for the memory database.

use std::path::PathBuf;

use thiserror::Error;

use assistd_ipc::{TaskId, TaskRunId};

use crate::{BranchId, SessionId, TurnId};

/// An error from the SQLite driver, kept opaque so callers don't depend on it.
pub type BackendError = Box<dyn std::error::Error + Send + Sync>;

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
        source: BackendError,
    },

    #[error("run migrations: {0}")]
    Migration(#[source] BackendError),

    #[error("no branch with id {0}")]
    BranchNotFound(BranchId),

    #[error("no saved session {0}")]
    SessionNotFound(SessionId),

    #[error("no turn with id {0}")]
    TurnNotFound(TurnId),

    #[error("this session already has a branch named {0:?}")]
    BranchNameTaken(String),

    #[error("no task with id {0}")]
    TaskNotFound(TaskId),

    #[error("a task named {0:?} already exists")]
    TaskNameTaken(String),

    #[error("no unfinished task run with id {0}")]
    TaskRunNotFound(TaskRunId),

    /// The store has no database behind it this run.
    #[error("memory.db is not open; persistence is disabled this run")]
    Disabled,

    /// A SQLite statement issued by `op` failed.
    #[error("{op}: {source}")]
    Sqlite {
        op: &'static str,
        #[source]
        source: BackendError,
    },
}

impl MemoryError {
    pub(crate) fn sqlite(op: &'static str) -> impl FnOnce(tokio_rusqlite::Error) -> Self {
        move |source| Self::Sqlite {
            op,
            source: source.into(),
        }
    }

    /// Like [`MemoryError::sqlite`], but a query that matched no row becomes `missing`.
    pub(crate) fn sqlite_or_missing(
        op: &'static str,
        missing: Self,
    ) -> impl FnOnce(tokio_rusqlite::Error) -> Self {
        move |source| match source {
            tokio_rusqlite::Error::Error(rusqlite::Error::QueryReturnedNoRows) => missing,
            source => Self::Sqlite {
                op,
                source: source.into(),
            },
        }
    }
}

/// Result alias for this crate's fallible operations.
pub type Result<T, E = MemoryError> = std::result::Result<T, E>;
