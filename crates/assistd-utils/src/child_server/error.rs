use std::path::PathBuf;
use std::time::Duration;

use thiserror::Error;

/// Failures of a child server's supervisor and process.
#[derive(Debug, Error)]
pub enum ChildServerError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("failed to spawn {server} binary {}: {source}", path.display())]
    Spawn {
        server: &'static str,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("{server} did not become ready within {timeout:?}")]
    HealthTimeout {
        server: &'static str,
        timeout: Duration,
    },

    #[error("health check aborted due to shutdown")]
    ShutdownDuringHealth,

    #[error("HTTP client error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("{server} startup failed after {attempts} attempts")]
    StartupFailed { server: &'static str, attempts: u32 },

    #[error("supervisor task panicked")]
    SupervisorPanic,
}
