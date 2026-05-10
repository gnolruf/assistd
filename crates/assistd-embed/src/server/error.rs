use std::process::ExitStatus;
use std::time::Duration;
use thiserror::Error;

/// Errors produced by the embed-server supervisor and its supporting components.
#[derive(Debug, Error)]
pub enum EmbedServerError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("failed to spawn embed-server binary {path}: {source}")]
    Spawn {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("embed-server exited before reaching ready: {status}")]
    ExitedBeforeReady { status: ExitStatus },

    #[error("embed-server did not become ready within {timeout:?}")]
    HealthTimeout { timeout: Duration },

    #[error("health check aborted due to shutdown")]
    ShutdownDuringHealth,

    #[error("HTTP client error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("embed-server startup failed after {attempts} attempts")]
    StartupFailed { attempts: u32 },

    #[error("supervisor task panicked")]
    SupervisorPanic,

    #[error("dim probe failed: server returned no embedding vectors")]
    DimProbeEmpty,
}
