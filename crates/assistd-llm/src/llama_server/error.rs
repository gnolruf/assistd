use std::process::ExitStatus;
use std::time::Duration;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LlamaServerError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("failed to spawn llama-server binary {path}: {source}")]
    Spawn {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("llama-server exited before reaching ready: {status}")]
    ExitedBeforeReady { status: ExitStatus },

    #[error("llama-server did not become ready within {timeout:?}")]
    HealthTimeout { timeout: Duration },

    #[error("health check aborted due to shutdown")]
    ShutdownDuringHealth,

    #[error("HTTP client error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("llama-server startup failed after {attempts} attempts")]
    StartupFailed { attempts: u32 },

    #[error("supervisor task panicked")]
    SupervisorPanic,
}
