use std::time::Duration;

use tokio::sync::watch;
use tokio::task::JoinHandle;

use assistd_config::EmbeddingConfig;

use super::backoff::MAX_CONSECUTIVE_FAILURES;
use super::error::EmbedServerError;
use super::supervisor::Supervisor;

/// Lifecycle state broadcast by the supervisor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadyState {
    /// Spawning, or waiting for `/health` to pass.
    Starting,
    /// `/health` returned 200; accepting requests.
    Ready,
    /// Waiting before the next restart.
    BackingOff {
        /// 1-based count of consecutive failures.
        attempt: u32,
    },
    /// Too many consecutive failures; no further restarts.
    Degraded,
}

/// Handle to the managed embed-server. On shutdown, flip the shutdown
/// watch first, then call [`EmbedService::shutdown`].
pub struct EmbedService {
    task: Option<JoinHandle<()>>,
}

impl EmbedService {
    /// Spawn the supervisor and wait until the child is `Ready`, or fail
    /// once it goes `Degraded`. `ready_timeout` caps each health wait.
    #[tracing::instrument(skip(cfg, shutdown_rx), fields(host = %cfg.host, port = cfg.port))]
    pub async fn start(
        cfg: EmbeddingConfig,
        ready_timeout: Duration,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Result<Self, EmbedServerError> {
        let (ready_tx, mut ready_rx) = watch::channel(ReadyState::Starting);
        let supervisor = Supervisor {
            cfg,
            ready_timeout,
            shutdown_rx,
            ready_tx,
        };
        let task = tokio::spawn(async move { supervisor.run().await });

        loop {
            match ready_rx.changed().await {
                Err(_) => {
                    let _ = task.await;
                    return Err(EmbedServerError::ShutdownDuringHealth);
                }
                Ok(()) => {
                    let state = *ready_rx.borrow();
                    match state {
                        ReadyState::Ready => {
                            return Ok(Self { task: Some(task) });
                        }
                        ReadyState::Degraded => {
                            task.abort();
                            return Err(EmbedServerError::StartupFailed {
                                attempts: MAX_CONSECUTIVE_FAILURES,
                            });
                        }
                        ReadyState::Starting | ReadyState::BackingOff { .. } => continue,
                    }
                }
            }
        }
    }

    /// Join the supervisor task; the shutdown watch must already be set.
    pub async fn shutdown(mut self) -> Result<(), EmbedServerError> {
        if let Some(task) = self.task.take() {
            task.await.map_err(|_| EmbedServerError::SupervisorPanic)?;
        }
        Ok(())
    }
}

impl Drop for EmbedService {
    fn drop(&mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}
