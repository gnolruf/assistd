use super::backoff::MAX_CONSECUTIVE_FAILURES;
use super::error::EmbedServerError;
use super::supervisor::Supervisor;
use assistd_config::EmbeddingConfig;
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::watch;
use tokio::task::JoinHandle;

/// Lifecycle state broadcast by the supervisor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadyState {
    /// Child is spawning or waiting for its `/health` check to pass.
    Starting,
    /// Child responded `200 OK` to `/health` and is accepting requests.
    Ready,
    /// Child failed and the supervisor is waiting before the next restart attempt.
    BackingOff {
        /// 1-based count of consecutive failures so far.
        attempt: u32,
    },
    /// Too many consecutive failures; supervisor has parked and will not restart.
    Degraded,
}

/// Handle to the managed embed-server. On shutdown, flip the shutdown
/// watch first, then call [`EmbedService::shutdown`].
pub struct EmbedService {
    task: Option<JoinHandle<()>>,
    ready_rx: watch::Receiver<ReadyState>,
    pid: Arc<Mutex<Option<u32>>>,
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
        let pid = Arc::new(Mutex::new(None));

        let supervisor = Supervisor {
            cfg,
            ready_timeout,
            shutdown_rx,
            ready_tx,
            pid: pid.clone(),
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
                            return Ok(Self {
                                task: Some(task),
                                ready_rx,
                                pid,
                            });
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

    pub fn is_ready(&self) -> bool {
        matches!(*self.ready_rx.borrow(), ReadyState::Ready)
    }

    pub fn state(&self) -> ReadyState {
        *self.ready_rx.borrow()
    }

    pub fn pid(&self) -> Option<u32> {
        *self.pid.lock()
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
