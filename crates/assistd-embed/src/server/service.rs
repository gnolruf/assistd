use super::backoff::MAX_CONSECUTIVE_FAILURES;
use super::error::EmbedServerError;
use super::supervisor::Supervisor;
use assistd_config::EmbeddingConfig;
use std::sync::{Arc, Mutex};
use tokio::sync::watch;
use tokio::task::JoinHandle;

/// State broadcast by the supervisor as the embed child moves through
/// its lifecycle. Mirrors `assistd_llm::llama_server::ReadyState` so the
/// daemon's startup messages read consistently across both subsystems.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadyState {
    Starting,
    Ready,
    BackingOff { attempt: u32 },
    Degraded,
}

/// Handle to the managed embed-server. Construct via [`EmbedService::start`];
/// drop order at daemon shutdown: flip the daemon-wide shutdown watch,
/// then call [`EmbedService::shutdown`] to await the supervisor task.
pub struct EmbedService {
    task: Option<JoinHandle<()>>,
    ready_rx: watch::Receiver<ReadyState>,
    pid: Arc<Mutex<Option<u32>>>,
}

impl EmbedService {
    /// Spawn the supervisor and block until the child reports `Ready` or
    /// the supervisor enters `Degraded`. Errors are surfaced via
    /// [`EmbedServerError`].
    #[tracing::instrument(skip(cfg, shutdown_rx), fields(host = %cfg.host, port = cfg.port))]
    pub async fn start(
        cfg: EmbeddingConfig,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Result<Self, EmbedServerError> {
        let (ready_tx, mut ready_rx) = watch::channel(ReadyState::Starting);
        let pid = Arc::new(Mutex::new(None));

        let supervisor = Supervisor {
            cfg,
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
        *self.pid.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Await the supervisor task. Caller is expected to have flipped the
    /// shared shutdown watch already; this simply joins.
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
