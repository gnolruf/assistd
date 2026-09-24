use super::backoff::MAX_CONSECUTIVE_FAILURES;
use super::error::LlamaServerError;
use super::supervisor::Supervisor;
use assistd_config::{LlamaServerConfig, ModelConfig};
use parking_lot::Mutex;
use std::sync::Arc;
use tokio::sync::watch;
use tokio::task::JoinHandle;

/// State broadcast by the supervisor as the child moves through its lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadyState {
    /// A spawn attempt is in progress but the child has not yet reported ready.
    Starting,
    /// The child has reported 200 OK on `/health`.
    Ready,
    /// The last spawn attempt failed; a restart is scheduled after `attempt`
    /// consecutive failures.
    BackingOff { attempt: u32 },
    /// A restart limit was hit ([`MAX_CONSECUTIVE_FAILURES`] or the
    /// rolling-window cap); the supervisor stopped restarting.
    Degraded,
}

/// Handle to the managed llama-server, constructed via
/// [`LlamaService::start`]. Dropping it aborts the supervisor task;
/// [`LlamaService::shutdown`] joins it instead.
pub struct LlamaService {
    task: Option<JoinHandle<()>>,
    ready_rx: watch::Receiver<ReadyState>,
    pid: Arc<Mutex<Option<u32>>>,
}

impl LlamaService {
    /// Spawns the supervisor and waits until the child reports Ready.
    ///
    /// Flipping `shutdown_rx` makes the supervisor tear down the child and
    /// exit. Returns [`LlamaServerError::ShutdownDuringHealth`] if that
    /// happens before Ready, and [`LlamaServerError::StartupFailed`] if the
    /// supervisor gives up and enters Degraded.
    #[tracing::instrument(skip(cfg, model, shutdown_rx), fields(host = %cfg.host, port = cfg.port))]
    pub async fn start(
        cfg: LlamaServerConfig,
        model: ModelConfig,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Result<Self, LlamaServerError> {
        let (ready_tx, mut ready_rx) = watch::channel(ReadyState::Starting);
        let pid = Arc::new(Mutex::new(None));

        let supervisor = Supervisor {
            cfg,
            model,
            shutdown_rx,
            ready_tx,
            pid: pid.clone(),
        };
        let task = tokio::spawn(async move { supervisor.run().await });

        loop {
            match ready_rx.changed().await {
                Err(_) => {
                    // The supervisor exited and dropped its sender, which
                    // before Ready normally means shutdown was signaled.
                    let _ = task.await;
                    return Err(LlamaServerError::ShutdownDuringHealth);
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
                            return Err(LlamaServerError::StartupFailed {
                                attempts: MAX_CONSECUTIVE_FAILURES,
                            });
                        }
                        ReadyState::Starting | ReadyState::BackingOff { .. } => continue,
                    }
                }
            }
        }
    }

    /// Returns `true` iff the supervisor is currently in [`ReadyState::Ready`].
    pub fn is_ready(&self) -> bool {
        matches!(*self.ready_rx.borrow(), ReadyState::Ready)
    }

    /// Snapshot of the current [`ReadyState`].
    pub fn state(&self) -> ReadyState {
        *self.ready_rx.borrow()
    }

    /// Subscribe to the supervisor's readiness watch, to await a
    /// transition (e.g. crash → restart → Ready) rather than snapshot the
    /// current state.
    pub fn subscribe_ready(&self) -> watch::Receiver<ReadyState> {
        self.ready_rx.clone()
    }

    /// PID of the currently-running child, or `None` if no child is alive.
    /// The value changes as the supervisor restarts the child.
    pub fn pid(&self) -> Option<u32> {
        *self.pid.lock()
    }

    /// Joins the supervisor task. The shutdown watch passed to
    /// [`Self::start`] must already be flipped; this does not signal it.
    /// Errors with [`LlamaServerError::SupervisorPanic`] if the task panicked.
    pub async fn shutdown(mut self) -> Result<(), LlamaServerError> {
        if let Some(task) = self.task.take() {
            task.await.map_err(|_| LlamaServerError::SupervisorPanic)?;
        }
        Ok(())
    }
}

impl Drop for LlamaService {
    fn drop(&mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}
