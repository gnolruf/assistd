use super::backoff::MAX_CONSECUTIVE_FAILURES;
use super::config::{ModelSpec, ServerSpec};
use super::error::LlamaServerError;
use super::supervisor::Supervisor;
use std::sync::{Arc, Mutex};
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
    /// `MAX_CONSECUTIVE_FAILURES` hit; the supervisor stopped restarting.
    Degraded,
}

/// Handle to the managed llama-server. Construct via [`LlamaService::start`];
/// drop order: call [`LlamaService::shutdown`] after the daemon's shutdown
/// watch has been flipped (the supervisor is already winding down by then).
pub struct LlamaService {
    task: Option<JoinHandle<()>>,
    ready_rx: watch::Receiver<ReadyState>,
    pid: Arc<Mutex<Option<u32>>>,
}

impl LlamaService {
    /// Spawns the supervisor and blocks until the child reports Ready or the
    /// supervisor gives up and enters Degraded.
    ///
    /// `shutdown_rx` is a subscriber on the daemon's shutdown watch. Flipping
    /// that watch externally causes the supervisor to tear down the child and
    /// exit; a call to `start` that sees shutdown before Ready returns
    /// `ShutdownDuringHealth`.
    pub async fn start(
        cfg: ServerSpec,
        model: ModelSpec,
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
                    // Supervisor task dropped its sender — most likely the
                    // daemon signaled shutdown before the child reached Ready.
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
                            // Supervisor is parked waiting for shutdown. We
                            // don't own the shutdown sender here, so abort
                            // the task rather than `await` it.
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

    pub fn is_ready(&self) -> bool {
        matches!(*self.ready_rx.borrow(), ReadyState::Ready)
    }

    pub fn state(&self) -> ReadyState {
        *self.ready_rx.borrow()
    }

    /// PID of the currently-running child, or `None` if no child is alive.
    /// The value changes as the supervisor restarts the child.
    pub fn pid(&self) -> Option<u32> {
        *self.pid.lock().expect("pid mutex poisoned")
    }

    /// Awaits the supervisor task. The caller is expected to have already
    /// flipped the shared shutdown watch; this just joins.
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
