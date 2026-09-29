use std::sync::Arc;

use parking_lot::Mutex;
use tokio::sync::watch;
use tokio::task::JoinHandle;

use super::ChildServerSpec;
use super::error::ChildServerError;
use super::supervisor::Supervisor;
use crate::backoff::MAX_CONSECUTIVE_FAILURES;

/// State broadcast by the supervisor as the child moves through its lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadyState {
    /// A spawn attempt is in progress but the child has not yet passed `/health`.
    Starting,
    /// The child has reported 200 OK on `/health`.
    Ready,
    /// The last spawn attempt failed; a restart is scheduled after `attempt`
    /// consecutive failures.
    BackingOff { attempt: u32 },
    /// A restart limit was hit; the supervisor stopped restarting.
    Degraded,
}

/// Handle to a supervised child server, constructed via [`ChildServer::start`].
/// Dropping it aborts the supervisor task; [`ChildServer::shutdown`] joins it.
#[derive(Debug)]
pub struct ChildServer {
    server: &'static str,
    task: Option<JoinHandle<()>>,
    ready_rx: watch::Receiver<ReadyState>,
    pid: Arc<Mutex<Option<u32>>>,
}

impl ChildServer {
    /// Spawn the supervisor and wait until the child reports `Ready`. Errors
    /// with [`ChildServerError::ShutdownDuringHealth`] if the supervisor exits
    /// first, or [`ChildServerError::StartupFailed`] once it goes `Degraded`.
    #[tracing::instrument(skip(spec, shutdown_rx), fields(server = spec.name(), addr = %spec.listen_addr()))]
    pub async fn start<S: ChildServerSpec>(
        spec: S,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Result<Self, ChildServerError> {
        let server = spec.name();
        let (ready_tx, mut ready_rx) = watch::channel(ReadyState::Starting);
        let pid = Arc::new(Mutex::new(None));
        let supervisor = Supervisor::new(spec, shutdown_rx, ready_tx, pid.clone());
        let task = tokio::spawn(supervisor.run());

        loop {
            match ready_rx.changed().await {
                Err(_) => {
                    let _ = task.await;
                    return Err(ChildServerError::ShutdownDuringHealth);
                }
                Ok(()) => {
                    let state = *ready_rx.borrow();
                    match state {
                        ReadyState::Ready => {
                            return Ok(Self {
                                server,
                                task: Some(task),
                                ready_rx,
                                pid,
                            });
                        }
                        ReadyState::Degraded => {
                            task.abort();
                            return Err(ChildServerError::StartupFailed {
                                server,
                                attempts: MAX_CONSECUTIVE_FAILURES,
                            });
                        }
                        ReadyState::Starting | ReadyState::BackingOff { .. } => continue,
                    }
                }
            }
        }
    }

    /// The name given by the spec this server was started from.
    pub fn name(&self) -> &'static str {
        self.server
    }

    /// Whether the supervisor is currently in [`ReadyState::Ready`].
    pub fn is_ready(&self) -> bool {
        matches!(*self.ready_rx.borrow(), ReadyState::Ready)
    }

    /// Snapshot of the current [`ReadyState`].
    pub fn state(&self) -> ReadyState {
        *self.ready_rx.borrow()
    }

    /// Subscribe to the supervisor's readiness transitions.
    pub fn subscribe_ready(&self) -> watch::Receiver<ReadyState> {
        self.ready_rx.clone()
    }

    /// PID of the currently running child, or `None` if no child is alive.
    pub fn pid(&self) -> Option<u32> {
        *self.pid.lock()
    }

    /// Join the supervisor task. The shutdown watch passed to [`Self::start`]
    /// must already be flipped; this does not signal it. Errors with
    /// [`ChildServerError::SupervisorPanic`] if the task panicked.
    pub async fn shutdown(mut self) -> Result<(), ChildServerError> {
        if let Some(task) = self.task.take() {
            task.await.map_err(|_| ChildServerError::SupervisorPanic)?;
        }
        Ok(())
    }
}

impl Drop for ChildServer {
    fn drop(&mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}
