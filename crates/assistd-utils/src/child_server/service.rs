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
    status: ChildServerStatus,
}

impl ChildServer {
    /// Spawn the supervisor and wait until the child reports `Ready`. Errors
    /// with [`ChildServerError::ShutdownDuringHealth`] if the supervisor exits
    /// first, or [`ChildServerError::StartupFailed`] once it goes `Degraded`.
    /// Dropping the returned future aborts the supervisor.
    #[tracing::instrument(skip(spec, shutdown_rx), fields(server = spec.name(), addr = %spec.listen_addr()))]
    pub async fn start<S: ChildServerSpec>(
        spec: S,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Result<Self, ChildServerError> {
        let server = spec.name();
        let (ready_tx, ready_rx) = watch::channel(ReadyState::Starting);
        let pid = Arc::new(Mutex::new(None));
        let supervisor = Supervisor::new(spec, shutdown_rx, ready_tx, pid.clone());
        let mut child_server = Self {
            server,
            task: Some(tokio::spawn(supervisor.run())),
            status: ChildServerStatus { ready_rx, pid },
        };
        child_server.await_ready().await?;
        Ok(child_server)
    }

    async fn await_ready(&mut self) -> Result<(), ChildServerError> {
        loop {
            if self.status.ready_rx.changed().await.is_err() {
                if let Some(task) = self.task.take() {
                    let _ = task.await;
                }
                return Err(ChildServerError::ShutdownDuringHealth);
            }
            match *self.status.ready_rx.borrow() {
                ReadyState::Ready => return Ok(()),
                ReadyState::Degraded => {
                    return Err(ChildServerError::StartupFailed {
                        server: self.server,
                        attempts: MAX_CONSECUTIVE_FAILURES,
                    });
                }
                ReadyState::Starting | ReadyState::BackingOff { .. } => {}
            }
        }
    }

    /// The name given by the spec this server was started from.
    pub fn name(&self) -> &'static str {
        self.server
    }

    /// Whether the supervisor is currently in [`ReadyState::Ready`].
    pub fn is_ready(&self) -> bool {
        self.status.is_ready()
    }

    /// Whether the supervisor is [`ReadyState::Ready`] and its child is alive,
    /// so the listener that passed the readiness check is still the child's.
    pub fn is_serving(&self) -> bool {
        self.status.is_serving()
    }

    /// Snapshot of the current [`ReadyState`].
    pub fn state(&self) -> ReadyState {
        *self.status.ready_rx.borrow()
    }

    /// Subscribe to the supervisor's readiness transitions.
    pub fn subscribe_ready(&self) -> watch::Receiver<ReadyState> {
        self.status.ready_rx.clone()
    }

    /// PID of the currently running child, or `None` if no child is alive.
    pub fn pid(&self) -> Option<u32> {
        self.status.pid()
    }

    /// A cloneable view of this server's readiness, for clients that must
    /// check it before every request.
    pub fn status(&self) -> ChildServerStatus {
        self.status.clone()
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

/// Read-only view of a [`ChildServer`]'s readiness and child pid. It never
/// reports serving once the supervisor has stopped.
#[derive(Debug, Clone)]
pub struct ChildServerStatus {
    ready_rx: watch::Receiver<ReadyState>,
    pid: Arc<Mutex<Option<u32>>>,
}

impl ChildServerStatus {
    /// A status driven by the returned sender, reporting `pid` as the child's.
    /// Dropping the sender reads as a stopped supervisor.
    #[cfg(feature = "test-support")]
    pub fn scripted(state: ReadyState, pid: Option<u32>) -> (watch::Sender<ReadyState>, Self) {
        let (ready_tx, ready_rx) = watch::channel(state);
        let status = Self {
            ready_rx,
            pid: Arc::new(Mutex::new(pid)),
        };
        (ready_tx, status)
    }

    /// Whether the supervisor is running, [`ReadyState::Ready`], and its child
    /// is alive, so the listener that passed the readiness check is still the child's.
    pub fn is_serving(&self) -> bool {
        let supervisor_running = self.ready_rx.has_changed().is_ok();
        supervisor_running && self.is_ready() && self.pid().is_some()
    }

    fn is_ready(&self) -> bool {
        matches!(*self.ready_rx.borrow(), ReadyState::Ready)
    }

    fn pid(&self) -> Option<u32> {
        *self.pid.lock()
    }
}
