//! The supervisor loop: spawn, wait for health, and restart under the restart policy.

use std::ops::ControlFlow;
use std::process::ExitStatus;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use tokio::sync::watch;
use tracing::{error, info, warn};

use super::ChildServerSpec;
use super::error::ChildServerError;
use super::health::HealthChecker;
use super::process::ChildProcess;
use super::service::ReadyState;
use crate::backoff::{MAX_CONSECUTIVE_FAILURES, RestartDecision, RestartPolicy};

/// How long a stopping child gets to exit after SIGTERM before its process
/// group gets SIGKILL.
const TERM_TIMEOUT: Duration = Duration::from_secs(5);

/// What happened during one supervisor cycle.
enum CycleResult {
    ShutdownRequested,
    /// Child exited before its health check ever returned 200.
    FailedToStart {
        status: ExitStatus,
    },
    CrashedAfterReady {
        status: ExitStatus,
        ran_for: Duration,
    },
    /// Child stopped answering its health check while ready and was killed.
    HungAfterReady {
        ran_for: Duration,
    },
}

/// How a freshly spawned child's startup ended.
enum StartupOutcome {
    Ready,
    ChildExited(ExitStatus),
    ShuttingDown,
    StartupError(ChildServerError),
}

/// Drives the restart loop for one child, broadcasting [`ReadyState`] transitions.
pub(super) struct Supervisor<S> {
    spec: S,
    shutdown_rx: watch::Receiver<bool>,
    ready_tx: watch::Sender<ReadyState>,
    pid: Arc<Mutex<Option<u32>>>,
}

impl<S: ChildServerSpec> Supervisor<S> {
    pub(super) fn new(
        spec: S,
        shutdown_rx: watch::Receiver<bool>,
        ready_tx: watch::Sender<ReadyState>,
        pid: Arc<Mutex<Option<u32>>>,
    ) -> Self {
        Self {
            spec,
            shutdown_rx,
            ready_tx,
            pid,
        }
    }

    /// Run until shutdown is requested. While a restart cap is tripped it broadcasts
    /// [`ReadyState::Degraded`] between restarts and makes them at the slow cadence.
    pub(super) async fn run(mut self) {
        let mut policy = RestartPolicy::default();

        loop {
            if *self.shutdown_rx.borrow() {
                return;
            }
            let _ = self.ready_tx.send(ReadyState::Starting);

            let outcome = self.supervise_once().await;
            if self.record_outcome(outcome, &mut policy).is_break() {
                return;
            }

            let decision = policy.next_restart(Instant::now());
            if self.wait_to_restart(decision).await.is_break() {
                return;
            }
        }
    }

    /// Log a finished cycle and feed it to `policy`; `Break` when the cycle
    /// ended because shutdown was requested.
    fn record_outcome(
        &self,
        outcome: Result<CycleResult, ChildServerError>,
        policy: &mut RestartPolicy,
    ) -> ControlFlow<()> {
        let server = self.spec.name();
        match outcome {
            Ok(CycleResult::ShutdownRequested) => {
                info!(target: "assistd::child_server", server, "supervisor shutdown");
                return ControlFlow::Break(());
            }
            Ok(CycleResult::CrashedAfterReady { status, ran_for }) => {
                warn!(
                    target: "assistd::child_server",
                    server,
                    "{server} exited after {ran_for:?} post-ready: {status}; restarting"
                );
                policy.record_session_end(ran_for);
            }
            Ok(CycleResult::HungAfterReady { ran_for }) => {
                warn!(
                    target: "assistd::child_server",
                    server,
                    "{server} stopped answering health checks after {ran_for:?} post-ready; restarting"
                );
                policy.record_session_end(ran_for);
            }
            Ok(CycleResult::FailedToStart { status }) => {
                error!(
                    target: "assistd::child_server",
                    server,
                    "{server} exited before reaching ready: {status}"
                );
                policy.record_spawn_failure();
            }
            Err(e) => {
                error!(target: "assistd::child_server", server, "{server} startup failed: {e}");
                policy.record_spawn_failure();
            }
        }
        ControlFlow::Continue(())
    }

    /// Broadcast the wait `decision` calls for and sleep it out; breaks if
    /// shutdown arrives first.
    async fn wait_to_restart(&mut self, decision: RestartDecision) -> ControlFlow<()> {
        let server = self.spec.name();
        match decision {
            RestartDecision::Backoff { failures: 0, .. } => return ControlFlow::Continue(()),
            RestartDecision::Backoff { delay, failures } => {
                warn!(
                    target: "assistd::child_server",
                    server,
                    "restarting {server} in {delay:?} (attempt {failures}/{MAX_CONSECUTIVE_FAILURES})"
                );
                let _ = self
                    .ready_tx
                    .send(ReadyState::BackingOff { attempt: failures });
            }
            RestartDecision::SlowRetry { delay, cap } => {
                error!(
                    target: "assistd::child_server",
                    server,
                    retry_secs = delay.as_secs(),
                    "{server} {cap}; entering degraded state and slowing restarts"
                );
                let _ = self.ready_tx.send(ReadyState::Degraded);
            }
        }

        tokio::select! {
            () = tokio::time::sleep(decision.delay()) => ControlFlow::Continue(()),
            _ = self.shutdown_rx.changed() => {
                info!(
                    target: "assistd::child_server",
                    server,
                    "supervisor shutdown while waiting to restart"
                );
                ControlFlow::Break(())
            }
        }
    }

    async fn supervise_once(&mut self) -> Result<CycleResult, ChildServerError> {
        let child = ChildProcess::spawn(&self.spec)?;
        *self.pid.lock() = child.pid();
        let result = self.supervise_child(child).await;
        *self.pid.lock() = None;
        result
    }

    async fn supervise_child(
        &mut self,
        mut child: ChildProcess,
    ) -> Result<CycleResult, ChildServerError> {
        let health = HealthChecker::new(&self.spec, child.process_group())?;

        match wait_for_startup(&mut child, &health, &mut self.shutdown_rx).await {
            StartupOutcome::Ready => {}
            StartupOutcome::ChildExited(status) => {
                return Ok(CycleResult::FailedToStart { status });
            }
            StartupOutcome::ShuttingDown => {
                stop_child(child).await;
                return Ok(CycleResult::ShutdownRequested);
            }
            StartupOutcome::StartupError(e) => {
                stop_child(child).await;
                return Err(e);
            }
        }

        let ready_at = Instant::now();
        let _ = self.ready_tx.send(ReadyState::Ready);
        info!(target: "assistd::child_server", server = self.spec.name(), "{} ready", self.spec.name());

        tokio::select! {
            exit = child.wait() => {
                let status = exit?;
                Ok(CycleResult::CrashedAfterReady {
                    status,
                    ran_for: ready_at.elapsed(),
                })
            }
            () = health.wait_hung() => {
                let ran_for = ready_at.elapsed();
                stop_child(child).await;
                Ok(CycleResult::HungAfterReady { ran_for })
            }
            _ = self.shutdown_rx.changed() => {
                stop_child(child).await;
                Ok(CycleResult::ShutdownRequested)
            }
        }
    }
}

/// Stop `child`, logging rather than returning a failure to reap it so the
/// cycle's own result survives.
async fn stop_child(child: ChildProcess) {
    if let Err(e) = child.shutdown(TERM_TIMEOUT).await {
        warn!(target: "assistd::child_server", "stopping child failed: {e}");
    }
}

async fn wait_for_startup(
    child: &mut ChildProcess,
    health: &HealthChecker,
    shutdown_rx: &mut watch::Receiver<bool>,
) -> StartupOutcome {
    tokio::select! {
        res = health.wait_ready(shutdown_rx) => match res {
            Ok(()) => StartupOutcome::Ready,
            Err(ChildServerError::ShutdownDuringHealth) => StartupOutcome::ShuttingDown,
            Err(e) => StartupOutcome::StartupError(e),
        },
        exit = child.wait() => match exit {
            Ok(status) => StartupOutcome::ChildExited(status),
            Err(e) => StartupOutcome::StartupError(ChildServerError::Io(e)),
        }
    }
}
