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
use crate::backoff::{
    MAX_CONSECUTIVE_FAILURES, MAX_RESTARTS_PER_WINDOW, RESTART_WINDOW, RestartDecision,
    RestartPolicy,
};

/// Graceful shutdown budget per child; fits under systemd's 15s stop budget.
const TERM_TIMEOUT: Duration = Duration::from_secs(10);

/// What happened during one supervisor cycle.
enum CycleResult {
    ShutdownRequested,
    /// Child exited before `/health` ever returned 200.
    FailedToStart {
        status: ExitStatus,
    },
    CrashedAfterReady {
        status: ExitStatus,
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

    /// Run until shutdown is requested. Once a restart cap trips it broadcasts
    /// [`ReadyState::Degraded`] and only waits for shutdown.
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

            match policy.next_restart(Instant::now()) {
                RestartDecision::Backoff { failures: 0, .. } => {}
                RestartDecision::Backoff { delay, failures } => {
                    if self.back_off(delay, failures).await.is_break() {
                        return;
                    }
                }
                RestartDecision::ConsecutiveCapReached { .. } => {
                    error!(
                        target: "assistd::child_server",
                        server = self.spec.name(),
                        "{MAX_CONSECUTIVE_FAILURES} consecutive failures; entering degraded state"
                    );
                    self.park_degraded().await;
                    return;
                }
                RestartDecision::WindowCapReached { restarts } => {
                    error!(
                        target: "assistd::child_server",
                        server = self.spec.name(),
                        restarts,
                        window_secs = RESTART_WINDOW.as_secs(),
                        "{MAX_RESTARTS_PER_WINDOW} restarts in rolling window; entering degraded state"
                    );
                    self.park_degraded().await;
                    return;
                }
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

    async fn park_degraded(&mut self) {
        let _ = self.ready_tx.send(ReadyState::Degraded);
        let _ = self.shutdown_rx.wait_for(|v| *v).await;
    }

    /// Sleep out `delay` before restart `attempt`; breaks if shutdown arrives first.
    async fn back_off(&mut self, delay: Duration, attempt: u32) -> ControlFlow<()> {
        warn!(
            target: "assistd::child_server",
            server = self.spec.name(),
            "restarting {} in {delay:?} (attempt {attempt}/{MAX_CONSECUTIVE_FAILURES})",
            self.spec.name(),
        );
        let _ = self.ready_tx.send(ReadyState::BackingOff { attempt });

        tokio::select! {
            () = tokio::time::sleep(delay) => ControlFlow::Continue(()),
            _ = self.shutdown_rx.changed() => {
                info!(
                    target: "assistd::child_server",
                    server = self.spec.name(),
                    "supervisor shutdown during backoff"
                );
                ControlFlow::Break(())
            }
        }
    }

    async fn supervise_once(&mut self) -> Result<CycleResult, ChildServerError> {
        let mut child = ChildProcess::spawn(&self.spec)?;
        *self.pid.lock() = child.pid();
        let health = HealthChecker::new(
            self.spec.name(),
            self.spec.listen_addr(),
            child.process_group(),
            self.spec.ready_timeout(),
        )?;

        match wait_for_startup(&mut child, &health, &mut self.shutdown_rx).await {
            StartupOutcome::Ready => {}
            StartupOutcome::ChildExited(status) => {
                *self.pid.lock() = None;
                return Ok(CycleResult::FailedToStart { status });
            }
            StartupOutcome::ShuttingDown => {
                child.shutdown(TERM_TIMEOUT).await?;
                *self.pid.lock() = None;
                return Ok(CycleResult::ShutdownRequested);
            }
            StartupOutcome::StartupError(e) => {
                child.shutdown(TERM_TIMEOUT).await?;
                *self.pid.lock() = None;
                return Err(e);
            }
        }

        let ready_at = Instant::now();
        let _ = self.ready_tx.send(ReadyState::Ready);
        info!(target: "assistd::child_server", server = self.spec.name(), "{} ready", self.spec.name());

        let result = tokio::select! {
            exit = child.wait() => match exit {
                Ok(status) => Ok(CycleResult::CrashedAfterReady {
                    status,
                    ran_for: ready_at.elapsed(),
                }),
                Err(e) => Err(ChildServerError::Io(e)),
            },
            _ = self.shutdown_rx.changed() => {
                child.shutdown(TERM_TIMEOUT).await?;
                Ok(CycleResult::ShutdownRequested)
            }
        };
        *self.pid.lock() = None;
        result
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
