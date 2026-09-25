use std::collections::VecDeque;
use std::ops::ControlFlow;
use std::process::ExitStatus;
use std::sync::Arc;
use std::time::{Duration, Instant};

use assistd_config::{LlamaServerConfig, ModelConfig};
use parking_lot::Mutex;
use tokio::sync::watch;
use tracing::{error, info, warn};

use super::backoff::{
    MAX_CONSECUTIVE_FAILURES, MAX_RESTARTS_PER_WINDOW, RESTART_WINDOW, backoff_delay,
};
use super::error::LlamaServerError;
use super::health::HealthChecker;
use super::process::ChildProcess;
use super::service::ReadyState;

/// Seconds a child must stay healthy after Ready for its exit to count as a
/// runtime crash (counter reset) rather than a startup flap.
pub(crate) const MIN_HEALTHY_SECONDS: u64 = 30;

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
    StartupError(LlamaServerError),
}

/// Drives the llama-server restart loop, broadcasting [`ReadyState`] transitions.
pub struct Supervisor {
    pub cfg: LlamaServerConfig,
    pub model: ModelConfig,
    pub shutdown_rx: watch::Receiver<bool>,
    pub ready_tx: watch::Sender<ReadyState>,
    pub pid: Arc<Mutex<Option<u32>>>,
}

impl Supervisor {
    /// Runs the supervisor loop until shutdown is requested. Once
    /// [`MAX_CONSECUTIVE_FAILURES`] is reached, or [`MAX_RESTARTS_PER_WINDOW`]
    /// restarts land inside [`RESTART_WINDOW`], it broadcasts
    /// [`ReadyState::Degraded`] and stops restarting until shutdown.
    pub async fn run(mut self) {
        let mut consecutive_failures: u32 = 0;
        let mut restart_history: VecDeque<Instant> = VecDeque::new();

        loop {
            if *self.shutdown_rx.borrow() {
                return;
            }
            let _ = self.ready_tx.send(ReadyState::Starting);

            let outcome = self.supervise_once().await;
            if !matches!(outcome, Ok(CycleResult::ShutdownRequested)) {
                record_restart(&mut restart_history, Instant::now());
            }
            let Some(failures) = failures_after(outcome, consecutive_failures) else {
                return;
            };
            consecutive_failures = failures;

            if restart_budget_exhausted(consecutive_failures, restart_history.len()) {
                let _ = self.ready_tx.send(ReadyState::Degraded);
                let _ = self.shutdown_rx.wait_for(|v| *v).await;
                return;
            }

            if consecutive_failures == 0 {
                continue;
            }
            if self.back_off(consecutive_failures).await.is_break() {
                return;
            }
        }
    }

    /// Sleep out the backoff for `attempt`; breaks if shutdown arrives first.
    async fn back_off(&mut self, attempt: u32) -> ControlFlow<()> {
        let delay = backoff_delay(attempt - 1);
        warn!(
            target: "assistd::llama_server",
            "restarting llama-server in {delay:?} (attempt {attempt}/{MAX_CONSECUTIVE_FAILURES})"
        );
        let _ = self.ready_tx.send(ReadyState::BackingOff { attempt });

        tokio::select! {
            _ = tokio::time::sleep(delay) => ControlFlow::Continue(()),
            _ = self.shutdown_rx.changed() => {
                info!(target: "assistd::llama_server", "supervisor shutdown during backoff");
                ControlFlow::Break(())
            }
        }
    }

    async fn supervise_once(&mut self) -> Result<CycleResult, LlamaServerError> {
        let mut child = ChildProcess::spawn(&self.cfg, &self.model)?;
        *self.pid.lock() = child.pid();
        let ready_timeout = Duration::from_secs(self.cfg.ready_timeout_secs.get());
        let health = HealthChecker::new(
            &self.cfg.host.to_string(),
            self.cfg.port.get(),
            ready_timeout,
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
        info!(target: "assistd::llama_server", "llama-server ready");

        let result = tokio::select! {
            exit = child.wait() => match exit {
                Ok(status) => Ok(CycleResult::CrashedAfterReady {
                    status,
                    ran_for: ready_at.elapsed(),
                }),
                Err(e) => Err(LlamaServerError::Io(e)),
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
            Err(LlamaServerError::ShutdownDuringHealth) => StartupOutcome::ShuttingDown,
            Err(e) => StartupOutcome::StartupError(e),
        },
        exit = child.wait() => match exit {
            Ok(status) => StartupOutcome::ChildExited(status),
            Err(e) => StartupOutcome::StartupError(LlamaServerError::Io(e)),
        }
    }
}

/// Push `now` onto the rolling restart window, evicting entries older than
/// [`RESTART_WINDOW`].
fn record_restart(history: &mut VecDeque<Instant>, now: Instant) {
    while let Some(&oldest) = history.front() {
        if now.duration_since(oldest) > RESTART_WINDOW {
            history.pop_front();
        } else {
            break;
        }
    }
    history.push_back(now);
}

/// Log a finished cycle and return the updated consecutive-failure count, or
/// `None` when the cycle ended because shutdown was requested.
fn failures_after(
    outcome: Result<CycleResult, LlamaServerError>,
    consecutive_failures: u32,
) -> Option<u32> {
    match outcome {
        Ok(CycleResult::ShutdownRequested) => {
            info!(target: "assistd::llama_server", "supervisor shutdown");
            None
        }
        Ok(CycleResult::CrashedAfterReady { status, ran_for }) => {
            warn!(
                target: "assistd::llama_server",
                "llama-server exited after {ran_for:?} post-ready: {status}; restarting"
            );
            if ran_for >= Duration::from_secs(MIN_HEALTHY_SECONDS) {
                Some(0)
            } else {
                Some(consecutive_failures + 1)
            }
        }
        Ok(CycleResult::FailedToStart { status }) => {
            error!(
                target: "assistd::llama_server",
                "llama-server exited before reaching ready: {status}"
            );
            Some(consecutive_failures + 1)
        }
        Err(e) => {
            error!(
                target: "assistd::llama_server",
                "llama-server startup failed: {e}"
            );
            Some(consecutive_failures + 1)
        }
    }
}

/// Whether either restart limit has tripped, logging which one did.
fn restart_budget_exhausted(consecutive_failures: u32, restarts_in_window: usize) -> bool {
    let window_tripped = restarts_in_window >= MAX_RESTARTS_PER_WINDOW;
    if window_tripped {
        error!(
            target: "assistd::llama_server",
            restarts = restarts_in_window,
            window_secs = RESTART_WINDOW.as_secs(),
            "llama-server hit {MAX_RESTARTS_PER_WINDOW} restarts in rolling window; entering degraded state"
        );
        return true;
    }
    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
        error!(
            target: "assistd::llama_server",
            "{MAX_CONSECUTIVE_FAILURES} consecutive failures; entering degraded state"
        );
        return true;
    }
    false
}
