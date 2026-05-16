use super::backoff::{
    MAX_CONSECUTIVE_FAILURES, MAX_RESTARTS_PER_WINDOW, RESTART_WINDOW, backoff_delay,
};
use super::error::LlamaServerError;
use super::health::HealthChecker;
use super::process::ChildProcess;
use super::service::ReadyState;
use assistd_config::{LlamaServerConfig, ModelConfig};
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::process::ExitStatus;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::watch;
use tracing::{error, info, warn};

/// Seconds a child must stay healthy after reaching Ready before a subsequent
/// exit is treated as a runtime crash (counter reset) rather than a startup
/// flap (counter increment).
pub(crate) const MIN_HEALTHY_SECONDS: u64 = 30;

/// Graceful shutdown budget per child; fits under systemd's 15s stop budget.
const TERM_TIMEOUT: Duration = Duration::from_secs(10);

/// What happened during one supervisor cycle.
enum CycleResult {
    /// The shutdown watch flipped; the cycle tore down cleanly.
    ShutdownRequested,
    /// Child exited before `/health` ever returned 200.
    FailedToStart { status: ExitStatus },
    /// Child reached Ready, ran for `ran_for`, then exited.
    CrashedAfterReady {
        status: ExitStatus,
        ran_for: Duration,
    },
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
    /// Runs the supervisor loop until shutdown is requested, [`MAX_CONSECUTIVE_FAILURES`]
    /// is reached, or [`MAX_RESTARTS_PER_WINDOW`] is exceeded inside
    /// [`RESTART_WINDOW`]. Intended to be spawned as a Tokio task.
    pub async fn run(mut self) {
        let mut consecutive_failures: u32 = 0;
        // Rolling window of restart timestamps. Catches the crash-loop case
        // where each child runs for ≥ MIN_HEALTHY_SECONDS (resetting
        // consecutive_failures) but the process keeps dying in a short
        // wall-clock window.
        let mut restart_history: VecDeque<Instant> = VecDeque::new();

        loop {
            if *self.shutdown_rx.borrow() {
                return;
            }
            let _ = self.ready_tx.send(ReadyState::Starting);

            let outcome = self.supervise_once().await;

            // Record any non-shutdown outcome in the rolling window before
            // updating consecutive_failures, so the cap is enforced even when
            // the per-cycle counter keeps resetting.
            let recorded_restart = !matches!(outcome, Ok(CycleResult::ShutdownRequested));
            if recorded_restart {
                let now = Instant::now();
                while let Some(&oldest) = restart_history.front() {
                    if now.duration_since(oldest) > RESTART_WINDOW {
                        restart_history.pop_front();
                    } else {
                        break;
                    }
                }
                restart_history.push_back(now);
            }

            match outcome {
                Ok(CycleResult::ShutdownRequested) => {
                    info!(target: "assistd::llama_server", "supervisor shutdown");
                    return;
                }
                Ok(CycleResult::CrashedAfterReady { status, ran_for }) => {
                    warn!(
                        target: "assistd::llama_server",
                        "llama-server exited after {ran_for:?} post-ready: {status}; restarting"
                    );
                    if ran_for >= Duration::from_secs(MIN_HEALTHY_SECONDS) {
                        consecutive_failures = 0;
                    } else {
                        consecutive_failures += 1;
                    }
                }
                Ok(CycleResult::FailedToStart { status }) => {
                    error!(
                        target: "assistd::llama_server",
                        "llama-server exited before reaching ready: {status}"
                    );
                    consecutive_failures += 1;
                }
                Err(e) => {
                    error!(
                        target: "assistd::llama_server",
                        "llama-server startup failed: {e}"
                    );
                    consecutive_failures += 1;
                }
            }

            let consecutive_tripped = consecutive_failures >= MAX_CONSECUTIVE_FAILURES;
            let window_tripped = restart_history.len() >= MAX_RESTARTS_PER_WINDOW;
            if consecutive_tripped || window_tripped {
                if window_tripped {
                    error!(
                        target: "assistd::llama_server",
                        restarts = restart_history.len(),
                        window_secs = RESTART_WINDOW.as_secs(),
                        "llama-server hit {MAX_RESTARTS_PER_WINDOW} restarts in rolling window; entering degraded state"
                    );
                } else {
                    error!(
                        target: "assistd::llama_server",
                        "{MAX_CONSECUTIVE_FAILURES} consecutive failures; entering degraded state"
                    );
                }
                let _ = self.ready_tx.send(ReadyState::Degraded);
                // Park until the daemon tells us to stop. Keeping the task
                // alive means the daemon process doesn't crash, and the
                // LlamaService::shutdown() path still works.
                let _ = self.shutdown_rx.wait_for(|v| *v).await;
                return;
            }

            if consecutive_failures == 0 {
                continue;
            }

            let delay = backoff_delay(consecutive_failures - 1);
            warn!(
                target: "assistd::llama_server",
                "restarting llama-server in {delay:?} (attempt {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})"
            );
            let _ = self.ready_tx.send(ReadyState::BackingOff {
                attempt: consecutive_failures,
            });

            tokio::select! {
                _ = tokio::time::sleep(delay) => {}
                _ = self.shutdown_rx.changed() => {
                    info!(target: "assistd::llama_server", "supervisor shutdown during backoff");
                    return;
                }
            }
        }
    }

    /// Spawns the child, waits for it to become healthy, then watches for exit or shutdown.
    async fn supervise_once(&mut self) -> Result<CycleResult, LlamaServerError> {
        let mut child = ChildProcess::spawn(&self.cfg, &self.model)?;
        *self.pid.lock() = child.pid();
        let ready_timeout = Duration::from_secs(self.cfg.ready_timeout_secs);
        let health = HealthChecker::new(&self.cfg.host, self.cfg.port, ready_timeout)?;

        enum Phase1 {
            Ready,
            ChildExited(ExitStatus),
            ShuttingDown,
            StartupError(LlamaServerError),
        }

        let phase1 = tokio::select! {
            res = health.wait_ready(&mut self.shutdown_rx) => match res {
                Ok(()) => Phase1::Ready,
                Err(LlamaServerError::ShutdownDuringHealth) => Phase1::ShuttingDown,
                Err(e) => Phase1::StartupError(e),
            },
            exit = child.wait() => match exit {
                Ok(status) => Phase1::ChildExited(status),
                Err(e) => Phase1::StartupError(LlamaServerError::Io(e)),
            }
        };

        match phase1 {
            Phase1::Ready => { /* fall through */ }
            Phase1::ChildExited(status) => {
                *self.pid.lock() = None;
                return Ok(CycleResult::FailedToStart { status });
            }
            Phase1::ShuttingDown => {
                child.shutdown(TERM_TIMEOUT).await?;
                *self.pid.lock() = None;
                return Ok(CycleResult::ShutdownRequested);
            }
            Phase1::StartupError(e) => {
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
