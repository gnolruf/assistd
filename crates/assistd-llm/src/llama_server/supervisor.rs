use super::backoff::{MAX_CONSECUTIVE_FAILURES, backoff_delay};
use super::error::LlamaServerError;
use super::health::HealthChecker;
use super::process::ChildProcess;
use super::service::ReadyState;
use assistd_config::{LlamaServerConfig, ModelConfig};
use std::process::ExitStatus;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::watch;
use tracing::{error, info, warn};

/// Seconds a child must stay healthy after reaching Ready before a subsequent
/// exit is treated as a runtime crash (counter reset) rather than a startup
/// flap (counter increment).
const MIN_HEALTHY_SECONDS: u64 = 30;

/// Graceful shutdown budget per child — fits under systemd's 15s stop budget.
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

pub struct Supervisor {
    pub cfg: LlamaServerConfig,
    pub model: ModelConfig,
    pub shutdown_rx: watch::Receiver<bool>,
    pub ready_tx: watch::Sender<ReadyState>,
    pub pid: Arc<Mutex<Option<u32>>>,
}

impl Supervisor {
    pub async fn run(mut self) {
        let mut consecutive_failures: u32 = 0;

        loop {
            if *self.shutdown_rx.borrow() {
                return;
            }
            let _ = self.ready_tx.send(ReadyState::Starting);

            let outcome = self.supervise_once().await;

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

            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
                error!(
                    target: "assistd::llama_server",
                    "{MAX_CONSECUTIVE_FAILURES} consecutive failures; entering degraded state"
                );
                let _ = self.ready_tx.send(ReadyState::Degraded);
                // Park until the daemon tells us to stop. Keeping the task
                // alive means the daemon process doesn't crash, and the
                // LlamaService::shutdown() path still works.
                let _ = self.shutdown_rx.wait_for(|v| *v).await;
                return;
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

    async fn supervise_once(&mut self) -> Result<CycleResult, LlamaServerError> {
        let mut child = ChildProcess::spawn(&self.cfg, &self.model)?;
        *self.pid.lock().unwrap_or_else(|e| e.into_inner()) = child.pid();
        let ready_timeout = Duration::from_secs(self.cfg.ready_timeout_secs);
        let health = HealthChecker::new(&self.cfg.host, self.cfg.port, ready_timeout)?;

        // Phase 1: wait for /health to return 200. Race against child exit.
        // `health.wait_ready` internally handles shutdown signals.
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
                *self.pid.lock().unwrap_or_else(|e| e.into_inner()) = None;
                return Ok(CycleResult::FailedToStart { status });
            }
            Phase1::ShuttingDown => {
                child.shutdown(TERM_TIMEOUT).await?;
                *self.pid.lock().unwrap_or_else(|e| e.into_inner()) = None;
                return Ok(CycleResult::ShutdownRequested);
            }
            Phase1::StartupError(e) => {
                child.shutdown(TERM_TIMEOUT).await?;
                *self.pid.lock().unwrap_or_else(|e| e.into_inner()) = None;
                return Err(e);
            }
        }

        // Phase 2: child is healthy. Watch for exit or shutdown.
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
        *self.pid.lock().unwrap_or_else(|e| e.into_inner()) = None;
        result
    }
}
