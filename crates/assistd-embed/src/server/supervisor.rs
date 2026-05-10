use super::backoff::{MAX_CONSECUTIVE_FAILURES, backoff_delay};
use super::error::EmbedServerError;
use super::health::HealthChecker;
use super::process::ChildProcess;
use super::service::ReadyState;
use assistd_config::EmbeddingConfig;
use std::process::ExitStatus;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::watch;
use tracing::{error, info, warn};

const MIN_HEALTHY_SECONDS: u64 = 30;
const TERM_TIMEOUT: Duration = Duration::from_secs(10);

enum CycleResult {
    ShutdownRequested,
    FailedToStart {
        status: ExitStatus,
    },
    CrashedAfterReady {
        status: ExitStatus,
        ran_for: Duration,
    },
}

/// Drives the embed-server process lifecycle: spawn, health-check, restart on crash,
/// and graceful shutdown.
pub struct Supervisor {
    /// Embedding server configuration (host, port, model, timeouts).
    pub cfg: EmbeddingConfig,
    /// Daemon-wide shutdown signal; `true` means stop.
    pub shutdown_rx: watch::Receiver<bool>,
    /// Channel used to publish the current [`ReadyState`] to [`EmbedService`].
    pub ready_tx: watch::Sender<ReadyState>,
    /// Shared slot for the child's OS PID, cleared when the child exits.
    pub pid: Arc<Mutex<Option<u32>>>,
}

impl Supervisor {
    /// Run the supervision loop until shutdown is requested or the child enters
    /// [`ReadyState::Degraded`] after too many consecutive failures.
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
                    info!(target: "assistd::embed_server", "supervisor shutdown");
                    return;
                }
                Ok(CycleResult::CrashedAfterReady { status, ran_for }) => {
                    warn!(
                        target: "assistd::embed_server",
                        "embed-server exited after {ran_for:?} post-ready: {status}; restarting"
                    );
                    if ran_for >= Duration::from_secs(MIN_HEALTHY_SECONDS) {
                        consecutive_failures = 0;
                    } else {
                        consecutive_failures += 1;
                    }
                }
                Ok(CycleResult::FailedToStart { status }) => {
                    error!(
                        target: "assistd::embed_server",
                        "embed-server exited before reaching ready: {status}"
                    );
                    consecutive_failures += 1;
                }
                Err(e) => {
                    error!(
                        target: "assistd::embed_server",
                        "embed-server startup failed: {e}"
                    );
                    consecutive_failures += 1;
                }
            }

            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
                error!(
                    target: "assistd::embed_server",
                    "{MAX_CONSECUTIVE_FAILURES} consecutive failures; entering degraded state"
                );
                let _ = self.ready_tx.send(ReadyState::Degraded);
                let _ = self.shutdown_rx.wait_for(|v| *v).await;
                return;
            }

            let delay = backoff_delay(consecutive_failures - 1);
            warn!(
                target: "assistd::embed_server",
                "restarting embed-server in {delay:?} (attempt {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})"
            );
            let _ = self.ready_tx.send(ReadyState::BackingOff {
                attempt: consecutive_failures,
            });

            tokio::select! {
                _ = tokio::time::sleep(delay) => {}
                _ = self.shutdown_rx.changed() => {
                    info!(target: "assistd::embed_server", "supervisor shutdown during backoff");
                    return;
                }
            }
        }
    }

    async fn supervise_once(&mut self) -> Result<CycleResult, EmbedServerError> {
        let mut child = ChildProcess::spawn(&self.cfg)?;
        *self.pid.lock().unwrap_or_else(|e| e.into_inner()) = child.pid();
        let ready_timeout = Duration::from_secs(self.cfg.ready_timeout_secs);
        let health = HealthChecker::new(&self.cfg.host, self.cfg.port, ready_timeout)?;

        enum Phase1 {
            Ready,
            ChildExited(ExitStatus),
            ShuttingDown,
            StartupError(EmbedServerError),
        }

        let phase1 = tokio::select! {
            res = health.wait_ready(&mut self.shutdown_rx) => match res {
                Ok(()) => Phase1::Ready,
                Err(EmbedServerError::ShutdownDuringHealth) => Phase1::ShuttingDown,
                Err(e) => Phase1::StartupError(e),
            },
            exit = child.wait() => match exit {
                Ok(status) => Phase1::ChildExited(status),
                Err(e) => Phase1::StartupError(EmbedServerError::Io(e)),
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

        let ready_at = Instant::now();
        let _ = self.ready_tx.send(ReadyState::Ready);
        info!(target: "assistd::embed_server", "embed-server ready");

        let result = tokio::select! {
            exit = child.wait() => match exit {
                Ok(status) => Ok(CycleResult::CrashedAfterReady {
                    status,
                    ran_for: ready_at.elapsed(),
                }),
                Err(e) => Err(EmbedServerError::Io(e)),
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
