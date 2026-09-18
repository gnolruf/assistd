use super::backoff::{MAX_CONSECUTIVE_FAILURES, backoff_delay};
use super::error::EmbedServerError;
use super::health::HealthChecker;
use super::process::ChildProcess;
use super::service::ReadyState;
use assistd_config::EmbeddingConfig;
use parking_lot::Mutex;
use std::process::ExitStatus;
use std::sync::Arc;
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

enum Startup {
    Ready,
    ChildExited(ExitStatus),
    ShuttingDown,
    Failed(EmbedServerError),
}

/// Drives the embed-server lifecycle: spawn, health check, restart on
/// crash, graceful shutdown.
pub struct Supervisor {
    pub cfg: EmbeddingConfig,
    pub ready_timeout: Duration,
    pub shutdown_rx: watch::Receiver<bool>,
    pub ready_tx: watch::Sender<ReadyState>,
    /// The child's PID while it runs.
    pub pid: Arc<Mutex<Option<u32>>>,
}

impl Supervisor {
    /// Run until shutdown or until the child enters
    /// [`ReadyState::Degraded`].
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
        *self.pid.lock() = child.pid();
        let ready_timeout = self.ready_timeout;
        let health = HealthChecker::new(
            &self.cfg.host.to_string(),
            self.cfg.port.get(),
            ready_timeout,
        )?;

        let startup = tokio::select! {
            res = health.wait_ready(&mut self.shutdown_rx) => match res {
                Ok(()) => Startup::Ready,
                Err(EmbedServerError::ShutdownDuringHealth) => Startup::ShuttingDown,
                Err(e) => Startup::Failed(e),
            },
            exit = child.wait() => match exit {
                Ok(status) => Startup::ChildExited(status),
                Err(e) => Startup::Failed(EmbedServerError::Io(e)),
            }
        };

        match startup {
            Startup::Ready => {}
            Startup::ChildExited(status) => {
                *self.pid.lock() = None;
                return Ok(CycleResult::FailedToStart { status });
            }
            Startup::ShuttingDown => {
                child.shutdown(TERM_TIMEOUT).await?;
                *self.pid.lock() = None;
                return Ok(CycleResult::ShutdownRequested);
            }
            Startup::Failed(e) => {
                child.shutdown(TERM_TIMEOUT).await?;
                *self.pid.lock() = None;
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
        *self.pid.lock() = None;
        result
    }
}
