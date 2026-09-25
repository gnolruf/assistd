use std::process::ExitStatus;
use std::time::{Duration, Instant};

use tokio::sync::watch;
use tracing::{error, info, warn};

use assistd_config::EmbeddingConfig;

use super::backoff::{MAX_CONSECUTIVE_FAILURES, backoff_delay};
use super::error::EmbedServerError;
use super::health::HealthChecker;
use super::process::ChildProcess;
use super::service::ReadyState;

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
}

impl Supervisor {
    /// Run until shutdown. Once in [`ReadyState::Degraded`] it stops
    /// restarting the child and only waits for shutdown.
    pub async fn run(mut self) {
        let mut consecutive_failures: u32 = 0;

        loop {
            if *self.shutdown_rx.borrow() {
                return;
            }
            let _ = self.ready_tx.send(ReadyState::Starting);

            let outcome = self.supervise_once().await;
            let Some(failures) = tally_failures(outcome, consecutive_failures) else {
                return;
            };
            consecutive_failures = failures;

            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
                self.park_degraded().await;
                return;
            }
            if consecutive_failures > 0 && !self.wait_backoff(consecutive_failures).await {
                return;
            }
        }
    }

    async fn park_degraded(&mut self) {
        error!(
            target: "assistd::embed_server",
            "{MAX_CONSECUTIVE_FAILURES} consecutive failures; entering degraded state"
        );
        let _ = self.ready_tx.send(ReadyState::Degraded);
        let _ = self.shutdown_rx.wait_for(|v| *v).await;
    }

    /// Returns `false` if shutdown arrived during the wait.
    async fn wait_backoff(&mut self, attempt: u32) -> bool {
        let delay = backoff_delay(attempt - 1);
        warn!(
            target: "assistd::embed_server",
            "restarting embed-server in {delay:?} (attempt {attempt}/{MAX_CONSECUTIVE_FAILURES})"
        );
        let _ = self.ready_tx.send(ReadyState::BackingOff { attempt });

        tokio::select! {
            _ = tokio::time::sleep(delay) => true,
            _ = self.shutdown_rx.changed() => {
                info!(target: "assistd::embed_server", "supervisor shutdown during backoff");
                false
            }
        }
    }

    async fn supervise_once(&mut self) -> Result<CycleResult, EmbedServerError> {
        let mut child = ChildProcess::spawn(&self.cfg)?;
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
                return Ok(CycleResult::FailedToStart { status });
            }
            Startup::ShuttingDown => {
                child.shutdown(TERM_TIMEOUT).await?;
                return Ok(CycleResult::ShutdownRequested);
            }
            Startup::Failed(e) => {
                child.shutdown(TERM_TIMEOUT).await?;
                return Err(e);
            }
        }

        let ready_at = Instant::now();
        let _ = self.ready_tx.send(ReadyState::Ready);
        info!(target: "assistd::embed_server", "embed-server ready");

        tokio::select! {
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
        }
    }
}

/// Log one cycle's outcome and return the updated consecutive-failure count, or `None`
/// on shutdown. A crash after [`MIN_HEALTHY_SECONDS`] of uptime resets the count.
fn tally_failures(
    outcome: Result<CycleResult, EmbedServerError>,
    consecutive_failures: u32,
) -> Option<u32> {
    match outcome {
        Ok(CycleResult::ShutdownRequested) => {
            info!(target: "assistd::embed_server", "supervisor shutdown");
            None
        }
        Ok(CycleResult::CrashedAfterReady { status, ran_for }) => {
            warn!(
                target: "assistd::embed_server",
                "embed-server exited after {ran_for:?} post-ready: {status}; restarting"
            );
            if ran_for >= Duration::from_secs(MIN_HEALTHY_SECONDS) {
                Some(0)
            } else {
                Some(consecutive_failures + 1)
            }
        }
        Ok(CycleResult::FailedToStart { status }) => {
            error!(
                target: "assistd::embed_server",
                "embed-server exited before reaching ready: {status}"
            );
            Some(consecutive_failures + 1)
        }
        Err(e) => {
            error!(
                target: "assistd::embed_server",
                "embed-server startup failed: {e}"
            );
            Some(consecutive_failures + 1)
        }
    }
}
