use super::backoff::{MAX_CONSECUTIVE_FAILURES, backoff_delay};
use super::config::{ModelSpec, ServerSpec};
use super::error::LlamaServerError;
use super::gguf::{self, LayerBudget};
use super::health::HealthChecker;
use super::process::ChildProcess;
use super::service::ReadyState;
use std::path::PathBuf;
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
    pub cfg: ServerSpec,
    pub model: ModelSpec,
    pub shutdown_rx: watch::Receiver<bool>,
    pub ready_tx: watch::Sender<ReadyState>,
    pub ngl: u32,
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
        let mut child = ChildProcess::spawn(&self.cfg, &self.model, self.ngl)?;
        *self.pid.lock().expect("pid mutex poisoned") = child.pid();
        let health = HealthChecker::new(&self.cfg.host, self.cfg.port)?;

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
                *self.pid.lock().expect("pid mutex poisoned") = None;
                return Ok(CycleResult::FailedToStart { status });
            }
            Phase1::ShuttingDown => {
                child.shutdown(TERM_TIMEOUT).await?;
                *self.pid.lock().expect("pid mutex poisoned") = None;
                return Ok(CycleResult::ShutdownRequested);
            }
            Phase1::StartupError(e) => {
                child.shutdown(TERM_TIMEOUT).await?;
                *self.pid.lock().expect("pid mutex poisoned") = None;
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
        *self.pid.lock().expect("pid mutex poisoned") = None;
        result
    }
}

/// Resolve `-ngl` once at startup: manual override takes precedence, otherwise
/// parse the GGUF header and derive from the VRAM budget. I/O errors on the
/// GGUF file propagate (bad model path is a config issue worth failing on),
/// but parse/metadata errors soft-fall-back to `ngl = 0` with a warning.
pub async fn resolve_ngl(cfg: &ServerSpec, model: &ModelSpec) -> Result<u32, LlamaServerError> {
    if let Some(n) = cfg.gpu_layers {
        info!(
            target: "assistd::llama_server",
            "using manual gpu_layers override: {n}"
        );
        return Ok(n);
    }

    let path = PathBuf::from(&model.path);
    let file_size_bytes = std::fs::metadata(&path)?.len();
    let file_size_mb = file_size_bytes / (1024 * 1024);

    let parse_path = path.clone();
    let header_result = tokio::task::spawn_blocking(move || gguf::parse_header(&parse_path))
        .await
        .map_err(|_| LlamaServerError::SupervisorPanic)?;

    let header = match header_result {
        Ok(h) => h,
        Err(e @ LlamaServerError::Io(_)) => return Err(e),
        Err(other) => {
            warn!(
                target: "assistd::llama_server",
                "GGUF parse failed ({other}); falling back to ngl=0"
            );
            return Ok(0);
        }
    };

    let budget = LayerBudget {
        file_size_mb,
        block_count: header.block_count,
    };
    let ngl = gguf::compute_ngl(model.vram_budget_mb, model.context_length, &budget);
    info!(
        target: "assistd::llama_server",
        arch = %header.architecture,
        block_count = header.block_count,
        file_size_mb,
        vram_budget_mb = model.vram_budget_mb,
        "computed ngl={ngl}"
    );
    Ok(ngl)
}
