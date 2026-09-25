use std::process::{ExitStatus, Stdio};
use std::time::Duration;

use assistd_config::{LlamaServerConfig, ModelConfig};
#[cfg(unix)]
use rustix::process::{Pid, Signal, kill_process_group};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, ChildStderr, ChildStdout, Command};
use tokio::task::JoinHandle;
use tokio::time::timeout;
use tracing::{info, warn};

use super::error::LlamaServerError;

/// A running llama-server child plus the tasks forwarding its output to
/// tracing. On unix the child leads its own process group (pgid == pid).
pub struct ChildProcess {
    child: Child,
    stdout_task: Option<JoinHandle<()>>,
    stderr_task: Option<JoinHandle<()>>,
}

impl ChildProcess {
    /// Spawn a llama-server child and forward its stdout/stderr to tracing.
    pub fn spawn(cfg: &LlamaServerConfig, model: &ModelConfig) -> Result<Self, LlamaServerError> {
        let mut cmd = router_command(cfg, model);
        push_tuning_args(&mut cmd, cfg);
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        #[cfg(unix)]
        cmd.process_group(0);
        #[cfg(target_os = "linux")]
        set_parent_death_signal(&mut cmd);

        let mut child = cmd.spawn().map_err(|source| LlamaServerError::Spawn {
            path: cfg.binary_path.clone(),
            source,
        })?;

        let stdout = child.stdout.take().expect("stdout piped but not captured");
        let stderr = child.stderr.take().expect("stderr piped but not captured");

        let stdout_task = tokio::spawn(forward_stdout(stdout));
        let stderr_task = tokio::spawn(forward_stderr(stderr));

        info!(
            target: "assistd::llama_server",
            pid = child.id(),
            "spawned llama-server (router mode): {} -ngl {} --host {} --port {} -c {} (model {} loaded on demand)",
            cfg.binary_path.display(),
            cfg.gpu_layers,
            cfg.host,
            cfg.port,
            model.context_length,
            model.name,
        );

        Ok(Self {
            child,
            stdout_task: Some(stdout_task),
            stderr_task: Some(stderr_task),
        })
    }

    /// OS PID of the child, or `None` once it has exited.
    pub fn pid(&self) -> Option<u32> {
        self.child.id()
    }

    /// Wait for the child to exit and return its status.
    pub async fn wait(&mut self) -> std::io::Result<ExitStatus> {
        self.child.wait().await
    }

    /// SIGTERM the child's process group, wait up to `term_timeout`, then
    /// SIGKILL if it is still running. Log forwarders are drained briefly.
    pub async fn shutdown(mut self, term_timeout: Duration) -> Result<(), LlamaServerError> {
        #[cfg(unix)]
        let pgid = self.child.id().and_then(|pid| Pid::from_raw(pid as i32));
        #[cfg(unix)]
        if let Some(pgid) = pgid {
            let _ = kill_process_group(pgid, Signal::TERM);
        }

        match timeout(term_timeout, self.child.wait()).await {
            Ok(Ok(status)) => {
                info!(
                    target: "assistd::llama_server",
                    "llama-server exited after SIGTERM: {status}"
                );
            }
            Ok(Err(e)) => return Err(LlamaServerError::Io(e)),
            Err(_) => {
                warn!(
                    target: "assistd::llama_server",
                    "llama-server did not exit within {term_timeout:?}; sending SIGKILL"
                );
                #[cfg(unix)]
                if let Some(pgid) = pgid {
                    let _ = kill_process_group(pgid, Signal::KILL);
                }
                let _ = self.child.start_kill();
                let _ = self.child.wait().await;
            }
        }

        if let Some(task) = self.stdout_task.take() {
            let _ = timeout(Duration::from_millis(500), task).await;
        }
        if let Some(task) = self.stderr_task.take() {
            let _ = timeout(Duration::from_millis(500), task).await;
        }

        Ok(())
    }
}

/// Router-mode command line: no model is named, so weights load on demand
/// through `/models/load` while the process stays alive.
fn router_command(cfg: &LlamaServerConfig, model: &ModelConfig) -> Command {
    let mut cmd = Command::new(&cfg.binary_path);
    cmd.arg("--jinja")
        .arg("-ngl")
        .arg(cfg.gpu_layers.to_string())
        .arg("--host")
        .arg(cfg.host.to_string())
        .arg("--port")
        .arg(cfg.port.to_string())
        .arg("-c")
        .arg(model.context_length.to_string());
    cmd
}

fn push_tuning_args(cmd: &mut Command, cfg: &LlamaServerConfig) {
    if let Some(alias) = &cfg.alias {
        cmd.arg("--alias").arg(alias);
    }
    if let Some(override_tensor) = &cfg.override_tensor {
        cmd.arg("-ot").arg(override_tensor);
    }
    if let Some(flash) = cfg.flash_attn {
        cmd.arg("--flash-attn")
            .arg(if flash { "on" } else { "off" });
    }
    if let Some(cache_type_k) = &cfg.cache_type_k {
        cmd.arg("--cache-type-k").arg(cache_type_k);
    }
    if let Some(cache_type_v) = &cfg.cache_type_v {
        cmd.arg("--cache-type-v").arg(cache_type_v);
    }
    if let Some(threads) = cfg.threads {
        cmd.arg("--threads").arg(threads.to_string());
    }
    if let Some(batch_size) = cfg.batch_size {
        cmd.arg("--batch-size").arg(batch_size.to_string());
    }
    if let Some(ubatch_size) = cfg.ubatch_size {
        cmd.arg("--ubatch-size").arg(ubatch_size.to_string());
    }
    if let Some(n_cpu_moe) = cfg.n_cpu_moe {
        cmd.arg("--n-cpu-moe").arg(n_cpu_moe.to_string());
    }
    if let Some(cache_ram_mib) = cfg.cache_ram_mib {
        cmd.arg("--cache-ram").arg(cache_ram_mib.to_string());
    }
    if cfg.mlock == Some(true) {
        cmd.arg("--mlock");
    }
    if let Some(offload) = cfg.mmproj_offload {
        cmd.arg(if offload {
            "--mmproj-offload"
        } else {
            "--no-mmproj-offload"
        });
    }
}

/// Have the kernel SIGTERM the child when the daemon dies, even by SIGKILL.
/// `pre_exec` is the only way to set PDEATHSIG on a spawned child.
#[cfg(target_os = "linux")]
#[allow(unsafe_code)]
fn set_parent_death_signal(cmd: &mut Command) {
    // SAFETY: the closure runs in the child between fork() and exec(). It
    // captures nothing and only issues the prctl(PR_SET_PDEATHSIG) syscall,
    // which is async-signal-safe.
    unsafe {
        cmd.pre_exec(|| {
            rustix::process::set_parent_process_death_signal(Some(Signal::TERM)).map_err(Into::into)
        });
    }
}

async fn forward_stdout(stream: ChildStdout) {
    let mut lines = BufReader::new(stream).lines();
    loop {
        match lines.next_line().await {
            Ok(Some(line)) => info!(target: "assistd::llama_server", "{line}"),
            Ok(None) => return,
            Err(e) => {
                warn!(target: "assistd::llama_server", "stdout read error: {e}");
                return;
            }
        }
    }
}

async fn forward_stderr(stream: ChildStderr) {
    let mut lines = BufReader::new(stream).lines();
    loop {
        match lines.next_line().await {
            Ok(Some(line)) => warn!(target: "assistd::llama_server", "{line}"),
            Ok(None) => return,
            Err(e) => {
                warn!(target: "assistd::llama_server", "stderr read error: {e}");
                return;
            }
        }
    }
}
