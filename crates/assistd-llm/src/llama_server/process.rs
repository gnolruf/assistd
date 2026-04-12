use super::config::{ModelSpec, ServerSpec};
use super::error::LlamaServerError;
use std::process::{ExitStatus, Stdio};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, ChildStderr, ChildStdout, Command};
use tokio::task::JoinHandle;
use tracing::{info, warn};

/// Wraps a running llama-server child process plus the tasks forwarding its
/// stdout/stderr to tracing.
pub struct ChildProcess {
    child: Child,
    stdout_task: Option<JoinHandle<()>>,
    stderr_task: Option<JoinHandle<()>>,
}

impl ChildProcess {
    pub fn spawn(cfg: &ServerSpec, model: &ModelSpec) -> Result<Self, LlamaServerError> {
        let mut cmd = Command::new(&cfg.binary_path);
        cmd.arg("--jinja")
            .arg("--hf")
            .arg(&model.name)
            .arg("-ngl")
            .arg(cfg.gpu_layers.to_string())
            .arg("--host")
            .arg(&cfg.host)
            .arg("--port")
            .arg(cfg.port.to_string())
            .arg("-c")
            .arg(model.context_length.to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        #[cfg(unix)]
        {
            // Put the child in its own process group so `kill(-pid, SIGTERM)`
            // takes down any grandchildren along with it.
            cmd.process_group(0);
        }

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
            "spawned llama-server: {} --hf {} -ngl {} --host {} --port {}",
            cfg.binary_path,
            model.name,
            cfg.gpu_layers,
            cfg.host,
            cfg.port,
        );

        Ok(Self {
            child,
            stdout_task: Some(stdout_task),
            stderr_task: Some(stderr_task),
        })
    }

    pub fn pid(&self) -> Option<u32> {
        self.child.id()
    }

    /// Wait for the child to exit. Used inside the supervisor's `select!`.
    pub async fn wait(&mut self) -> std::io::Result<ExitStatus> {
        self.child.wait().await
    }

    /// Send SIGTERM to the child's process group, wait up to `term_timeout`,
    /// then fall back to SIGKILL if the child is still running. Both log
    /// forwarding tasks are awaited (briefly) so their buffered output lands
    /// in the journal before returning.
    pub async fn shutdown(mut self, term_timeout: Duration) -> Result<(), LlamaServerError> {
        #[cfg(unix)]
        if let Some(pid) = self.child.id() {
            // Negative pid signals the entire process group.
            let gid: i32 = -(pid as i32);
            // SAFETY: libc::kill is safe to call with any pid/signal pair;
            // failures are surfaced as a return code that we ignore because
            // the child may have already exited between this check and now.
            unsafe {
                libc::kill(gid, libc::SIGTERM);
            }
        }

        match tokio::time::timeout(term_timeout, self.child.wait()).await {
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
                let _ = self.child.start_kill();
                let _ = self.child.wait().await;
            }
        }

        if let Some(task) = self.stdout_task.take() {
            let _ = tokio::time::timeout(Duration::from_millis(500), task).await;
        }
        if let Some(task) = self.stderr_task.take() {
            let _ = tokio::time::timeout(Duration::from_millis(500), task).await;
        }

        Ok(())
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
