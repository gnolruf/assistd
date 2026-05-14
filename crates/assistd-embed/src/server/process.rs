use super::error::EmbedServerError;
use assistd_config::EmbeddingConfig;
use std::process::{ExitStatus, Stdio};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, ChildStderr, ChildStdout, Command};
use tokio::task::JoinHandle;
use tracing::{info, warn};

/// Wraps a running embedding llama-server child plus the tasks
/// forwarding its stdout/stderr to tracing.
pub struct ChildProcess {
    child: Child,
    stdout_task: Option<JoinHandle<()>>,
    stderr_task: Option<JoinHandle<()>>,
}

impl ChildProcess {
    /// Spawn a `llama-server` child process configured for embedding and begin
    /// forwarding its stdout/stderr to tracing.
    ///
    /// # Errors
    ///
    /// Returns [`EmbedServerError::Spawn`] if the process cannot be started.
    #[allow(clippy::expect_used)]
    pub fn spawn(cfg: &EmbeddingConfig) -> Result<Self, EmbedServerError> {
        // The embedding server is NOT in router mode. We pass `--hf-repo`
        // directly so the model is downloaded once and held resident for
        // the daemon's lifetime, so embedding requests should be cheap.
        // `--embedding` enables the `/v1/embeddings` endpoint.
        //
        // We also pass `--pooling mean` to make the endpoint match the
        // OpenAI-compatible response shape that nomic / bge models use.
        // (Without it, llama.cpp returns per-token vectors instead of one
        // sentence vector.)
        let mut cmd = Command::new("llama-server");
        cmd.arg("--embedding")
            .arg("--pooling")
            .arg("mean")
            .arg("--hf-repo")
            .arg(&cfg.model)
            .arg("-ngl")
            .arg(cfg.gpu_layers.to_string())
            .arg("--host")
            .arg(&cfg.host)
            .arg("--port")
            .arg(cfg.port.to_string());

        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        #[cfg(unix)]
        {
            // Process group so SIGTERM to -pid takes down any grandchildren.
            cmd.process_group(0);
        }

        let mut child = cmd.spawn().map_err(|source| EmbedServerError::Spawn {
            path: "llama-server".to_string(),
            source,
        })?;

        let stdout = child.stdout.take().expect("stdout piped but not captured");
        let stderr = child.stderr.take().expect("stderr piped but not captured");

        let stdout_task = tokio::spawn(forward_stdout(stdout));
        let stderr_task = tokio::spawn(forward_stderr(stderr));

        info!(
            target: "assistd::embed_server",
            pid = child.id(),
            "spawned embed-server: --embedding --pooling mean --hf-repo {} -ngl {} --host {} --port {}",
            cfg.model,
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

    /// Returns the OS PID of the child process, or `None` if it has already exited.
    pub fn pid(&self) -> Option<u32> {
        self.child.id()
    }

    /// Wait for the child process to exit and return its [`ExitStatus`].
    pub async fn wait(&mut self) -> std::io::Result<ExitStatus> {
        self.child.wait().await
    }

    /// Send SIGTERM to the child's process group, wait up to
    /// `term_timeout`, then SIGKILL if still running.
    pub async fn shutdown(mut self, term_timeout: Duration) -> Result<(), EmbedServerError> {
        #[cfg(unix)]
        if let Some(pid) = self.child.id() {
            // Child was spawned with `process_group(0)`, so its pgid equals
            // its pid. `kill_process_group` is the safe rustix wrapper
            // around `killpg(2)`; we ignore the result because the child
            // may have already exited.
            if let Some(pgid) = rustix::process::Pid::from_raw(pid as i32) {
                let _ = rustix::process::kill_process_group(pgid, rustix::process::Signal::TERM);
            }
        }

        match tokio::time::timeout(term_timeout, self.child.wait()).await {
            Ok(Ok(status)) => {
                info!(
                    target: "assistd::embed_server",
                    "embed-server exited after SIGTERM: {status}"
                );
            }
            Ok(Err(e)) => return Err(EmbedServerError::Io(e)),
            Err(_) => {
                warn!(
                    target: "assistd::embed_server",
                    "embed-server did not exit within {term_timeout:?}; sending SIGKILL"
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
            Ok(Some(line)) => info!(target: "assistd::embed_server", "{line}"),
            Ok(None) => return,
            Err(e) => {
                warn!(target: "assistd::embed_server", "stdout read error: {e}");
                return;
            }
        }
    }
}

async fn forward_stderr(stream: ChildStderr) {
    let mut lines = BufReader::new(stream).lines();
    loop {
        match lines.next_line().await {
            Ok(Some(line)) => warn!(target: "assistd::embed_server", "{line}"),
            Ok(None) => return,
            Err(e) => {
                warn!(target: "assistd::embed_server", "stderr read error: {e}");
                return;
            }
        }
    }
}
