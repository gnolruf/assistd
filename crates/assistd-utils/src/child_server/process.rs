use std::io;
use std::path::PathBuf;
use std::process::{ExitStatus, Stdio};
use std::time::Duration;

use rustix::process::{Pid, Signal, kill_process_group};
use tokio::process::{Child, ChildStderr, ChildStdout, Command};
use tokio::task::JoinHandle;
use tokio::time::timeout;
use tracing::{info, warn};

use super::ChildServerSpec;
use super::error::ChildServerError;
use crate::log_lines::forward_lines;

const OUTPUT_FLUSH_TIMEOUT: Duration = Duration::from_millis(500);

/// A running child plus the tasks forwarding its output to tracing. The
/// child leads its own process group, so `pgid == pid`.
pub(super) struct ChildProcess {
    server: &'static str,
    child: Child,
    process_group: Pid,
    stdout_task: Option<JoinHandle<()>>,
    stderr_task: Option<JoinHandle<()>>,
}

impl ChildProcess {
    /// Spawn `spec`'s command and forward its stdout and stderr to tracing.
    pub(super) fn spawn<S: ChildServerSpec>(spec: &S) -> Result<Self, ChildServerError> {
        let server = spec.name();
        let mut cmd = spec.command();
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .process_group(0);
        set_parent_death_signal(&mut cmd);
        let path = PathBuf::from(cmd.as_std().get_program());

        let mut child = cmd.spawn().map_err(|source| ChildServerError::Spawn {
            server,
            path: path.clone(),
            source,
        })?;
        let process_group = child
            .id()
            .and_then(|pid| Pid::from_raw(pid as i32))
            .ok_or_else(|| ChildServerError::Spawn {
                server,
                path,
                source: io::Error::other("spawned child reported no pid"),
            })?;

        let stdout = child.stdout.take().expect("stdout piped but not captured");
        let stderr = child.stderr.take().expect("stderr piped but not captured");
        let stdout_task = tokio::spawn(forward_stdout(server, stdout));
        let stderr_task = tokio::spawn(forward_stderr(server, stderr));

        info!(
            target: "assistd::child_server",
            server,
            pid = child.id(),
            "spawned {server}: {:?}",
            cmd.as_std(),
        );

        Ok(Self {
            server,
            child,
            process_group,
            stdout_task: Some(stdout_task),
            stderr_task: Some(stderr_task),
        })
    }

    /// OS PID of the child, or `None` once it has exited.
    pub(super) fn pid(&self) -> Option<u32> {
        self.child.id()
    }

    /// Process group the child leads, fixed at spawn.
    pub(super) fn process_group(&self) -> Pid {
        self.process_group
    }

    /// Wait for the child to exit and return its status.
    pub(super) async fn wait(&mut self) -> io::Result<ExitStatus> {
        self.child.wait().await
    }

    /// SIGTERM the process group, wait up to `term_timeout`, then SIGKILL
    /// whatever is left of it. Log forwarders are drained briefly.
    pub(super) async fn shutdown(mut self, term_timeout: Duration) -> Result<(), ChildServerError> {
        let pgid = self.child.id().and_then(|pid| Pid::from_raw(pid as i32));
        if let Some(pgid) = pgid {
            let _ = kill_process_group(pgid, Signal::TERM);
        }

        match timeout(term_timeout, self.child.wait()).await {
            Ok(Ok(status)) => {
                info!(
                    target: "assistd::child_server",
                    server = self.server,
                    "{} exited after SIGTERM: {status}",
                    self.server,
                );
            }
            Ok(Err(e)) => return Err(ChildServerError::Io(e)),
            Err(_) => {
                warn!(
                    target: "assistd::child_server",
                    server = self.server,
                    "{} did not exit within {term_timeout:?}; sending SIGKILL",
                    self.server,
                );
                if let Some(pgid) = pgid {
                    let _ = kill_process_group(pgid, Signal::KILL);
                }
                let _ = self.child.start_kill();
                let _ = self.child.wait().await;
            }
        }

        if let Some(task) = self.stdout_task.take() {
            let _ = timeout(OUTPUT_FLUSH_TIMEOUT, task).await;
        }
        if let Some(task) = self.stderr_task.take() {
            let _ = timeout(OUTPUT_FLUSH_TIMEOUT, task).await;
        }

        Ok(())
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

#[cfg(not(target_os = "linux"))]
fn set_parent_death_signal(_cmd: &mut Command) {}

async fn forward_stdout(server: &'static str, stream: ChildStdout) {
    let forwarded = forward_lines(stream, |line| {
        info!(target: "assistd::child_server", server, "{line}");
    })
    .await;
    if let Err(e) = forwarded {
        warn!(target: "assistd::child_server", server, "stdout read error: {e}");
    }
}

async fn forward_stderr(server: &'static str, stream: ChildStderr) {
    let forwarded = forward_lines(stream, |line| {
        warn!(target: "assistd::child_server", server, "{line}");
    })
    .await;
    if let Err(e) = forwarded {
        warn!(target: "assistd::child_server", server, "stderr read error: {e}");
    }
}
