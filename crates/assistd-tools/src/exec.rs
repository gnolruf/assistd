//! Shared subprocess spawning for the commands that run real processes,
//! in the two shapes those commands need:
//!
//! - [`supervise`] runs a child to completion under a timeout and returns
//!   its output. Used by [`crate::commands::BashCommand`], whose contract
//!   is "run this and give me the result".
//! - [`spawn_detached`] watches a child only long enough to catch a
//!   failed startup, then leaves it running. Used by `wm open`, whose
//!   contract is "launch this and leave the window open".
//!
//! [`supervise`] puts the child in its own process group and kills the
//! whole group once the child ends, however it ends, so a forked
//! grandchild can't leak or hold the output pipes open.
//! [`spawn_detached`] cannot do that and stay useful; bubblewrap's
//! `--die-with-parent` bounds a launched application instead.

#[cfg(unix)]
use std::os::unix::process::ExitStatusExt;
use std::pin::pin;
use std::process::{ExitStatus, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, ChildStdin, Command as ProcCommand};
use tokio::sync::oneshot;
use tokio::task::JoinSet;
use tokio::time::timeout;

use crate::chain::PIPE_BUF_MAX;
use crate::command::{CommandOutput, Hint, error_line};

/// Exit code for policy denial. POSIX "command found but not executable" is
/// the closest semantic match to "we recognize the command but refuse it".
pub(crate) const POLICY_DENIED_EXIT: i32 = 126;

/// Exit code for a spawn that never got off the ground (binary missing,
/// permission denied, sandbox helper absent).
pub(crate) const SPAWN_FAILED_EXIT: i32 = 127;

/// Exit code for timeout: 128 + SIGKILL, the convention bash uses.
pub(crate) const TIMEOUT_EXIT: i32 = 137;

/// Max bytes captured per stream while a supervised child runs. The chain
/// executor's own `PIPE_BUF_MAX` check only runs *after* this module
/// returns, so without a cap here a runaway script balloons daemon memory
/// before the timeout ever fires.
pub(crate) const OUTPUT_BUF_MAX: usize = PIPE_BUF_MAX;

/// Exit code returned when a child exceeds [`OUTPUT_BUF_MAX`] on either
/// pipe. Matches the chain executor's pipe-overflow exit so `||`
/// fallbacks behave the same regardless of where the overflow happened.
pub(crate) const OUTPUT_OVERFLOW_EXIT: i32 = 141;

/// How long [`spawn_detached`] watches a child before declaring it
/// launched. The failures worth catching are all fast — exec error, a
/// rejected bwrap flag, `cannot open display`, a crash on bad arguments.
/// One that fails slowly is reported as a successful launch.
const STARTUP_PROBE: Duration = Duration::from_millis(300);

/// Cap on output captured from a detached launch. Far below
/// [`OUTPUT_BUF_MAX`] because the readers hold this buffer for as long as
/// the application runs: it bounds what the daemon retains per launched
/// application, not merely per call.
const STARTUP_OUTPUT_MAX: usize = 64 * 1024;

/// Bound on waiting for the output pipes to reach EOF after a child has
/// exited. Normally instant, but a launcher that forks and exits leaves
/// the write end open in a grandchild, which would never EOF.
const POST_EXIT_DRAIN: Duration = Duration::from_millis(100);

/// How a child run by [`capture`] ended.
pub(crate) enum WaitOutcome {
    Exited(std::process::ExitStatus),
    WaitErr(std::io::Error),
    Timeout,
    /// A stream exceeded the byte cap; the child was killed.
    Overflow,
}

/// Output of a child run by [`capture`].
pub(crate) struct Captured {
    pub(crate) stdout: Vec<u8>,
    pub(crate) stderr: Vec<u8>,
    pub(crate) outcome: WaitOutcome,
}

/// Spawn `cmd` in its own process group, feed it `stdin`, and collect
/// up to `max_output` bytes per stream until it exits or `limit`
/// elapses. `stdin` is written while the child runs, so a child that
/// never reads it cannot stall the call. However the child ends, the
/// whole group is then killed, and the output pipes get at most
/// [`POST_EXIT_DRAIN`] to reach EOF, so a grandchild that escaped the
/// group while holding them cannot stall it either. `Err` is returned
/// only when the spawn itself fails.
pub(crate) async fn capture(
    mut cmd: ProcCommand,
    stdin: &[u8],
    limit: Duration,
    max_output: usize,
) -> std::io::Result<Captured> {
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    #[cfg(unix)]
    cmd.process_group(0);

    let mut child = cmd.spawn()?;
    let pgid = child.id();
    let stdin_pipe = child.stdin.take();
    let stdout_pipe = child.stdout.take().expect("stdout was piped");
    let stderr_pipe = child.stderr.take().expect("stderr was piped");

    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let outcome = {
        let mut readers = pin!(async {
            tokio::try_join!(
                read_capped(stdout_pipe, max_output, &mut stdout),
                read_capped(stderr_pipe, max_output, &mut stderr),
            )
        });

        let (mut outcome, drained) = {
            let mut wait = pin!(timeout(limit, wait_feeding(&mut child, stdin_pipe, stdin)));
            let mut drained = false;
            loop {
                tokio::select! {
                    res = &mut wait => break (match res {
                        Ok(Ok(status)) => WaitOutcome::Exited(status),
                        Ok(Err(e)) => WaitOutcome::WaitErr(e),
                        Err(_) => WaitOutcome::Timeout,
                    }, drained),
                    res = &mut readers, if !drained => match res {
                        Ok(_) => drained = true,
                        Err(Overflow) => break (WaitOutcome::Overflow, true),
                    },
                }
            }
        };

        kill_group(pgid);
        if !matches!(outcome, WaitOutcome::Exited(_)) {
            let _ = child.start_kill();
            let _ = child.wait().await;
        }

        if !drained && let Ok(Err(Overflow)) = timeout(POST_EXIT_DRAIN, &mut readers).await {
            outcome = WaitOutcome::Overflow;
        }
        outcome
    };

    Ok(Captured {
        stdout,
        stderr,
        outcome,
    })
}

async fn wait_feeding(
    child: &mut Child,
    pipe: Option<ChildStdin>,
    input: &[u8],
) -> std::io::Result<ExitStatus> {
    let feed = async move {
        if let Some(mut pipe) = pipe {
            let _ = pipe.write_all(input).await;
        }
    };
    let exited = tokio::select! {
        status = child.wait() => Some(status),
        () = feed => None,
    };
    match exited {
        Some(status) => status,
        None => child.wait().await,
    }
}

/// Run `cmd` to completion with [`capture`] and render the result as a
/// command output. `tool` names the caller in the timeout and overflow
/// error lines. `Err` is returned only when the spawn itself fails, so
/// each caller can attach its own recovery hint.
pub(crate) async fn supervise(
    tool: &str,
    cmd: ProcCommand,
    stdin: &[u8],
    limit: Duration,
) -> std::io::Result<CommandOutput> {
    let start = Instant::now();
    let Captured {
        stdout,
        stderr: stderr_bytes,
        outcome,
    } = capture(cmd, stdin, limit, OUTPUT_BUF_MAX).await?;

    Ok(match outcome {
        WaitOutcome::Exited(status) => CommandOutput {
            stdout,
            stderr: stderr_bytes,
            exit_code: exit_code(&status),
            attachments: Vec::new(),
        },
        WaitOutcome::WaitErr(e) => CommandOutput::failed(
            1,
            error_line(
                tool,
                format_args!("wait failed: {e}"),
                Hint::Try,
                "re-running the command",
            )
            .into_bytes(),
        ),
        WaitOutcome::Timeout => {
            let secs = limit.as_secs();
            let elapsed_secs = start.elapsed().as_secs_f64();
            let msg = format!(
                "[error] {tool}: timed out after {secs}s [exit:{TIMEOUT_EXIT} | {elapsed_secs:.1}s]\n"
            );
            CommandOutput::failed(TIMEOUT_EXIT, msg.into_bytes())
        }
        WaitOutcome::Overflow => {
            let overflow_msg = error_line(
                tool,
                format_args!("output exceeded {OUTPUT_BUF_MAX} bytes; child killed"),
                Hint::Try,
                "redirect to a file or pipe through head/wc -l to shrink the stream",
            )
            .into_bytes();
            let mut merged_stderr = stderr_bytes;
            merged_stderr.extend_from_slice(&overflow_msg);
            CommandOutput {
                stdout,
                stderr: merged_stderr,
                exit_code: OUTPUT_OVERFLOW_EXIT,
                attachments: Vec::new(),
            }
        }
    })
}

/// Owns the output readers of applications started by
/// [`spawn_detached`]. A reader runs until its application closes both
/// pipes; dropping the owner aborts any reader still running.
#[derive(Default)]
pub(crate) struct DetachedReaders(Mutex<JoinSet<()>>);

impl DetachedReaders {
    fn spawn(&self, reader: impl Future<Output = ()> + Send + 'static) {
        let mut set = self.0.lock();
        while set.try_join_next().is_some() {}
        set.spawn(reader);
    }
}

/// Spawn `cmd`, watch it for [`STARTUP_PROBE`], and leave it running if
/// it survives that window.
///
/// The counterpart to [`supervise`] for launching applications. A child
/// that exits inside the probe window is reported with its exit code and
/// captured output — this is the failure path, and it is what makes a
/// bad launch visible. A child still alive at the deadline is reported as
/// exit 0 and detached: no timeout bounds it and nothing kills it when
/// this future is dropped. Its output readers are handed to `readers`.
///
/// `Err` is returned only when the spawn itself fails, matching
/// [`supervise`] so callers can attach their own recovery hint.
pub(crate) async fn spawn_detached(
    tool: &str,
    mut cmd: ProcCommand,
    readers: &DetachedReaders,
) -> std::io::Result<CommandOutput> {
    // No `kill_on_drop`: dropping the handle below must leave the
    // application running. `process_group(0)` still isolates it from the
    // daemon's process group so a signal aimed at the daemon misses it.
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    cmd.process_group(0);

    let mut child = cmd.spawn()?;

    // The readers outlive this call by design: a long-lived application
    // must never block on a full pipe, nor be SIGPIPE'd by us closing the
    // read end. They end at EOF when it exits, dropping `drained`.
    let stdout_buf = Arc::new(Mutex::new(Vec::new()));
    let stderr_buf = Arc::new(Mutex::new(Vec::new()));
    let stdout_pipe = child.stdout.take().expect("stdout was piped");
    let stderr_pipe = child.stderr.take().expect("stderr was piped");
    let (drained, drained_rx) = oneshot::channel::<()>();
    let (stdout_sink, stderr_sink) = (stdout_buf.clone(), stderr_buf.clone());
    readers.spawn(async move {
        let _drained = drained;
        tokio::join!(
            drain_into(stdout_pipe, STARTUP_OUTPUT_MAX, stdout_sink),
            drain_into(stderr_pipe, STARTUP_OUTPUT_MAX, stderr_sink),
        );
    });

    let Ok(waited) = timeout(STARTUP_PROBE, child.wait()).await else {
        // Neither kills nor orphans it: tokio's reaper collects the
        // process when it eventually exits.
        drop(child);
        return Ok(CommandOutput::ok(Vec::new()));
    };

    // Exited during the probe. If a grandchild holds the write end open
    // the readers never finish, so we take whatever they captured.
    let _ = timeout(POST_EXIT_DRAIN, drained_rx).await;
    let stdout = std::mem::take(&mut *stdout_buf.lock());
    let stderr = std::mem::take(&mut *stderr_buf.lock());

    Ok(match waited {
        Ok(status) => CommandOutput {
            stdout,
            stderr,
            exit_code: exit_code(&status),
            attachments: Vec::new(),
        },
        Err(e) => CommandOutput::failed(
            1,
            error_line(
                tool,
                format_args!("wait failed: {e}"),
                Hint::Try,
                "re-running the command",
            )
            .into_bytes(),
        ),
    })
}

async fn drain_into<R: tokio::io::AsyncRead + Unpin>(
    mut reader: R,
    limit: usize,
    sink: Arc<Mutex<Vec<u8>>>,
) {
    let mut tmp = [0u8; 8192];
    loop {
        match reader.read(&mut tmp).await {
            Ok(0) | Err(_) => return,
            Ok(n) => {
                let mut buf = sink.lock();
                let room = limit.saturating_sub(buf.len());
                if room > 0 {
                    buf.extend_from_slice(&tmp[..n.min(room)]);
                }
            }
        }
    }
}

#[cfg(unix)]
fn kill_group(pgid: Option<u32>) {
    if let Some(pgid) = pgid.and_then(|p| rustix::process::Pid::from_raw(p as i32)) {
        let _ = rustix::process::kill_process_group(pgid, rustix::process::Signal::KILL);
    }
}

#[cfg(not(unix))]
fn kill_group(_pgid: Option<u32>) {}

/// Shell-style exit code: the status code, or 128 plus the signal that
/// killed the child.
pub(crate) fn exit_code(status: &std::process::ExitStatus) -> i32 {
    status
        .code()
        .or_else(|| signal_exit_code(status))
        .unwrap_or(1)
}

#[cfg(unix)]
fn signal_exit_code(status: &std::process::ExitStatus) -> Option<i32> {
    status.signal().map(|s| 128 + s)
}

#[cfg(not(unix))]
fn signal_exit_code(_status: &std::process::ExitStatus) -> Option<i32> {
    None
}

/// A stream passed its byte cap.
struct Overflow;

async fn read_capped<R: tokio::io::AsyncRead + Unpin>(
    mut reader: R,
    limit: usize,
    buf: &mut Vec<u8>,
) -> Result<(), Overflow> {
    let mut tmp = [0u8; 8192];
    loop {
        match reader.read(&mut tmp).await {
            Ok(0) | Err(_) => return Ok(()),
            Ok(n) => {
                buf.extend_from_slice(&tmp[..n]);
                if buf.len() > limit {
                    buf.truncate(limit);
                    return Err(Overflow);
                }
            }
        }
    }
}
