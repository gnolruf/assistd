//! Subprocess spawning: [`supervise`] runs a child to completion under a
//! timeout, while [`watch_detached`] only watches a launch for early failure.

use std::io;
#[cfg(unix)]
use std::os::unix::process::ExitStatusExt;
use std::pin::{Pin, pin};
use std::process::{ExitStatus, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, ChildStdin, Command as ProcCommand};
use tokio::sync::oneshot;
use tokio::task::JoinSet;
use tokio::time::timeout;

use crate::chain::PIPE_BUF_MAX;
use crate::command::{CommandOutput, Hint, error_line};

/// Exit code for policy denial (POSIX "found but not executable").
pub(crate) const POLICY_DENIED_EXIT: i32 = 126;

/// Exit code for a spawn that never started (binary missing, permission
/// denied, sandbox helper absent).
pub(crate) const SPAWN_FAILED_EXIT: i32 = 127;

/// Exit code for timeout: 128 + SIGKILL, as bash reports it.
pub(crate) const TIMEOUT_EXIT: i32 = 137;

/// Max bytes captured per stream from a supervised child, enforced while it
/// runs so a runaway script cannot balloon daemon memory before the timeout.
pub(crate) const OUTPUT_BUF_MAX: usize = PIPE_BUF_MAX;

/// Exit code when a child exceeds [`OUTPUT_BUF_MAX`]; matches the chain
/// executor's pipe-overflow exit.
pub(crate) const OUTPUT_OVERFLOW_EXIT: i32 = 141;

/// How long [`watch_detached`] watches a child before declaring it launched.
/// Launch failures worth catching are fast; a slow one reads as success.
const STARTUP_PROBE: Duration = Duration::from_millis(300);

/// Cap on output kept from a detached launch; held for as long as the
/// application runs, so it bounds retention per launched application.
const STARTUP_OUTPUT_MAX: usize = 64 * 1024;

/// Bound on waiting for output pipes to reach EOF after the child exits,
/// since a grandchild can keep the write end open forever.
const POST_EXIT_DRAIN: Duration = Duration::from_millis(100);

/// How a child run by [`capture`] ended.
pub(crate) enum WaitOutcome {
    Exited(ExitStatus),
    WaitErr(io::Error),
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

/// A stream passed its byte cap.
struct Overflow;

/// Owns the output readers of applications watched by [`watch_detached`];
/// dropping it aborts any reader still running.
#[derive(Default)]
pub(crate) struct DetachedReaders(Mutex<JoinSet<()>>);

impl DetachedReaders {
    fn spawn(&self, reader: impl Future<Output = ()> + Send + 'static) {
        let mut set = self.0.lock();
        while set.try_join_next().is_some() {}
        set.spawn(reader);
    }
}

/// Output of a detached child, filled by readers that outlive the call so a
/// long-lived application never blocks on a full pipe or gets SIGPIPE.
struct StartupOutput {
    stdout: Arc<Mutex<Vec<u8>>>,
    stderr: Arc<Mutex<Vec<u8>>>,
    drained: oneshot::Receiver<()>,
}

impl StartupOutput {
    fn start(child: &mut Child, readers: &DetachedReaders) -> Self {
        let stdout = Arc::new(Mutex::new(Vec::new()));
        let stderr = Arc::new(Mutex::new(Vec::new()));
        let stdout_pipe = child.stdout.take().expect("stdout was piped");
        let stderr_pipe = child.stderr.take().expect("stderr was piped");
        let (drained_tx, drained) = oneshot::channel::<()>();
        let (stdout_sink, stderr_sink) = (stdout.clone(), stderr.clone());
        readers.spawn(async move {
            let _drained = drained_tx;
            tokio::join!(
                drain_into(stdout_pipe, STARTUP_OUTPUT_MAX, stdout_sink),
                drain_into(stderr_pipe, STARTUP_OUTPUT_MAX, stderr_sink),
            );
        });
        Self {
            stdout,
            stderr,
            drained,
        }
    }

    /// Wait up to [`POST_EXIT_DRAIN`] for the readers to finish, then take
    /// whatever they captured.
    async fn take_after_exit(self) -> (Vec<u8>, Vec<u8>) {
        let _ = timeout(POST_EXIT_DRAIN, self.drained).await;
        let stdout = std::mem::take(&mut *self.stdout.lock());
        let stderr = std::mem::take(&mut *self.stderr.lock());
        (stdout, stderr)
    }
}

/// Spawn `cmd` in its own process group, feed it `stdin` while it runs, and
/// collect up to `max_output` bytes per stream until it exits or `limit`
/// elapses. The whole group is then killed. `Err` means the spawn failed.
pub(crate) async fn capture(
    mut cmd: ProcCommand,
    stdin: &[u8],
    limit: Duration,
    max_output: usize,
) -> io::Result<Captured> {
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

        let (mut outcome, drained) =
            wait_racing_readers(&mut child, stdin_pipe, stdin, limit, readers.as_mut()).await;

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

/// Wait for `child` under `limit` while polling `readers`; returns how the
/// wait ended and whether the readers already reached EOF.
async fn wait_racing_readers<F>(
    child: &mut Child,
    stdin_pipe: Option<ChildStdin>,
    stdin: &[u8],
    limit: Duration,
    mut readers: Pin<&mut F>,
) -> (WaitOutcome, bool)
where
    F: Future<Output = Result<((), ()), Overflow>>,
{
    let mut wait = pin!(timeout(limit, wait_feeding(child, stdin_pipe, stdin)));
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
}

async fn wait_feeding(
    child: &mut Child,
    pipe: Option<ChildStdin>,
    input: &[u8],
) -> io::Result<ExitStatus> {
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

/// Run `cmd` to completion with [`capture`] and render the result; `tool`
/// names the command on timeout and overflow lines. `Err` means the spawn
/// failed.
pub(crate) async fn supervise(
    tool: &str,
    cmd: ProcCommand,
    stdin: &[u8],
    limit: Duration,
) -> io::Result<CommandOutput> {
    let start = Instant::now();
    let Captured {
        stdout,
        stderr,
        outcome,
    } = capture(cmd, stdin, limit, OUTPUT_BUF_MAX).await?;

    Ok(match outcome {
        WaitOutcome::Exited(status) => exited(stdout, stderr, &status),
        WaitOutcome::WaitErr(e) => wait_failed(tool, &e),
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
            );
            let mut merged_stderr = stderr;
            merged_stderr.extend_from_slice(overflow_msg.as_bytes());
            CommandOutput {
                stdout,
                stderr: merged_stderr,
                exit_code: OUTPUT_OVERFLOW_EXIT,
                attachments: Vec::new(),
            }
        }
    })
}

/// Configure `cmd` for [`watch_detached`]: no stdin, piped output, and a
/// process group of its own.
pub(crate) fn detach(cmd: &mut ProcCommand) {
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    cmd.process_group(0);
}

/// Watch `child`, spawned from a command [`detach`] configured, for
/// [`STARTUP_PROBE`]. A child that exits in that window is reported with
/// its exit code and output; one still alive is reported as exit 0 and left
/// running, its readers handed to `readers`.
///
/// Nothing kills it on drop; bubblewrap's `--die-with-parent` bounds it.
pub(crate) async fn watch_detached(
    tool: &str,
    mut child: Child,
    readers: &DetachedReaders,
) -> CommandOutput {
    let output = StartupOutput::start(&mut child, readers);

    let Ok(waited) = timeout(STARTUP_PROBE, child.wait()).await else {
        // Neither kills nor orphans it: tokio's reaper collects it on exit.
        drop(child);
        return CommandOutput::ok(Vec::new());
    };

    let (stdout, stderr) = output.take_after_exit().await;
    match waited {
        Ok(status) => exited(stdout, stderr, &status),
        Err(e) => wait_failed(tool, &e),
    }
}

fn exited(stdout: Vec<u8>, stderr: Vec<u8>, status: &ExitStatus) -> CommandOutput {
    CommandOutput {
        stdout,
        stderr,
        exit_code: exit_code(status),
        attachments: Vec::new(),
    }
}

fn wait_failed(tool: &str, e: &io::Error) -> CommandOutput {
    CommandOutput::failed(
        1,
        error_line(
            tool,
            format_args!("wait failed: {e}"),
            Hint::Try,
            "re-running the command",
        )
        .into_bytes(),
    )
}

async fn drain_into<R: AsyncRead + Unpin>(mut reader: R, limit: usize, sink: Arc<Mutex<Vec<u8>>>) {
    let mut chunk = [0u8; 8192];
    loop {
        match reader.read(&mut chunk).await {
            Ok(0) | Err(_) => return,
            Ok(n) => {
                let mut buf = sink.lock();
                let room = limit.saturating_sub(buf.len());
                if room > 0 {
                    buf.extend_from_slice(&chunk[..n.min(room)]);
                }
            }
        }
    }
}

async fn read_capped<R: AsyncRead + Unpin>(
    mut reader: R,
    limit: usize,
    buf: &mut Vec<u8>,
) -> Result<(), Overflow> {
    let mut chunk = [0u8; 8192];
    loop {
        match reader.read(&mut chunk).await {
            Ok(0) | Err(_) => return Ok(()),
            Ok(n) => {
                buf.extend_from_slice(&chunk[..n]);
                if buf.len() > limit {
                    buf.truncate(limit);
                    return Err(Overflow);
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

/// Shell-style exit code: the status code, or 128 plus the killing signal.
pub(crate) fn exit_code(status: &ExitStatus) -> i32 {
    status
        .code()
        .or_else(|| signal_exit_code(status))
        .unwrap_or(1)
}

#[cfg(unix)]
fn signal_exit_code(status: &ExitStatus) -> Option<i32> {
    status.signal().map(|s| 128 + s)
}

#[cfg(not(unix))]
fn signal_exit_code(_status: &ExitStatus) -> Option<i32> {
    None
}
