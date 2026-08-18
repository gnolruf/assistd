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
//! [`supervise`] takes the whole process group down with it
//! (`kill_on_drop` + `process_group(0)`, as in
//! `assistd-llm/src/llama_server/process.rs`), so a forked grandchild
//! can't leak. [`spawn_detached`] cannot do that and stay useful;
//! bubblewrap's `--die-with-parent` bounds a launched application
//! instead.

#[cfg(unix)]
use std::os::unix::process::ExitStatusExt;
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::Command as ProcCommand;
use tokio::sync::Notify;
use tokio::time::timeout;

use crate::chain::PIPE_BUF_MAX;
use crate::command::{CommandOutput, error_line};

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

/// Spawn `cmd`, write `stdin` to it, and collect its output under `limit`.
///
/// `tool` names the caller in the timeout and overflow error lines. The
/// `Err` variant is returned only when the spawn itself fails, so each
/// caller can attach its own recovery hint (which binary to check).
pub(crate) async fn supervise(
    tool: &str,
    mut cmd: ProcCommand,
    stdin: &[u8],
    limit: Duration,
) -> std::io::Result<CommandOutput> {
    let start = Instant::now();
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    #[cfg(unix)]
    cmd.process_group(0);

    let mut child = cmd.spawn()?;

    if let Some(mut pipe) = child.stdin.take() {
        if !stdin.is_empty() {
            let _ = pipe.write_all(stdin).await;
        }
        // Drop closes the pipe, signaling EOF to the child.
        drop(pipe);
    }

    // Read the pipes ourselves; `wait_with_output()` would collect them
    // into unbounded Vecs, and `yes` buffers tens of GB before the
    // timeout fires.
    let stdout_pipe = child.stdout.take().expect("stdout was piped");
    let stderr_pipe = child.stderr.take().expect("stderr was piped");
    let overflow = Arc::new(Notify::new());
    let stdout_task = tokio::spawn(read_capped(stdout_pipe, OUTPUT_BUF_MAX, overflow.clone()));
    let stderr_task = tokio::spawn(read_capped(stderr_pipe, OUTPUT_BUF_MAX, overflow.clone()));

    let outcome = tokio::select! {
        res = timeout(limit, child.wait()) => match res {
            Ok(Ok(status)) => WaitOutcome::Exited(status),
            Ok(Err(e)) => WaitOutcome::WaitErr(e),
            Err(_) => WaitOutcome::Timeout,
        },
        _ = overflow.notified() => WaitOutcome::Overflow,
    };

    if matches!(outcome, WaitOutcome::Timeout | WaitOutcome::Overflow) {
        let _ = child.start_kill();
        let _ = child.wait().await;
    }

    let (stdout, stdout_overflowed) = stdout_task.await.unwrap_or_default();
    let (stderr_bytes, stderr_overflowed) = stderr_task.await.unwrap_or_default();

    // Closing a pipe on overflow can SIGPIPE the child, making
    // `child.wait()` ready alongside the overflow notify; `select!` then
    // picks pseudo-randomly. The reader's bool is the source of truth.
    let outcome = if stdout_overflowed || stderr_overflowed {
        WaitOutcome::Overflow
    } else {
        outcome
    };

    Ok(match outcome {
        WaitOutcome::Exited(status) => {
            // `code()` is `None` for signal death; encoding it as
            // 128 + signum keeps a segfault reading as 139 instead of
            // masquerading as the 137 timeout sentinel.
            let exit_code = status
                .code()
                .or_else(|| signal_exit_code(&status))
                .unwrap_or(1);
            CommandOutput {
                stdout,
                stderr: stderr_bytes,
                exit_code,
                attachments: Vec::new(),
            }
        }
        WaitOutcome::WaitErr(e) => CommandOutput::failed(
            1,
            error_line(
                tool,
                format_args!("wait failed: {e}"),
                "Try",
                "re-running the command",
            )
            .into_bytes(),
        ),
        WaitOutcome::Timeout => {
            // Built inline rather than through `error_line`: the
            // elapsed-time suffix has no place in that helper's shape.
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
                "Try",
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

/// Spawn `cmd`, watch it for [`STARTUP_PROBE`], and leave it running if
/// it survives that window.
///
/// The counterpart to [`supervise`] for launching applications. A child
/// that exits inside the probe window is reported with its exit code and
/// captured output — this is the failure path, and it is what makes a
/// bad launch visible. A child still alive at the deadline is reported as
/// exit 0 and detached: no timeout bounds it and nothing kills it when
/// this future is dropped.
///
/// `Err` is returned only when the spawn itself fails, matching
/// [`supervise`] so callers can attach their own recovery hint.
pub(crate) async fn spawn_detached(
    tool: &str,
    mut cmd: ProcCommand,
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
    // read end. They end at EOF when it exits.
    let stdout_buf = Arc::new(Mutex::new(Vec::new()));
    let stderr_buf = Arc::new(Mutex::new(Vec::new()));
    let stdout_pipe = child.stdout.take().expect("stdout was piped");
    let stderr_pipe = child.stderr.take().expect("stderr was piped");
    let stdout_task = tokio::spawn(drain_into(
        stdout_pipe,
        STARTUP_OUTPUT_MAX,
        stdout_buf.clone(),
    ));
    let stderr_task = tokio::spawn(drain_into(
        stderr_pipe,
        STARTUP_OUTPUT_MAX,
        stderr_buf.clone(),
    ));

    let Ok(waited) = timeout(STARTUP_PROBE, child.wait()).await else {
        // Neither kills nor orphans it: tokio's reaper collects the
        // process when it eventually exits.
        drop(child);
        return Ok(CommandOutput::ok(Vec::new()));
    };

    // Exited during the probe. If a grandchild holds the write end open
    // the joins never finish, so the readers stay detached and we take
    // whatever they captured.
    let _ = timeout(POST_EXIT_DRAIN, async {
        let _ = stdout_task.await;
        let _ = stderr_task.await;
    })
    .await;
    let stdout = std::mem::take(&mut *stdout_buf.lock());
    let stderr = std::mem::take(&mut *stderr_buf.lock());

    Ok(match waited {
        Ok(status) => CommandOutput {
            stdout,
            stderr,
            exit_code: status
                .code()
                .or_else(|| signal_exit_code(&status))
                .unwrap_or(1),
            attachments: Vec::new(),
        },
        Err(e) => CommandOutput::failed(
            1,
            error_line(
                tool,
                format_args!("wait failed: {e}"),
                "Try",
                "re-running the command",
            )
            .into_bytes(),
        ),
    })
}

/// Read `reader` to EOF, keeping only the first `limit` bytes in `sink`.
///
/// Unlike [`read_capped`], the cap does not stop the read — the excess is
/// discarded and draining continues. Returning early would leave a
/// long-lived application blocked on a full pipe, or SIGPIPE it once the
/// read end closed.
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

enum WaitOutcome {
    Exited(std::process::ExitStatus),
    WaitErr(std::io::Error),
    Timeout,
    Overflow,
}

#[cfg(unix)]
fn signal_exit_code(status: &std::process::ExitStatus) -> Option<i32> {
    status.signal().map(|s| 128 + s)
}

#[cfg(not(unix))]
fn signal_exit_code(_status: &std::process::ExitStatus) -> Option<i32> {
    None
}

/// Read `reader` into a `Vec` capped at `limit`, returning the bytes and
/// whether the cap was hit. Hitting it closes the pipe and signals
/// `overflow` to wake the caller's `select!`.
///
/// The returned flag is authoritative, not the `Notify`: closing the pipe
/// may SIGPIPE the child and make `child.wait()` race the notification.
async fn read_capped<R: tokio::io::AsyncRead + Unpin>(
    mut reader: R,
    limit: usize,
    overflow: Arc<Notify>,
) -> (Vec<u8>, bool) {
    let mut buf: Vec<u8> = Vec::new();
    let mut tmp = [0u8; 8192];
    loop {
        match reader.read(&mut tmp).await {
            Ok(0) => return (buf, false),
            Ok(n) => {
                buf.extend_from_slice(&tmp[..n]);
                if buf.len() > limit {
                    buf.truncate(limit);
                    overflow.notify_one();
                    return (buf, true);
                }
            }
            Err(_) => return (buf, false),
        }
    }
}
