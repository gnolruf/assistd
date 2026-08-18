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
//! [`supervise`]'s spawn pattern (`kill_on_drop(true)` +
//! `process_group(0)` on Unix) mirrors
//! `assistd-llm/src/llama_server/process.rs`: any mid-command daemon
//! shutdown kills the whole process group, preventing grandchild leaks if
//! the child itself forked. [`spawn_detached`] deliberately drops
//! `kill_on_drop`; bubblewrap's `--die-with-parent` is what bounds a
//! launched application's lifetime to the daemon's.

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

/// Exit code for timeout (128 + SIGKILL=9). The process group is killed
/// when the `tokio::time::timeout` future fires and the `tokio::Child`
/// handle's `kill_on_drop(true)` destructor runs.
pub(crate) const TIMEOUT_EXIT: i32 = 137;

/// Max bytes captured from the child's stdout or stderr while it runs.
/// Mirrors the chain executor's `PIPE_BUF_MAX` so a runaway child (e.g.
/// `bash "yes"`) can't balloon daemon memory before the timeout fires.
/// Applied per-stream: stdout and stderr are bounded independently.
pub(crate) const OUTPUT_BUF_MAX: usize = PIPE_BUF_MAX;

/// Exit code returned when a child exceeds [`OUTPUT_BUF_MAX`] on either
/// pipe. Matches the chain executor's pipe-overflow exit so `||`
/// fallbacks behave the same regardless of where the overflow happened.
pub(crate) const OUTPUT_OVERFLOW_EXIT: i32 = 141;

/// How long [`spawn_detached`] watches a child before declaring it
/// launched. Long enough to catch the failures that are all fast — exec
/// error, a rejected bwrap flag, `cannot open display`, a missing shared
/// library, a crash on bad arguments — and short enough that a successful
/// launch doesn't stall the agent turn. An application that fails *slowly*
/// is not observable this way, and is reported as a successful launch.
const STARTUP_PROBE: Duration = Duration::from_millis(300);

/// Cap on output captured from a detached launch, far below
/// [`OUTPUT_BUF_MAX`]. The only output that matters on this path is what
/// a failed launch prints in its first moments, and the readers hold this
/// buffer for as long as a surviving application runs — so the cap bounds
/// what the daemon retains per launched application, not just per call.
const STARTUP_OUTPUT_MAX: usize = 64 * 1024;

/// Bound on how long [`spawn_detached`] waits for the output pipes to
/// reach EOF after a child has already exited. Normally instant; a
/// launcher that forked before exiting leaves the write end open in a
/// grandchild, and waiting on that would hang the tool call forever.
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

    // Drive the child's stdout/stderr ourselves rather than letting
    // `wait_with_output()` collect them into unbounded Vecs. Without
    // this, a script like `yes` would buffer tens of GB in-process
    // before the chain executor's PIPE_BUF_MAX check ever fires; that
    // check only runs *after* this function returns.
    let stdout_pipe = child.stdout.take().expect("stdout was piped");
    let stderr_pipe = child.stderr.take().expect("stderr was piped");
    let overflow = Arc::new(Notify::new());
    let stdout_task = tokio::spawn(read_capped(stdout_pipe, OUTPUT_BUF_MAX, overflow.clone()));
    let stderr_task = tokio::spawn(read_capped(stderr_pipe, OUTPUT_BUF_MAX, overflow.clone()));

    // On timeout or overflow, the tokio::Child is killed; readers see
    // EOF as the kernel closes the pipes and the reader tasks join.
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

    // The reader closes its pipe on overflow, which can SIGPIPE the
    // child; that race makes `child.wait()` ready at the same time as
    // the overflow notify, and `select!` picks pseudo-randomly. The
    // bool returned by `read_capped` is the source of truth — if
    // either stream overflowed, force the Overflow outcome regardless
    // of which select branch won.
    let outcome = if stdout_overflowed || stderr_overflowed {
        WaitOutcome::Overflow
    } else {
        outcome
    };

    Ok(match outcome {
        WaitOutcome::Exited(status) => {
            // `ExitStatus::code()` returns `None` when the child was
            // killed by a signal. Encode signal death as `128 + signum`
            // (the POSIX convention bash itself uses) so a segfault
            // surfaces as 139 rather than masquerading as our timeout
            // sentinel 137. Non-unix or truly indeterminate cases fall
            // back to 1.
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
            // AC #2: exact one-line format including the `[exit:N | Ms]`
            // suffix inline. This is a deliberate divergence from the
            // `error_line` / presentation-footer convention because
            // the acceptance criterion pins the exact form.
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

    // The readers outlive this call on the detached path, by design: a
    // long-lived application must never block writing to a full pipe, and
    // must never be SIGPIPE'd by us closing the read end. They capture the
    // first `STARTUP_OUTPUT_MAX` bytes, discard the rest, and end at EOF
    // when the application exits.
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
        // Survived the probe. Dropping the handle neither kills nor
        // orphans it: tokio's reaper collects it when it eventually exits.
        drop(child);
        return Ok(CommandOutput::ok(Vec::new()));
    };

    // Exited during the probe. Both joins normally return the instant the
    // pipes hit EOF; the bound covers a surviving grandchild holding the
    // write end open, in which case the readers stay detached and we take
    // whatever they captured so far.
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
/// Unlike [`read_capped`], hitting the cap does not stop the read: the
/// excess is discarded and draining continues. On the detached path
/// returning early would leave a long-lived application blocked on a full
/// pipe, or kill it with SIGPIPE once the read end closed.
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

/// Read bytes from `reader` into a `Vec` capped at `limit`. Returns the
/// captured bytes and an `overflowed` flag: `true` iff the cap was hit.
/// On overflow the excess is dropped, `overflow` is signalled (to wake
/// the main `select!`), and the function returns. EOF and read errors
/// return `(buf, false)` without signalling.
///
/// The bool is the authoritative overflow signal — the `Notify` is only
/// an "wake up early" hint. The main task always checks this flag after
/// joining, because closing the pipe on return may SIGPIPE the child and
/// make `child.wait()` race the notify.
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
