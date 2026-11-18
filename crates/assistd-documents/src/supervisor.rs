//! Runs the extraction worker as a child in its own process group, restarts it
//! with backoff when it dies or stops answering, and keeps the [`Extractor`]
//! in step with it.

use std::io;
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};

use assistd_utils::backoff::{RestartDecision, RestartPolicy};
use assistd_utils::process_group::{ProcessGroup, set_parent_death_signal};
use assistd_utils::readiness::NotReady;
use rustix::process::Signal;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::{mpsc, watch};
use tracing::{info, warn};

use crate::extractor::Extractor;
use crate::protocol::{FromWorker, MAX_REPLY_LINE_BYTES, Request, decode, encode};

/// How long a worker whose stdin closed has to exit before SIGTERM, and
/// again before SIGKILL.
const EXIT_GRACE: Duration = Duration::from_secs(2);

/// Builds the command one worker process is spawned from. Called on the
/// blocking pool before every start, so it may open files.
pub type WorkerCommand = Arc<dyn Fn() -> io::Result<Command> + Send + Sync>;

/// How a worker's session ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionEnd {
    Died,
    Shutdown,
}

/// A spawned worker and its pipes.
struct RunningWorker {
    child: Child,
    group: ProcessGroup,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl RunningWorker {
    async fn spawn(command: &WorkerCommand) -> io::Result<Self> {
        let build = Arc::clone(command);
        let mut command = tokio::task::spawn_blocking(move || build())
            .await
            .map_err(io::Error::other)??;
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .process_group(0);
        set_parent_death_signal(&mut command);
        let mut child = command.spawn()?;
        let group = ProcessGroup::led_by(&child)
            .ok_or_else(|| io::Error::other("spawned extraction worker reported no pid"))?;
        let stdin = child.stdin.take().expect("stdin piped");
        let stdout = BufReader::new(child.stdout.take().expect("stdout piped"));
        Ok(Self {
            child,
            group,
            stdin,
            stdout,
        })
    }

    /// Serve `extractor`'s requests until the worker dies, misses a deadline
    /// or `shutdown` flips, then stop it.
    async fn run(self, extractor: &Extractor, shutdown: &mut watch::Receiver<bool>) -> SessionEnd {
        let Self {
            mut child,
            mut group,
            stdin,
            mut stdout,
        } = self;
        let (outgoing, requests) = mpsc::unbounded_channel();
        let unresponsive = extractor.connect(outgoing);
        let ended = tokio::select! {
            Ok(()) = unresponsive => {
                warn!(target: "assistd::documents::worker", "terminating the unresponsive extraction worker");
                SessionEnd::Died
            }
            () = write_requests(stdin, requests) => SessionEnd::Died,
            () = read_messages(&mut stdout, extractor) => SessionEnd::Died,
            status = group.reap_leader(&mut child) => {
                warn!(target: "assistd::documents::worker", "extraction worker exited: {status:?}");
                SessionEnd::Died
            }
            _ = shutdown.wait_for(|stop| *stop) => SessionEnd::Shutdown,
        };
        extractor.disconnect();
        match ended {
            SessionEnd::Shutdown => stop_gracefully(child, group).await,
            SessionEnd::Died => {
                group.signal(Signal::KILL);
                let _ = group.reap_leader(&mut child).await;
            }
        }
        ended
    }
}

/// Run the extraction worker `command` builds until `shutdown` flips, restarting
/// it with backoff when it dies. `extractor` is starting until the worker says
/// it is ready, and unavailable between the slow retries of a worker that keeps dying.
pub async fn supervise(
    command: WorkerCommand,
    extractor: Arc<Extractor>,
    mut shutdown: watch::Receiver<bool>,
) {
    let mut policy = RestartPolicy::default();
    loop {
        let started = Instant::now();
        let ended = match RunningWorker::spawn(&command).await {
            Ok(worker) => {
                let ended = worker.run(&extractor, &mut shutdown).await;
                policy.record_session_end(started.elapsed());
                ended
            }
            Err(e) => {
                warn!(target: "assistd::documents::worker", "failed to spawn the extraction worker: {e}");
                policy.record_spawn_failure();
                SessionEnd::Died
            }
        };
        if ended == SessionEnd::Shutdown {
            return;
        }
        if !wait_to_restart(&mut policy, &extractor, &mut shutdown).await {
            return;
        }
    }
}

/// Sleep out the policy's delay, with `extractor` starting through a backoff
/// and unavailable through a slow retry; false when shutdown flips first.
async fn wait_to_restart(
    policy: &mut RestartPolicy,
    extractor: &Extractor,
    shutdown: &mut watch::Receiver<bool>,
) -> bool {
    let delay = match policy.next_restart(Instant::now()) {
        RestartDecision::Backoff { delay, failures } => {
            info!(
                target: "assistd::documents::worker",
                "restarting the extraction worker in {delay:?} ({failures} failed in a row)"
            );
            extractor.set_readiness(Err(NotReady::Starting));
            delay
        }
        RestartDecision::SlowRetry { delay, cap } => {
            let reason = format!(
                "the extraction worker {cap}; retrying every {} minutes",
                delay.as_secs() / 60
            );
            warn!(target: "assistd::documents::worker", "{reason}");
            extractor.set_readiness(Err(NotReady::Unavailable(reason.into())));
            delay
        }
    };
    let slept = tokio::select! {
        () = tokio::time::sleep(delay) => true,
        _ = shutdown.wait_for(|stop| *stop) => false,
    };
    if slept {
        extractor.set_readiness(Err(NotReady::Starting));
    }
    slept
}

/// Write each request to the worker's stdin; returns once the pipe fails.
async fn write_requests(mut stdin: ChildStdin, mut requests: mpsc::UnboundedReceiver<Request>) {
    while let Some(request) = requests.recv().await {
        let line = encode(&request);
        let written = async {
            stdin.write_all(line.as_bytes()).await?;
            stdin.flush().await
        };
        if let Err(e) = written.await {
            warn!(target: "assistd::documents::worker", "extraction worker stdin: {e}");
            return;
        }
    }
}

/// Hand `extractor` the worker's messages until its stdout closes or it
/// sends a line longer than any reply may be.
async fn read_messages(stdout: &mut BufReader<ChildStdout>, extractor: &Extractor) {
    loop {
        let line = match read_line(stdout).await {
            Ok(Some(line)) => line,
            Ok(None) => return,
            Err(e) => {
                warn!(target: "assistd::documents::worker", "extraction worker stdout: {e}");
                return;
            }
        };
        match decode::<FromWorker>(&line) {
            Ok(message) => extractor.receive(message),
            Err(e) => warn!(target: "assistd::documents::worker", "{e}; line ignored"),
        }
    }
}

/// The worker's next line, `None` at end of file; a line longer than
/// [`MAX_REPLY_LINE_BYTES`] is an error, so the worker cannot grow the daemon.
async fn read_line(stdout: &mut BufReader<ChildStdout>) -> io::Result<Option<Vec<u8>>> {
    let mut line = Vec::new();
    let read = stdout
        .take(MAX_REPLY_LINE_BYTES as u64 + 1)
        .read_until(b'\n', &mut line)
        .await?;
    if read == 0 {
        return Ok(None);
    }
    if line.len() > MAX_REPLY_LINE_BYTES && !line.ends_with(b"\n") {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "a line longer than any reply",
        ));
    }
    Ok(Some(line))
}

/// Give a worker whose stdin has closed time to exit, then SIGTERM and
/// finally SIGKILL its process group.
async fn stop_gracefully(mut child: Child, mut group: ProcessGroup) {
    if tokio::time::timeout(EXIT_GRACE, group.reap_leader(&mut child))
        .await
        .is_ok()
    {
        return;
    }
    warn!(target: "assistd::documents::worker", "extraction worker still running; sending SIGTERM");
    group.signal(Signal::TERM);
    if tokio::time::timeout(EXIT_GRACE, group.reap_leader(&mut child))
        .await
        .is_err()
    {
        group.signal(Signal::KILL);
        let _ = group.reap_leader(&mut child).await;
    }
}
