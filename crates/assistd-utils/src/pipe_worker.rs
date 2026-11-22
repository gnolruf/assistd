//! Supervises a child process that speaks line-delimited messages over its
//! stdin and stdout, and matches the replies it sends to the calls awaiting them.

use std::collections::HashMap;
use std::io;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use rustix::process::Signal;
use thiserror::Error;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::{mpsc, oneshot, watch};
use tracing::{info, warn};

use crate::backoff::{RestartDecision, RestartPolicy, backoff_delay};
use crate::process_group::{ProcessGroup, set_parent_death_signal};
use crate::readiness::NotReady;

/// How long a worker whose stdin closed has to exit before SIGTERM, and
/// again before SIGKILL.
const EXIT_GRACE: Duration = Duration::from_secs(2);

/// What a worker's exit is held against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitBlame {
    /// The worker: the exit counts toward its restart policy.
    Worker,
    /// The work it was given, which has now ended a worker twice running: the
    /// restart policy does not count the exit.
    Work,
}

/// One kind of worker process, and the daemon's end of its pipes.
pub trait PipeWorker: Send + Sync + 'static {
    /// A message written to the worker's stdin.
    type Outgoing: Send;

    /// What the log calls the worker.
    const NAME: &'static str;

    /// Most bytes of one line read from the worker; a longer line ends its session.
    const MAX_LINE_BYTES: usize;

    /// How long a new worker has to write its first line before it is killed
    /// as one that never started; `None` when starting may take any time.
    const STARTUP_DEADLINE: Option<Duration>;

    /// The command one worker process is spawned from. Called on the
    /// blocking pool before every start, so it may open files.
    fn command(&self) -> io::Result<Command>;

    /// Route messages to a newly spawned worker through `outgoing`; the returned
    /// receiver resolves once that worker is to be killed as unresponsive.
    fn connect(&self, outgoing: mpsc::UnboundedSender<Self::Outgoing>) -> oneshot::Receiver<()>;

    /// `message` as one line, newline included.
    fn encode(message: &Self::Outgoing) -> String;

    /// Act on one line the worker wrote, line ending included.
    fn receive(&self, line: &[u8]);

    /// The connected worker is gone; returns what its exit is held against.
    fn disconnect(&self) -> ExitBlame;

    /// No worker runs until the next restart: [`NotReady::Starting`] through a
    /// backoff, [`NotReady::Unavailable`] between the slow retries of one that keeps dying.
    fn set_not_ready(&self, not_ready: NotReady);
}

/// A call the worker did not answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum CallError {
    #[error("the worker is not running")]
    NotRunning,
    #[error("the worker exited before it answered")]
    Exited,
    /// The worker missed the call's deadline and is being restarted.
    #[error("the worker did not answer within {deadline:?}")]
    Unresponsive { deadline: Duration },
}

/// The running worker's input queue and its calls awaiting a reply.
#[derive(Debug)]
struct Connection<M, P> {
    outgoing: mpsc::UnboundedSender<M>,
    pending: HashMap<u64, P>,
    unresponsive: Option<oneshot::Sender<()>>,
}

/// The calls sent to a worker as messages `M`, each holding a `P` until its
/// reply arrives. Disconnecting drops every `P` still held.
#[derive(Debug)]
pub struct PendingReplies<M, P> {
    name: &'static str,
    connection: Mutex<Option<Connection<M, P>>>,
    next_id: AtomicU64,
}

impl<M, P> PendingReplies<M, P> {
    /// No worker is connected; `name` is what the log calls the worker.
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            connection: Mutex::new(None),
            next_id: AtomicU64::new(0),
        }
    }

    /// Route calls to a newly spawned worker through `outgoing`; the returned
    /// receiver resolves once that worker misses a call's deadline.
    pub fn connect(&self, outgoing: mpsc::UnboundedSender<M>) -> oneshot::Receiver<()> {
        let (unresponsive_tx, unresponsive_rx) = oneshot::channel();
        *self.connection.lock() = Some(Connection {
            outgoing,
            pending: HashMap::new(),
            unresponsive: Some(unresponsive_tx),
        });
        unresponsive_rx
    }

    /// The worker is gone: every pending call fails with [`CallError::Exited`].
    pub fn disconnect(&self) {
        self.connection.lock().take();
    }

    /// Send the message `message` builds from a new call id, hold `awaiting`
    /// under that id, and wait for `reply`, which whoever takes `awaiting`
    /// resolves. A worker that misses `deadline` is flagged unresponsive.
    pub async fn call<T>(
        &self,
        message: impl FnOnce(u64) -> M,
        awaiting: P,
        mut reply: oneshot::Receiver<T>,
        deadline: Option<Duration>,
    ) -> Result<T, CallError> {
        let id = self.send_call(message, awaiting)?;
        let Some(deadline) = deadline else {
            return reply.await.map_err(|_| CallError::Exited);
        };
        if let Ok(answered) = tokio::time::timeout(deadline, &mut reply).await {
            return answered.map_err(|_| CallError::Exited);
        }
        if self.abandon(id) {
            return Err(CallError::Unresponsive { deadline });
        }
        reply.await.map_err(|_| CallError::Exited)
    }

    /// Send a message that has no reply; dropped while no worker runs.
    pub fn notify(&self, message: M) {
        if let Some(connection) = &*self.connection.lock() {
            let _ = connection.outgoing.send(message);
        }
    }

    /// What call `id` holds, when it is still pending.
    pub fn take(&self, id: u64) -> Option<P> {
        self.take_if(id, |_| true)
    }

    /// What call `id` holds, when it is still pending and `expects` the reply
    /// that arrived; otherwise the call stays pending.
    pub fn take_if(&self, id: u64, expects: impl FnOnce(&P) -> bool) -> Option<P> {
        let mut connection = self.connection.lock();
        let pending = &mut connection.as_mut()?.pending;
        if !expects(pending.get(&id)?) {
            return None;
        }
        pending.remove(&id)
    }

    fn send_call(&self, message: impl FnOnce(u64) -> M, awaiting: P) -> Result<u64, CallError> {
        let mut connection = self.connection.lock();
        let connection = connection.as_mut().ok_or(CallError::NotRunning)?;
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        connection.pending.insert(id, awaiting);
        connection
            .outgoing
            .send(message(id))
            .map_err(|_| CallError::Exited)?;
        Ok(id)
    }

    /// Give up on call `id` and flag its worker unresponsive; false when the
    /// call was already answered or its worker already gone.
    fn abandon(&self, id: u64) -> bool {
        let mut connection = self.connection.lock();
        let Some(connection) = connection.as_mut() else {
            return false;
        };
        if connection.pending.remove(&id).is_none() {
            return false;
        }
        warn!(target: "assistd::pipe_worker", "{} missed a call's deadline", self.name);
        if let Some(unresponsive) = connection.unresponsive.take() {
            let _ = unresponsive.send(());
        }
        true
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionEnd {
    NeverStarted,
    Died,
    /// Died over work that had already ended the worker before it.
    DiedOverItsWork,
    Shutdown,
}

struct RunningWorker {
    child: Child,
    group: ProcessGroup,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl RunningWorker {
    async fn spawn<W: PipeWorker>(worker: &Arc<W>) -> io::Result<Self> {
        let builder = Arc::clone(worker);
        let mut command = tokio::task::spawn_blocking(move || builder.command())
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
            .ok_or_else(|| io::Error::other(format!("spawned {} reported no pid", W::NAME)))?;
        let stdin = child.stdin.take().expect("stdin piped");
        let stdout = BufReader::new(child.stdout.take().expect("stdout piped"));
        Ok(Self {
            child,
            group,
            stdin,
            stdout,
        })
    }

    /// Carry `worker`'s messages until the process dies, is flagged
    /// unresponsive or `shutdown` flips, then stop it.
    async fn run<W: PipeWorker>(
        self,
        worker: &W,
        shutdown: &mut watch::Receiver<bool>,
    ) -> SessionEnd {
        let Self {
            mut child,
            mut group,
            stdin,
            mut stdout,
        } = self;
        let (outgoing, messages) = mpsc::unbounded_channel();
        let unresponsive = worker.connect(outgoing);
        let ended = tokio::select! {
            Ok(()) = unresponsive => {
                warn!(target: "assistd::pipe_worker", "terminating the unresponsive {}", W::NAME);
                SessionEnd::Died
            }
            () = write_messages::<W>(stdin, messages) => SessionEnd::Died,
            ended = read_messages(&mut stdout, worker) => ended,
            status = group.reap_leader(&mut child) => {
                warn!(target: "assistd::pipe_worker", "{} exited: {status:?}", W::NAME);
                SessionEnd::Died
            }
            _ = shutdown.wait_for(|stop| *stop) => SessionEnd::Shutdown,
        };
        let blame = worker.disconnect();
        if ended == SessionEnd::Shutdown {
            group.stop_gracefully(child, EXIT_GRACE, W::NAME).await;
            return ended;
        }
        group.signal(Signal::KILL);
        let _ = group.reap_leader(&mut child).await;
        match (ended, blame) {
            (SessionEnd::Died, ExitBlame::Work) => SessionEnd::DiedOverItsWork,
            _ => ended,
        }
    }
}

/// Run the process `worker` describes until `shutdown` flips, restarting it
/// with backoff when it dies, sends an overlong line, misses its startup
/// deadline or is flagged unresponsive. An exit held against the worker's work
/// does not count toward the backoff.
pub async fn supervise<W: PipeWorker>(worker: Arc<W>, mut shutdown: watch::Receiver<bool>) {
    let mut policy = RestartPolicy::default();
    loop {
        let started = Instant::now();
        let ended = match RunningWorker::spawn(&worker).await {
            Ok(running) => {
                let ended = running.run(&*worker, &mut shutdown).await;
                match ended {
                    SessionEnd::NeverStarted => policy.record_spawn_failure(),
                    SessionEnd::Died | SessionEnd::Shutdown => {
                        policy.record_session_end(started.elapsed());
                    }
                    SessionEnd::DiedOverItsWork => {}
                }
                ended
            }
            Err(e) => {
                warn!(target: "assistd::pipe_worker", "failed to spawn the {}: {e}", W::NAME);
                policy.record_spawn_failure();
                SessionEnd::NeverStarted
            }
        };
        let restart = match ended {
            SessionEnd::Shutdown => return,
            SessionEnd::DiedOverItsWork => uncounted_restart::<W>(),
            SessionEnd::NeverStarted | SessionEnd::Died => policy.next_restart(Instant::now()),
        };
        if !wait_to_restart(restart, &*worker, &mut shutdown).await {
            return;
        }
    }
}

/// The restart after an exit the policy does not count, at the shortest backoff.
fn uncounted_restart<W: PipeWorker>() -> RestartDecision {
    info!(
        target: "assistd::pipe_worker",
        "the {} died over the same work twice; the exit is not held against it",
        W::NAME
    );
    RestartDecision::Backoff {
        delay: backoff_delay(0),
        failures: 0,
    }
}

/// Sleep out `restart`'s delay, with `worker` starting through a backoff and
/// unavailable through a slow retry; false when shutdown flips first.
async fn wait_to_restart<W: PipeWorker>(
    restart: RestartDecision,
    worker: &W,
    shutdown: &mut watch::Receiver<bool>,
) -> bool {
    let delay = match restart {
        RestartDecision::Backoff { delay, failures } => {
            info!(
                target: "assistd::pipe_worker",
                "restarting the {} in {delay:?} ({failures} failed in a row)",
                W::NAME
            );
            worker.set_not_ready(NotReady::Starting);
            delay
        }
        RestartDecision::SlowRetry { delay, cap } => {
            let reason = format!(
                "the {} {cap}; retrying every {} minutes",
                W::NAME,
                delay.as_secs() / 60
            );
            warn!(target: "assistd::pipe_worker", "{reason}");
            worker.set_not_ready(NotReady::Unavailable(reason.into()));
            delay
        }
    };
    let slept = tokio::select! {
        () = tokio::time::sleep(delay) => true,
        _ = shutdown.wait_for(|stop| *stop) => false,
    };
    if slept {
        worker.set_not_ready(NotReady::Starting);
    }
    slept
}

/// Write each message to the worker's stdin; returns once the pipe fails.
async fn write_messages<W: PipeWorker>(
    mut stdin: ChildStdin,
    mut messages: mpsc::UnboundedReceiver<W::Outgoing>,
) {
    while let Some(message) = messages.recv().await {
        let line = W::encode(&message);
        let written = async {
            stdin.write_all(line.as_bytes()).await?;
            stdin.flush().await
        };
        if let Err(e) = written.await {
            warn!(target: "assistd::pipe_worker", "{} stdin: {e}", W::NAME);
            return;
        }
    }
}

/// Hand `worker` the process's lines until its stdout closes, it sends a
/// line longer than it may, or it misses its startup deadline.
async fn read_messages<W: PipeWorker>(
    stdout: &mut BufReader<ChildStdout>,
    worker: &W,
) -> SessionEnd {
    let Some(mut next) = read_first_line::<W>(stdout).await else {
        return SessionEnd::NeverStarted;
    };
    loop {
        match next {
            Ok(Some(line)) => worker.receive(&line),
            Ok(None) => return SessionEnd::Died,
            Err(e) => {
                warn!(target: "assistd::pipe_worker", "{} stdout: {e}", W::NAME);
                return SessionEnd::Died;
            }
        }
        next = read_line(stdout, W::MAX_LINE_BYTES).await;
    }
}

/// The worker's first read, or `None` when it wrote nothing within its
/// startup deadline.
async fn read_first_line<W: PipeWorker>(
    stdout: &mut BufReader<ChildStdout>,
) -> Option<io::Result<Option<Vec<u8>>>> {
    let first = read_line(stdout, W::MAX_LINE_BYTES);
    let Some(deadline) = W::STARTUP_DEADLINE else {
        return Some(first.await);
    };
    let started = tokio::time::timeout(deadline, first).await;
    if started.is_err() {
        warn!(
            target: "assistd::pipe_worker",
            "{} wrote nothing within {deadline:?} of starting; terminating it",
            W::NAME
        );
    }
    started.ok()
}

/// The next line, `None` at end of file; a line longer than `max_bytes` is
/// an error, so the worker cannot grow the daemon.
async fn read_line<R: AsyncBufRead + Unpin>(
    stdout: &mut R,
    max_bytes: usize,
) -> io::Result<Option<Vec<u8>>> {
    let mut line = Vec::new();
    let read = stdout
        .take(max_bytes as u64 + 1)
        .read_until(b'\n', &mut line)
        .await?;
    if read == 0 {
        return Ok(None);
    }
    if line.len() > max_bytes && !line.ends_with(b"\n") {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("a line longer than {max_bytes} bytes"),
        ));
    }
    Ok(Some(line))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_line_past_the_cap_is_an_error_and_shorter_ones_are_read_whole() {
        let mut lines: &[u8] = b"short\nexactly8\n123456789\n";
        assert_eq!(
            read_line(&mut lines, 8).await.unwrap().as_deref(),
            Some(&b"short\n"[..])
        );
        assert_eq!(
            read_line(&mut lines, 8).await.unwrap().as_deref(),
            Some(&b"exactly8\n"[..])
        );
        let overlong = read_line(&mut lines, 8).await.unwrap_err();
        assert_eq!(overlong.kind(), io::ErrorKind::InvalidData);
    }

    #[tokio::test(start_paused = true)]
    async fn a_missed_deadline_fails_the_call_and_flags_the_worker_unresponsive() {
        let replies = PendingReplies::<u64, oneshot::Sender<()>>::new("test worker");
        let (outgoing, mut sent) = mpsc::unbounded_channel();
        let mut unresponsive = replies.connect(outgoing);
        let deadline = Duration::from_secs(5);

        let (answer, reply) = oneshot::channel();
        let answered = replies.call(|id| id, answer, reply, Some(deadline));
        let answering = async {
            let id = sent.recv().await.expect("the call is sent");
            let _ = replies.take(id).expect("the call is pending").send(());
        };
        assert_eq!(tokio::join!(answered, answering).0, Ok(()));
        assert!(unresponsive.try_recv().is_err());

        let (_unanswered, reply) = oneshot::channel::<()>();
        let (held, _held_reply) = oneshot::channel();
        let missed = replies.call(|id| id, held, reply, Some(deadline)).await;
        assert_eq!(missed, Err(CallError::Unresponsive { deadline }));
        assert_eq!(unresponsive.try_recv(), Ok(()));
    }
}
