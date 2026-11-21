//! Unix-socket IPC server: one newline-delimited JSON request per
//! connection, answered by a stream of events.

use std::fs::{File, Permissions, TryLockError};
use std::future::Future;
use std::io;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use rustix::event::{PollFd, PollFlags, Timespec, poll};
use rustix::io::Errno;
use rustix::process::geteuid;
use serde::Serialize;
use thiserror::Error;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::unix::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc, watch};
use tokio::task::JoinSet;
use tracing::{Instrument, debug, error, info, warn};

use assistd_ipc::socket::{SocketDirError, ensure_private_socket_dir, open_private_lock_file};
use assistd_ipc::{Event, EventBody, Hello, Request, RequestBody};
use assistd_tools::{Answer, CONFIRM_TIMEOUT, ConfirmRouter, TurnContext};
use assistd_utils::time::saturating_millis;

use crate::bus::EventSink;
use crate::recovery::drain_join_set;
use crate::state::{AppState, Requester};

const EVENT_CHANNEL_CAPACITY: usize = 32;

/// Longest one event write may block on a client that stopped reading,
/// since a stalled reader holds the agent turn lock for later queries.
const EVENT_WRITE_TIMEOUT: Duration = Duration::from_secs(10);

/// How often a connection checks whether its peer closed the socket, so
/// a client that leaves is noticed even while no event is being written.
const PEER_HANGUP_POLL_INTERVAL: Duration = Duration::from_secs(1);

/// Cap on one `query` frame: a 32 MiB image base64-encoded plus JSON
/// overhead, so a runaway client cannot OOM the daemon.
const MAX_REQUEST_BYTES: u64 = 64 * 1024 * 1024;

/// Cap on every other frame, which carries no attachments.
const MAX_CONTROL_REQUEST_BYTES: u64 = 64 * 1024;

/// Connections that may hold a frame past [`MAX_CONTROL_REQUEST_BYTES`] at once; each keeps
/// its slot until the connection ends, bounding the memory queued image queries pin.
const LARGE_FRAME_SLOTS: usize = 4;

/// How long a new connection may take to deliver its first frame.
const INITIAL_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// Connections served at once; further ones are sent an error and closed on accept.
const MAX_CONNECTIONS: usize = 128;

/// Pause after EMFILE/ENFILE from `accept()`, which would otherwise spin.
const FD_EXHAUSTION_BACKOFF: Duration = Duration::from_millis(100);

/// How long an existing socket gets to accept a probe before it is
/// treated as stale; one that hangs past this is treated as live.
const PROBE_TIMEOUT: Duration = Duration::from_secs(1);

/// Mode for the socket itself: only the daemon's user may connect.
const SOCKET_MODE: u32 = 0o600;

/// Errors produced by the socket listener and per-connection handlers.
#[derive(Debug, Error)]
pub enum SocketError {
    #[error(
        "another assistd daemon is already accepting connections at {path}; refusing to clobber \
         its socket"
    )]
    AlreadyRunning { path: PathBuf },

    #[error(
        "another assistd daemon holds the startup lock at {path}; it is still starting or already \
         running"
    )]
    AlreadyStarting { path: PathBuf },

    #[error("failed to take the startup lock at {path}: {source}")]
    StartupLock {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("failed to remove stale socket file at {path}: {source}")]
    StaleCleanup {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error(transparent)]
    SocketDir(#[from] SocketDirError),

    #[error("failed to bind unix socket at {path}: {source}")]
    Bind {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("socket I/O error: {0}")]
    Io(#[from] io::Error),

    #[error("client did not accept an event within {0:?}")]
    WriteTimeout(Duration),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Exclusive `flock` on `assistd.lock` beside the socket, held for the
/// daemon's lifetime so a second daemon is refused while this one is
/// still initialising and has not yet bound the socket.
#[derive(Debug)]
pub struct StartupLock {
    _file: File,
}

impl StartupLock {
    /// Take the lock beside the default [`assistd_ipc::socket_path`].
    pub fn acquire() -> Result<Self, SocketError> {
        Self::acquire_at(&assistd_ipc::socket_path())
    }

    /// Take the lock beside `socket_path`, creating the socket directory owner-only if
    /// needed; it is released when the guard drops or the process exits. Fails with
    /// `AlreadyStarting` when another process holds it.
    pub fn acquire_at(socket_path: &Path) -> Result<Self, SocketError> {
        let lock_path = socket_path.with_extension("lock");
        let file = open_private_lock_file(&lock_path)?;
        match file.try_lock() {
            Ok(()) => Ok(Self { _file: file }),
            Err(TryLockError::WouldBlock) => Err(SocketError::AlreadyStarting { path: lock_path }),
            Err(TryLockError::Error(source)) => Err(SocketError::StartupLock {
                path: lock_path,
                source,
            }),
        }
    }
}

/// How much of the first frame was read.
enum FrameRead {
    /// Read up to `limit` bytes; `large_slot` is held when the frame may exceed
    /// [`MAX_CONTROL_REQUEST_BYTES`].
    Read {
        limit: u64,
        large_slot: Option<OwnedSemaphorePermit>,
    },
    /// The frame outgrew [`MAX_CONTROL_REQUEST_BYTES`] while every large-frame slot was taken.
    NoLargeSlot,
}

/// Why a connection stopped forwarding events to its client.
enum ForwardEnd {
    /// Every event was written; the client still holds the socket open.
    Drained(OwnedWriteHalf),
    /// The client closed its socket before the stream ended.
    PeerHungUp,
}

/// [`serve_at`] on the default path from [`assistd_ipc::socket_path`].
pub async fn serve<F>(state: Arc<AppState>, shutdown: F) -> Result<(), SocketError>
where
    F: Future<Output = ()>,
{
    let path = assistd_ipc::socket_path();
    serve_at(&path, state, shutdown).await
}

/// Serve the IPC socket at `path` until `shutdown` resolves, then drain in-flight connections.
/// The directory must be this user's and not writable by others; the socket is owner-only.
/// A stale socket file at `path` is removed first; a live one is an error.
pub async fn serve_at<F>(path: &Path, state: Arc<AppState>, shutdown: F) -> Result<(), SocketError>
where
    F: Future<Output = ()>,
{
    if let Some(dir) = path.parent() {
        ensure_private_socket_dir(dir)?;
    }
    prepare_socket_path(path).await?;

    let bind_error = |source| SocketError::Bind {
        path: path.to_path_buf(),
        source,
    };
    let listener = UnixListener::bind(path).map_err(bind_error)?;
    std::fs::set_permissions(path, Permissions::from_mode(SOCKET_MODE)).map_err(bind_error)?;
    info!("listening on {}", path.display());

    let result = accept_loop(listener, state, shutdown).await;

    if let Err(e) = std::fs::remove_file(path) {
        warn!("failed to remove socket file at {}: {}", path.display(), e);
    }

    result
}

async fn prepare_socket_path(path: &Path) -> Result<(), SocketError> {
    match path.try_exists() {
        Ok(false) => return Ok(()),
        Ok(true) => {}
        Err(e) => {
            return Err(SocketError::StaleCleanup {
                path: path.to_path_buf(),
                source: e,
            });
        }
    }

    match tokio::time::timeout(PROBE_TIMEOUT, UnixStream::connect(path)).await {
        Ok(Ok(_stream)) => Err(SocketError::AlreadyRunning {
            path: path.to_path_buf(),
        }),
        Ok(Err(e)) => {
            warn!("removing stale socket file at {} ({})", path.display(), e);
            std::fs::remove_file(path).map_err(|source| SocketError::StaleCleanup {
                path: path.to_path_buf(),
                source,
            })
        }
        Err(_) => Err(SocketError::AlreadyRunning {
            path: path.to_path_buf(),
        }),
    }
}

async fn accept_loop<F>(
    listener: UnixListener,
    state: Arc<AppState>,
    shutdown: F,
) -> Result<(), SocketError>
where
    F: Future<Output = ()>,
{
    let mut connections: JoinSet<()> = JoinSet::new();
    let large_frames = Arc::new(Semaphore::new(LARGE_FRAME_SLOTS));
    let mut fd_exhausted = false;
    let (drain_tx, drain_rx) = watch::channel(false);

    tokio::pin!(shutdown);
    loop {
        tokio::select! {
            () = &mut shutdown => {
                info!("shutting down socket listener");
                break;
            }
            accepted = listener.accept() => {
                match accepted {
                    Ok((stream, _addr)) => {
                        if fd_exhausted {
                            info!(
                                "accept loop recovered from FD exhaustion; resuming \
                                 normal operation"
                            );
                            fd_exhausted = false;
                        }
                        if connections.len() >= MAX_CONNECTIONS {
                            warn!(
                                limit = MAX_CONNECTIONS,
                                "rejecting IPC connection: too many open connections"
                            );
                            refuse_connection(
                                &stream,
                                format!("daemon is serving its limit of {MAX_CONNECTIONS} connections"),
                            );
                        } else if peer_is_daemon_user(&stream) {
                            spawn_connection(
                                &mut connections,
                                stream,
                                state.clone(),
                                large_frames.clone(),
                                drain_rx.clone(),
                            );
                        }
                    }
                    Err(e) if is_fd_exhaustion(&e) => {
                        back_off_from_fd_exhaustion(&e, &mut fd_exhausted).await;
                    }
                    Err(e) => {
                        error!("accept error: {e}");
                    }
                }
            }
            Some(res) = connections.join_next(), if !connections.is_empty() => {
                if let Err(e) = res
                    && e.is_panic()
                {
                    error!("connection task panicked: {e}");
                }
            }
        }
    }

    drop(listener);
    let _ = drain_tx.send(true);
    state.stop_turns();
    drain_join_set(&mut connections, "connection").await;
    Ok(())
}

fn peer_is_daemon_user(stream: &UnixStream) -> bool {
    match stream.peer_cred() {
        Ok(cred) if cred.uid() == geteuid().as_raw() => true,
        Ok(cred) => {
            warn!(
                peer_uid = cred.uid(),
                "rejecting IPC connection from another user"
            );
            false
        }
        Err(e) => {
            warn!("rejecting IPC connection with unreadable peer credentials: {e}");
            false
        }
    }
}

/// Tell a client turned away on accept why, without waiting on it; a fresh socket's
/// send buffer always has room for the hello and the one event.
fn refuse_connection(stream: &UnixStream, message: String) {
    let event = Event::new(String::new(), EventBody::Error { message });
    let frame = match refusal_frame(&event) {
        Ok(frame) => frame,
        Err(e) => {
            debug!("could not serialize connection refusal: {e}");
            return;
        }
    };
    if let Err(e) = stream.try_write(frame.as_bytes()) {
        debug!("could not tell refused client why: {e}");
    }
}

fn refusal_frame(event: &Event) -> serde_json::Result<String> {
    let hello = serde_json::to_string(&Hello::current())?;
    let refusal = serde_json::to_string(event)?;
    Ok(format!("{hello}\n{refusal}\n"))
}

fn peer_pid(stream: &UnixStream) -> Option<u32> {
    let pid = stream.peer_cred().ok()?.pid()?;
    u32::try_from(pid).ok()
}

fn spawn_connection(
    connections: &mut JoinSet<()>,
    stream: UnixStream,
    state: Arc<AppState>,
    large_frames: Arc<Semaphore>,
    drain: watch::Receiver<bool>,
) {
    connections.spawn(async move {
        if let Err(e) = handle_connection(stream, state, large_frames, drain).await {
            error!("connection error: {e}");
        }
    });
}

/// Warns only on the first failure of a run, then sleeps.
async fn back_off_from_fd_exhaustion(err: &io::Error, fd_exhausted: &mut bool) {
    if !*fd_exhausted {
        warn!(
            error = %err,
            backoff_ms = saturating_millis(FD_EXHAUSTION_BACKOFF),
            "accept failed: file-descriptor limit reached; backing \
             off until in-flight connections release descriptors. \
             Repeat occurrences suppressed until recovery."
        );
        *fd_exhausted = true;
    }
    tokio::time::sleep(FD_EXHAUSTION_BACKOFF).await;
}

/// Matches the raw errno because EMFILE maps to the unstable
/// `io::ErrorKind::Uncategorized`.
fn is_fd_exhaustion(err: &io::Error) -> bool {
    matches!(
        err.raw_os_error().map(Errno::from_raw_os_error),
        Some(Errno::MFILE | Errno::NFILE)
    )
}

async fn handle_connection(
    stream: UnixStream,
    state: Arc<AppState>,
    large_frames: Arc<Semaphore>,
    mut drain: watch::Receiver<bool>,
) -> Result<(), SocketError> {
    let peer_pid = peer_pid(&stream);
    let (read_half, mut write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);
    match write_frame(&mut write_half, &Hello::current()).await {
        Err(SocketError::Io(e)) if is_peer_gone(&e) => {
            debug!("client disconnected before the hello");
            return Ok(());
        }
        result => result?,
    }
    let Some((req, _large_slot)) =
        read_initial_request(&mut reader, &mut write_half, large_frames).await?
    else {
        return Ok(());
    };

    let (tx, rx) = mpsc::channel::<Event>(EVENT_CHANNEL_CAPACITY);
    let span = tracing::info_span!("ipc", id = %req.id, req = req.kind());
    let (prompt_tx, prompt_rx) = mpsc::channel::<Event>(EVENT_CHANNEL_CAPACITY);
    let router = ConfirmRouter::new(
        req.id.clone(),
        prompt_tx,
        CONFIRM_TIMEOUT,
        state.safety_judge(),
    );
    let prompts = forward_prompts(prompt_rx, state.events_of(&req, tx.clone()));
    let is_subscribe = matches!(req.body, RequestBody::Subscribe { .. });

    let requester = Requester {
        pid: peer_pid,
        turn: TurnContext::answered_by(Arc::clone(&router)),
    };
    let dispatch_fut = Box::pin(async move {
        tokio::select! {
            res = state.dispatch(req, requester, tx) => res,
            () = prompts => Ok(()),
        }
    });
    let forward_fut = forward_events(rx, write_half);
    let read_fut = route_confirm_responses(reader, router);

    let dispatch_and_read = async {
        tokio::pin!(dispatch_fut);
        tokio::pin!(read_fut);
        tokio::select! {
            res = &mut dispatch_fut => res,
            () = &mut read_fut => dispatch_fut.await,
        }
    };

    let dispatch_and_read = async move {
        if is_subscribe {
            tokio::select! {
                res = dispatch_and_read => res,
                _ = drain.wait_for(|draining| *draining) => Ok(()),
            }
        } else {
            dispatch_and_read.await
        }
    };

    let (dispatch_res, forward_res) = async { tokio::join!(dispatch_and_read, forward_fut) }
        .instrument(span)
        .await;

    if let Err(e) = dispatch_res {
        error!("dispatch error: {e}");
    }
    match forward_res? {
        ForwardEnd::Drained(mut write_half) => write_half.shutdown().await?,
        ForwardEnd::PeerHungUp => debug!("client closed its socket before the stream ended"),
    }
    Ok(())
}

/// Read and parse the connection's first frame, with the large-frame slot it
/// holds, if any. `None` means the client left, missed
/// [`INITIAL_REQUEST_TIMEOUT`], or was already sent an error and closed.
async fn read_initial_request(
    reader: &mut BufReader<OwnedReadHalf>,
    write_half: &mut OwnedWriteHalf,
    large_frames: Arc<Semaphore>,
) -> Result<Option<(Request, Option<OwnedSemaphorePermit>)>, SocketError> {
    let mut line = Vec::new();
    let Ok(frame) = tokio::time::timeout(
        INITIAL_REQUEST_TIMEOUT,
        read_request_frame(reader, &mut line, large_frames),
    )
    .await
    else {
        warn!(
            timeout_secs = INITIAL_REQUEST_TIMEOUT.as_secs(),
            "client sent no complete request in time; closing"
        );
        return Ok(None);
    };
    let frame = frame?;
    if line.is_empty() {
        debug!("client disconnected without sending a request");
        return Ok(None);
    }
    let rejection = match frame {
        FrameRead::NoLargeSlot => format!(
            "{LARGE_FRAME_SLOTS} requests over {MAX_CONTROL_REQUEST_BYTES} bytes are already \
             in flight; retry once one finishes"
        ),
        FrameRead::Read { limit, .. } if !line.ends_with(b"\n") => {
            format!("request exceeded {limit}-byte limit")
        }
        FrameRead::Read { large_slot, .. } => {
            match serde_json::from_slice::<Request>(line.trim_ascii()) {
                Ok(req) if large_slot.is_some() && !is_query(&req) => {
                    format!("only a query may exceed {MAX_CONTROL_REQUEST_BYTES} bytes")
                }
                Ok(req) => return Ok(Some((req, large_slot))),
                Err(e) => format!("invalid request: {e}"),
            }
        }
    };
    let err = Event::new(String::new(), EventBody::Error { message: rejection });
    write_frame(write_half, &err).await?;
    write_half.shutdown().await?;
    Ok(None)
}

fn is_query(req: &Request) -> bool {
    matches!(req.body, RequestBody::Query { .. })
}

/// Read one frame into `line`. A frame that fills [`MAX_CONTROL_REQUEST_BYTES`]
/// reads on to [`MAX_REQUEST_BYTES`] only with a large-frame slot.
async fn read_request_frame(
    reader: &mut BufReader<OwnedReadHalf>,
    line: &mut Vec<u8>,
    large_frames: Arc<Semaphore>,
) -> io::Result<FrameRead> {
    (&mut *reader)
        .take(MAX_CONTROL_REQUEST_BYTES)
        .read_until(b'\n', line)
        .await?;
    let filled_control_cap =
        u64::try_from(line.len()).is_ok_and(|read| read >= MAX_CONTROL_REQUEST_BYTES);
    if line.ends_with(b"\n") || !filled_control_cap {
        return Ok(FrameRead::Read {
            limit: MAX_CONTROL_REQUEST_BYTES,
            large_slot: None,
        });
    }
    let Ok(large_slot) = large_frames.try_acquire_owned() else {
        return Ok(FrameRead::NoLargeSlot);
    };
    reader
        .take(MAX_REQUEST_BYTES - MAX_CONTROL_REQUEST_BYTES)
        .read_until(b'\n', line)
        .await?;
    Ok(FrameRead::Read {
        limit: MAX_REQUEST_BYTES,
        large_slot: Some(large_slot),
    })
}

/// Send the prompts a connection's router raises to its client through `events`, like
/// every other event of the request; never returns.
async fn forward_prompts(mut prompts: mpsc::Receiver<Event>, events: EventSink) {
    while let Some(prompt) = prompts.recv().await {
        let _ = events.send(prompt).await;
    }
    std::future::pending::<()>().await;
}

/// Write every dispatched event to the client. Returns early, dropping the receiver, once
/// the client hangs up.
async fn forward_events(
    mut rx: mpsc::Receiver<Event>,
    mut write_half: OwnedWriteHalf,
) -> Result<ForwardEnd, SocketError> {
    let mut hangup_poll = tokio::time::interval(PEER_HANGUP_POLL_INTERVAL);
    loop {
        tokio::select! {
            received = rx.recv() => {
                let Some(event) = received else {
                    return Ok(ForwardEnd::Drained(write_half));
                };
                match write_frame(&mut write_half, &event).await {
                    Err(SocketError::Io(e)) if is_peer_gone(&e) => {
                        return Ok(ForwardEnd::PeerHungUp);
                    }
                    result => result?,
                }
            }
            _ = hangup_poll.tick() => {
                if peer_hung_up(write_half.as_ref()) {
                    return Ok(ForwardEnd::PeerHungUp);
                }
            }
        }
    }
}

/// Whether the peer closed its end of `stream`. A client that only shut
/// its write side, as one-shot clients do, does not count.
fn peer_hung_up(stream: &UnixStream) -> bool {
    let mut fds = [PollFd::new(stream, PollFlags::empty())];
    match poll(&mut fds, Some(&Timespec::default())) {
        Ok(_) => fds[0].revents().contains(PollFlags::HUP),
        Err(e) => {
            debug!("peer hangup poll failed: {e}");
            false
        }
    }
}

/// Whether a write failed because the client already closed its socket.
fn is_peer_gone(err: &io::Error) -> bool {
    matches!(
        err.kind(),
        io::ErrorKind::BrokenPipe | io::ErrorKind::ConnectionReset
    )
}

/// Route mid-stream `ConfirmResponse` and `SetPermissionMode` frames to
/// `router` until the client closes its write side, then close the router.
async fn route_confirm_responses(mut reader: BufReader<OwnedReadHalf>, router: Arc<ConfirmRouter>) {
    let mut buf = String::new();
    loop {
        buf.clear();
        match (&mut reader)
            .take(MAX_CONTROL_REQUEST_BYTES)
            .read_line(&mut buf)
            .await
        {
            Ok(0) => break,
            Ok(_) => {}
            Err(e) => {
                debug!("connection read loop ended: {e}");
                break;
            }
        }
        if !buf.ends_with('\n') {
            warn!(
                bytes = buf.len(),
                "mid-stream request exceeded {MAX_CONTROL_REQUEST_BYTES}-byte limit; closing \
                 read side"
            );
            break;
        }
        let trimmed = buf.trim();
        if !trimmed.is_empty() {
            route_mid_stream_line(trimmed, &router);
        }
    }
    router.close();
}

fn route_mid_stream_line(line: &str, router: &ConfirmRouter) {
    let request = match serde_json::from_str::<Request>(line) {
        Ok(request) => request,
        Err(e) => {
            warn!(error = %e, "invalid mid-stream JSON; ignoring line");
            return;
        }
    };
    let kind = request.kind();
    match request.body {
        RequestBody::ConfirmResponse {
            confirm_id,
            allow,
            always,
            message,
        } => {
            let answer = Answer::from_wire(allow, always, message.as_deref());
            if let Err(e) = router.route_response(&confirm_id, answer) {
                warn!(confirm_id = %confirm_id, reason = %e, "unmatched ConfirmResponse");
            }
        }
        RequestBody::SetPermissionMode { mode, plan } => {
            info!(?mode, plan, "permission mode changed");
            router.set_mode(mode);
            router.set_planning(plan);
        }
        RequestBody::PlanVerdict { review_id, verdict } => {
            if let Err(e) = router.route_verdict(&review_id, verdict) {
                warn!(review_id = %review_id, reason = %e, "unmatched PlanVerdict");
            }
        }
        _ => {
            warn!(
                kind,
                "unexpected mid-stream request; only ConfirmResponse, SetPermissionMode and \
                 PlanVerdict are honored after the initial request"
            );
        }
    }
}

async fn write_frame(
    write_half: &mut OwnedWriteHalf,
    frame: &impl Serialize,
) -> Result<(), SocketError> {
    let mut out = serde_json::to_string(frame)?;
    out.push('\n');
    let write = async {
        write_half.write_all(out.as_bytes()).await?;
        write_half.flush().await
    };
    tokio::time::timeout(EVENT_WRITE_TIMEOUT, write)
        .await
        .map_err(|_| SocketError::WriteTimeout(EVENT_WRITE_TIMEOUT))??;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::os::unix::fs::MetadataExt;
    use std::sync::atomic::{AtomicBool, Ordering};

    use tokio::sync::oneshot;

    use assistd_config::Config;
    use assistd_ipc::{PresenceState, PresenceTarget};
    use assistd_llm::{Gate, ScriptedBackend, Step, ToolCall};

    use super::*;
    use crate::Subsystems;
    use crate::presence::tests::stub_presence;
    use crate::state::tests::TestState;

    async fn test_state() -> TestState {
        state_with_backend(Arc::new(assistd_llm::EchoBackend::new())).await
    }

    fn scripted(steps: impl IntoIterator<Item = Step>) -> Arc<ScriptedBackend> {
        Arc::new(ScriptedBackend::new(steps))
    }

    /// A step that calls tool `name` with no arguments.
    fn calling(id: &str, name: &str) -> Step {
        Step::tool_calls([ToolCall {
            id: id.into(),
            name: name.into(),
            arguments: serde_json::json!({}),
        }])
    }

    /// A step that streams 1 KiB deltas until nobody reads them.
    fn flood() -> Step {
        Step::default().flooding("x".repeat(1024))
    }

    async fn state_with_backend(backend: Arc<dyn assistd_llm::LlmBackend>) -> TestState {
        state_with(backend, Arc::new(assistd_tools::ToolRegistry::default())).await
    }

    async fn state_with(
        backend: Arc<dyn assistd_llm::LlmBackend>,
        tools: Arc<assistd_tools::ToolRegistry>,
    ) -> TestState {
        let subsystems = Subsystems::new(
            backend,
            stub_presence(PresenceTarget::Active),
            tools,
            assistd_voice::VoiceManager::new(true),
        );
        TestState::unsaved(Config::default(), subsystems).await
    }

    /// A `serve_at` task on a socket in a fresh temp dir.
    struct TestServer {
        _dir: tempfile::TempDir,
        path: PathBuf,
        shutdown: oneshot::Sender<()>,
        task: tokio::task::JoinHandle<()>,
    }

    impl TestServer {
        async fn start(state: &TestState) -> Self {
            let state = Arc::clone(state);
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("assistd.sock");
            let (shutdown, rx) = oneshot::channel::<()>();
            let server_path = path.clone();
            let task = tokio::spawn(async move {
                serve_at(&server_path, state, async {
                    let _ = rx.await;
                })
                .await
                .unwrap();
            });
            wait_for_listener(&path).await;
            Self {
                _dir: dir,
                path,
                shutdown,
                task,
            }
        }

        async fn stop(self) {
            self.shutdown.send(()).unwrap();
            self.task.await.unwrap();
        }
    }

    async fn with_server<F, Fut, R>(state: &TestState, body: F) -> R
    where
        F: FnOnce(PathBuf) -> Fut,
        Fut: Future<Output = R>,
    {
        let server = TestServer::start(state).await;
        let out = body(server.path.clone()).await;
        server.stop().await;
        out
    }

    async fn wait_for_listener(path: &Path) {
        for _ in 0..200 {
            if UnixStream::connect(path).await.is_ok() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("listener at {} did not become ready", path.display());
    }

    async fn read_event(reader: &mut BufReader<OwnedReadHalf>) -> Option<Event> {
        let mut line = String::new();
        let n = reader.read_line(&mut line).await.unwrap();
        (n > 0).then(|| serde_json::from_str(line.trim()).unwrap())
    }

    async fn read_until_terminal(reader: &mut BufReader<OwnedReadHalf>) -> Vec<Event> {
        let mut events = Vec::new();
        while let Some(event) = read_event(reader).await {
            let terminal = event.is_terminal();
            events.push(event);
            if terminal {
                break;
            }
        }
        events
    }

    /// Write `lines` as newline-delimited frames, optionally close the write
    /// side, and return both halves.
    async fn open_connection(
        path: &Path,
        lines: &[&str],
        close_write: bool,
    ) -> (OwnedWriteHalf, BufReader<OwnedReadHalf>) {
        let stream = UnixStream::connect(path).await.unwrap();
        let (read, mut write) = stream.into_split();
        for line in lines {
            write.write_all(line.as_bytes()).await.unwrap();
            write.write_all(b"\n").await.unwrap();
        }
        if close_write {
            write.shutdown().await.unwrap();
        } else {
            write.flush().await.unwrap();
        }
        let mut reader = BufReader::new(read);
        read_hello(&mut reader).await;
        (write, reader)
    }

    async fn read_hello(reader: &mut BufReader<OwnedReadHalf>) {
        let mut line = String::new();
        reader.read_line(&mut line).await.unwrap();
        assert_eq!(
            serde_json::from_str::<Hello>(line.trim()).unwrap(),
            Hello::current(),
            "the daemon's first line must be its hello"
        );
    }

    async fn send_request_collect_events(path: &Path, body: &str) -> Vec<Event> {
        let (_write, mut reader) = open_connection(path, &[body], true).await;
        read_until_terminal(&mut reader).await
    }

    /// Wait until `n` subscribers are attached to the event bus, so events
    /// published afterwards are guaranteed to reach them.
    async fn wait_for_subscribers(state: &AppState, n: usize) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while state.bus.sender().receiver_count() < n {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("subscriber never attached to the event bus");
    }

    fn delta(id: &str, text: &str) -> Event {
        Event::new(id, EventBody::Delta { text: text.into() })
    }

    fn done(id: &str) -> Event {
        Event::new(id, EventBody::Done)
    }

    #[tokio::test]
    async fn echoes_query_text_as_delta_then_done() {
        let server = TestServer::start(&test_state().await).await;
        let events = send_request_collect_events(
            &server.path,
            r#"{"type":"query","id":"req-1","text":"ping"}"#,
        )
        .await;
        assert_eq!(events, [delta("req-1", "ping"), done("req-1")]);

        let path = server.path.clone();
        server.stop().await;
        assert!(!path.exists(), "socket file should be removed on shutdown");
    }

    #[tokio::test]
    async fn concurrent_connections_sharing_a_request_id_stay_isolated() {
        with_server(&test_state().await, |path| async move {
            let mut handles = Vec::new();
            for i in 0..16 {
                let p = path.clone();
                handles.push(tokio::spawn(async move {
                    let text = format!("msg-{i}");
                    let body = format!(r#"{{"type":"query","id":"shared","text":"{text}"}}"#);
                    let events = send_request_collect_events(&p, &body).await;
                    assert_eq!(events, [delta("shared", &text), done("shared")], "conn {i}");
                }));
            }
            for h in handles {
                h.await.unwrap();
            }
        })
        .await;
    }

    #[tokio::test]
    async fn first_frame_may_pass_the_control_cap_only_as_a_query() {
        let past_control_cap = "a".repeat(usize::try_from(MAX_CONTROL_REQUEST_BYTES).unwrap());
        with_server(&test_state().await, |path| async move {
            let query = format!(r#"{{"text":"{past_control_cap}","type":"query","id":"q"}}"#);
            let events = send_request_collect_events(&path, &query).await;
            assert_eq!(events, [delta("q", &past_control_cap), done("q")]);

            let save = format!(
                r#"{{"id":"m","type":"memory_save","key":"k","value":"{past_control_cap}"}}"#
            );
            let events = send_request_collect_events(&path, &save).await;
            assert_eq!(
                events,
                [Event::new(
                    String::new(),
                    EventBody::Error {
                        message: format!(
                            "only a query may exceed {MAX_CONTROL_REQUEST_BYTES} bytes"
                        )
                    }
                )]
            );
        })
        .await;
    }

    #[tokio::test]
    async fn frame_past_the_request_cap_is_rejected() {
        let event = with_server(&test_state().await, |path| async move {
            let stream = UnixStream::connect(&path).await.unwrap();
            let (read, mut write) = stream.into_split();
            write_past_cap(&mut write, MAX_REQUEST_BYTES).await;
            let mut reader = BufReader::new(read);
            read_hello(&mut reader).await;
            read_event(&mut reader).await
        })
        .await;
        assert_eq!(
            event,
            Some(Event::new(
                String::new(),
                EventBody::Error {
                    message: format!("request exceeded {MAX_REQUEST_BYTES}-byte limit")
                }
            ))
        );
    }

    async fn write_past_cap(write: &mut OwnedWriteHalf, cap: u64) {
        let chunk = vec![b'a'; 1024 * 1024];
        let mut remaining = usize::try_from(cap).expect("request cap fits usize") + 1;
        while remaining > 0 {
            let n = remaining.min(chunk.len());
            if write.write_all(&chunk[..n]).await.is_err() {
                break;
            }
            remaining -= n;
        }
        let _ = write.shutdown().await;
    }

    #[tokio::test]
    async fn malformed_request_returns_error_event() {
        with_server(&test_state().await, |path| async move {
            let events = send_request_collect_events(&path, "not json").await;
            assert!(
                matches!(
                    events.as_slice(),
                    [Event { id, body: EventBody::Error { message } }]
                        if id.is_empty() && message.starts_with("invalid request: ")
                ),
                "{events:?}"
            );
        })
        .await;
    }

    #[tokio::test]
    async fn second_serve_refuses_to_clobber_live_socket() {
        let server = TestServer::start(&test_state().await).await;

        let second = test_state().await;
        let err = serve_at(
            &server.path,
            Arc::clone(&second),
            std::future::pending::<()>(),
        )
        .await
        .expect_err("second serve_at must fail while first is alive");
        assert!(
            matches!(&err, SocketError::AlreadyRunning { path } if *path == server.path),
            "{err:?}"
        );
        assert!(
            server.path.exists(),
            "live socket must remain after a refused second-bind attempt"
        );

        let events =
            send_request_collect_events(&server.path, r#"{"type":"query","id":"q","text":"ok"}"#)
                .await;
        assert_eq!(events.last(), Some(&done("q")));

        server.stop().await;
    }

    #[test]
    fn startup_lock_refuses_second_holder_until_first_drops() {
        let temp = tempfile::tempdir().unwrap();
        let socket_path = temp.path().join("assistd.sock");
        let lock_path = socket_path.with_extension("lock");

        let first = StartupLock::acquire_at(&socket_path).unwrap();
        let err = StartupLock::acquire_at(&socket_path).expect_err("second acquire must fail");
        assert!(
            matches!(&err, SocketError::AlreadyStarting { path } if *path == lock_path),
            "{err:?}"
        );

        drop(first);
        StartupLock::acquire_at(&socket_path).expect("lock must be free once the guard drops");
    }

    #[test]
    fn startup_lock_creates_owner_only_dir_and_file() {
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path().join("assistd-1000");
        let socket_path = dir.join("assistd.sock");

        let _lock = StartupLock::acquire_at(&socket_path).unwrap();

        assert_eq!(mode_of(&dir), 0o700);
        assert_eq!(mode_of(&socket_path.with_extension("lock")), 0o600);
    }

    fn mode_of(path: &Path) -> u32 {
        std::fs::symlink_metadata(path).unwrap().mode() & 0o777
    }

    #[tokio::test]
    async fn socket_is_owner_only() {
        let server = TestServer::start(&test_state().await).await;
        assert_eq!(mode_of(&server.path), SOCKET_MODE);
        server.stop().await;
    }

    #[tokio::test]
    async fn creates_missing_socket_dir_owner_only() {
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path().join("assistd-1000");
        let path = dir.join("assistd.sock");
        let server_path = path.clone();
        let state = test_state().await;
        let served = Arc::clone(&state);
        let (shutdown, rx) = oneshot::channel::<()>();
        let task = tokio::spawn(async move {
            serve_at(&server_path, served, async {
                let _ = rx.await;
            })
            .await
            .unwrap();
        });
        wait_for_listener(&path).await;

        assert_eq!(mode_of(&dir), 0o700);

        shutdown.send(()).unwrap();
        task.await.unwrap();
    }

    #[tokio::test]
    async fn refuses_socket_dir_writable_by_others() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::set_permissions(dir.path(), Permissions::from_mode(0o777)).unwrap();

        let state = test_state().await;
        let err = serve_at(
            &dir.path().join("assistd.sock"),
            Arc::clone(&state),
            std::future::pending::<()>(),
        )
        .await
        .expect_err("a world-writable socket dir must be refused");
        assert!(
            matches!(err, SocketError::SocketDir(SocketDirError::Unsafe { problem, .. }) if problem == "writable by other users"),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn refuses_socket_dir_that_is_a_symlink() {
        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("target");
        std::fs::create_dir(&target).unwrap();
        let link = temp.path().join("link");
        std::os::unix::fs::symlink(&target, &link).unwrap();

        let state = test_state().await;
        let err = serve_at(
            &link.join("assistd.sock"),
            Arc::clone(&state),
            std::future::pending::<()>(),
        )
        .await
        .expect_err("a symlinked socket dir must be refused");
        assert!(
            matches!(err, SocketError::SocketDir(SocketDirError::Unsafe { problem, .. }) if problem == "not a directory"),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn shutdown_interrupts_an_in_flight_turn_and_keeps_its_partial_reply() {
        let backend = scripted([Step::reply("The answer is ").waiting_at(&Gate::new())]);
        let state = state_with_backend(backend.clone()).await;
        let server = TestServer::start(&state).await;
        let (_write, mut reader) = open_connection(
            &server.path,
            &[r#"{"type":"query","id":"s1","text":"why?"}"#],
            true,
        )
        .await;
        let first = read_event(&mut reader).await.expect("first Delta");
        assert_eq!(first, delta("s1", "The answer is "));

        let stopping = tokio::spawn(server.stop());
        let rest = read_until_terminal(&mut reader).await;
        assert_eq!(rest.last(), Some(&done("s1")), "{rest:?}");
        tokio::time::timeout(Duration::from_secs(2), stopping)
            .await
            .expect("shutdown waited out the grace period on an interrupted turn")
            .unwrap();

        assert_eq!(
            backend.seen().interrupted_replies,
            ["The answer is\n\n[interrupted]"]
        );
    }

    /// Collect subscriber events until the stream goes quiet for `idle`.
    /// Stops early past a cap so a feedback loop fails instead of hanging.
    async fn drain_subscribe(reader: &mut BufReader<OwnedReadHalf>, idle: Duration) -> Vec<Event> {
        const CAP: usize = 64;
        let mut out = Vec::new();
        while out.len() < CAP {
            match tokio::time::timeout(idle, read_event(reader)).await {
                Ok(Some(ev)) => out.push(ev),
                Ok(None) | Err(_) => break,
            }
        }
        out
    }

    #[tokio::test]
    async fn subscribe_receives_each_query_event_from_other_client_once() {
        let state = &test_state().await;
        with_server(state, |path| async move {
            let (_write_a, mut read_a) = open_connection(
                &path,
                &[r#"{"type":"subscribe","id":"sub-1","filter":{"kinds":[]}}"#],
                false,
            )
            .await;
            wait_for_subscribers(state, 1).await;

            let b_events =
                send_request_collect_events(&path, r#"{"type":"query","id":"q-1","text":"hello"}"#)
                    .await;
            assert_eq!(b_events, [delta("q-1", "hello"), done("q-1")]);

            let a_events = drain_subscribe(&mut read_a, Duration::from_millis(200)).await;
            let count = |want: &Event| a_events.iter().filter(|e| *e == want).count();
            assert_eq!(count(&delta("q-1", "hello")), 1, "{a_events:?}");
            assert_eq!(count(&done("q-1")), 1, "{a_events:?}");
            assert!(
                a_events.contains(&Event::new(
                    "q-1",
                    EventBody::LastDelta {
                        text: "hello".into()
                    }
                )),
                "subscriber missed LastDelta: {a_events:?}"
            );
        })
        .await;
    }

    #[tokio::test]
    async fn subscribe_filter_rejects_unmatched_kinds() {
        let state = &test_state().await;
        with_server(state, |path| async move {
            let (_write_a, mut read_a) = open_connection(
                &path,
                &[r#"{"type":"subscribe","id":"sub-2","filter":{"kinds":["presence"]}}"#],
                false,
            )
            .await;
            wait_for_subscribers(state, 1).await;

            send_request_collect_events(&path, r#"{"type":"query","id":"q-2","text":"hi"}"#).await;
            send_request_collect_events(
                &path,
                r#"{"type":"set_presence","id":"sp-1","target":"sleeping"}"#,
            )
            .await;

            let a_events = drain_subscribe(&mut read_a, Duration::from_millis(200)).await;
            let presence = |id: &str, state| Event::new(id, EventBody::Presence { state });
            assert_eq!(
                a_events.first(),
                Some(&presence("sub-2", PresenceState::Active)),
                "a subscription opens with the current state: {a_events:?}"
            );
            assert!(
                a_events[1..].contains(&presence("sub-2", PresenceState::Sleeping)),
                "subscriber missed the presence change: {a_events:?}"
            );
            assert!(
                a_events
                    .iter()
                    .all(|ev| matches!(&ev.body, EventBody::Presence { .. })),
                "{a_events:?}"
            );
        })
        .await;
    }

    #[tokio::test]
    async fn client_sees_eof_promptly_after_terminal_event() {
        with_server(&test_state().await, |path| async move {
            let (_write, mut reader) = open_connection(
                &path,
                &[r#"{"type":"query","id":"eof-1","text":"ping"}"#],
                true,
            )
            .await;
            let events = read_until_terminal(&mut reader).await;
            assert_eq!(events.last(), Some(&done("eof-1")));

            let next = tokio::time::timeout(Duration::from_secs(1), read_event(&mut reader))
                .await
                .expect("connection stayed open after terminal event");
            assert_eq!(next, None);
        })
        .await;
    }

    #[tokio::test]
    async fn shutdown_closes_idle_subscriber_without_waiting_out_grace() {
        let state = test_state().await;
        let server = TestServer::start(&state).await;
        let (_write, mut reader) = open_connection(
            &server.path,
            &[r#"{"type":"subscribe","id":"sub-drain","filter":{"kinds":[]}}"#],
            false,
        )
        .await;
        wait_for_subscribers(&state, 1).await;
        assert!(matches!(
            read_event(&mut reader).await,
            Some(Event {
                body: EventBody::TaskQueue { .. },
                ..
            })
        ));

        tokio::time::timeout(Duration::from_secs(2), server.stop())
            .await
            .expect("server held by idle subscriber past drain");

        let greeting_then_eof = async { while read_event(&mut reader).await.is_some() {} };
        tokio::time::timeout(Duration::from_secs(1), greeting_then_eof)
            .await
            .expect("subscriber not released on shutdown");
    }

    /// Tool that runs the production IPC gate on a request the given modes may approve unasked,
    /// and reports its verdict.
    #[derive(Debug)]
    struct GatedTool(assistd_tools::Unattended);

    #[async_trait::async_trait]
    impl assistd_tools::Tool for GatedTool {
        fn name(&self) -> &'static str {
            "gated"
        }

        fn description(&self) -> Cow<'_, str> {
            "asks for confirmation".into()
        }

        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }

        async fn invoke(
            &self,
            turn: &TurnContext,
            _args: serde_json::Value,
        ) -> Result<assistd_tools::ToolOutput, assistd_tools::ToolError> {
            let approved = assistd_tools::ConfirmationGate::confirm(
                &assistd_tools::IpcConfirmationGate,
                turn,
                assistd_tools::ConfirmationRequest {
                    tool: "bash".into(),
                    script: "rm -rf /tmp/x".into(),
                    matched_pattern: "rm -rf".into(),
                    summary: assistd_tools::Summary::of("a shell command"),
                    always_allow: Vec::new(),
                    risk: assistd_tools::Risk::Elevated,
                    unattended: self.0.clone(),
                    source: assistd_tools::ConfirmSource::Main,
                },
            )
            .await;
            let approved = approved != assistd_tools::Approval::Deny;
            Ok(assistd_tools::ToolOutput::text(
                if approved { "approved" } else { "denied" },
                Duration::ZERO,
            ))
        }
    }

    async fn gated_tool_state() -> TestState {
        gated_tool_state_at(assistd_tools::Unattended::Never).await
    }

    async fn gated_tool_state_at(unattended: assistd_tools::Unattended) -> TestState {
        let mut tools = assistd_tools::ToolRegistry::new();
        tools.register(GatedTool(unattended));
        state_with(scripted([calling("call-1", "gated")]), Arc::new(tools)).await
    }

    fn tool_output(ev: &Event) -> Option<&str> {
        match &ev.body {
            EventBody::ToolResult { result, .. } => Some(result.output.as_str()),
            _ => None,
        }
    }

    #[tokio::test]
    async fn dialog_client_answer_reaches_gate_in_spawned_agent_turn() {
        with_server(&gated_tool_state().await, |path| async move {
            let (mut write, mut reader) =
                open_connection(&path, &[r#"{"type":"query","id":"q1","text":"go"}"#], false).await;

            let mut outputs = Vec::new();
            let run = async {
                while let Some(ev) = read_event(&mut reader).await {
                    if let Event { id, body: EventBody::ConfirmRequest { confirm_id, .. } } = &ev {
                        assert_eq!(id, "q1");
                        let answer = format!(
                            "{{\"type\":\"confirm_response\",\"id\":\"cr\",\"confirm_id\":\"{confirm_id}\",\"allow\":true}}\n"
                        );
                        write.write_all(answer.as_bytes()).await.unwrap();
                    }
                    if let Some(out) = tool_output(&ev) {
                        outputs.push(out.to_string());
                    }
                    if ev.is_terminal() {
                        break;
                    }
                }
            };
            tokio::time::timeout(Duration::from_secs(10), run)
                .await
                .expect("turn must finish once the prompt is answered");
            assert_eq!(outputs, ["approved"]);
        })
        .await;
    }

    #[tokio::test]
    async fn accept_edits_frame_approves_a_scoped_edit_without_a_prompt() {
        with_server(
            &gated_tool_state_at(assistd_tools::Unattended::ScopedEdit).await,
            |path| async move {
                let (_write, mut reader) = open_connection(
                    &path,
                    &[
                        r#"{"type":"query","id":"q1","text":"go"}"#,
                        r#"{"type":"set_permission_mode","id":"m","mode":"accept_edits"}"#,
                    ],
                    false,
                )
                .await;
                let events =
                    tokio::time::timeout(Duration::from_secs(10), read_until_terminal(&mut reader))
                        .await
                        .expect("a routine command must not wait on a prompt");
                assert!(
                    !events
                        .iter()
                        .any(|ev| matches!(&ev.body, EventBody::ConfirmRequest { .. })),
                    "{events:?}"
                );
                let outputs: Vec<&str> = events.iter().filter_map(tool_output).collect();
                assert_eq!(outputs, ["approved"]);
            },
        )
        .await;
    }

    #[tokio::test]
    async fn one_shot_client_eof_denies_prompt_without_waiting() {
        with_server(&gated_tool_state().await, |path| async move {
            let (_write, mut reader) =
                open_connection(&path, &[r#"{"type":"query","id":"q1","text":"go"}"#], true).await;
            let events =
                tokio::time::timeout(Duration::from_secs(10), read_until_terminal(&mut reader))
                    .await
                    .expect("a client that closed its write side must not park the turn");
            let outputs: Vec<&str> = events.iter().filter_map(tool_output).collect();
            assert_eq!(outputs, ["denied"]);
        })
        .await;
    }

    /// Sends a query and returns the connection after its first event has
    /// arrived, so the caller controls when the client goes away.
    async fn open_query_and_read_first_event(
        path: &Path,
        id: &str,
    ) -> (Event, BufReader<OwnedReadHalf>, OwnedWriteHalf) {
        let body = format!(r#"{{"type":"query","id":"{id}","text":"go"}}"#);
        let (write, mut reader) = open_connection(path, &[&body], false).await;
        let event = read_event(&mut reader)
            .await
            .unwrap_or_else(|| panic!("expected a first event for {id}"));
        (event, reader, write)
    }

    #[tokio::test]
    async fn client_disconnect_mid_turn_releases_the_turn_lock() {
        with_server(
            &state_with_backend(scripted([flood(), flood()])).await,
            |path| async move {
                let (first, reader, write) = open_query_and_read_first_event(&path, "gone").await;
                assert!(
                    matches!(first.body, EventBody::Delta { .. }),
                    "got {first:?}"
                );
                drop(reader);
                drop(write);

                let (next, _reader, _write) = tokio::time::timeout(
                    Duration::from_secs(5),
                    open_query_and_read_first_event(&path, "after"),
                )
                .await
                .expect("second query queued behind a turn whose client had disconnected");
                assert!(matches!(next.body, EventBody::Delta { .. }), "got {next:?}");
            },
        )
        .await;
    }

    #[tokio::test]
    async fn client_closing_mid_stream_is_not_a_connection_error() {
        let (server, client) = UnixStream::pair().unwrap();
        let (client_read, mut client_write) = client.into_split();
        client_write
            .write_all(b"{\"type\":\"query\",\"id\":\"bye\",\"text\":\"go\"}\n")
            .await
            .unwrap();
        let state = state_with_backend(scripted([flood()])).await;
        let (_drain_tx, drain_rx) = watch::channel(false);
        let large_frames = Arc::new(Semaphore::new(LARGE_FRAME_SLOTS));
        let connection = tokio::spawn(handle_connection(
            server,
            Arc::clone(&state),
            large_frames,
            drain_rx,
        ));

        let mut reader = BufReader::new(client_read);
        read_hello(&mut reader).await;
        assert!(read_event(&mut reader).await.is_some());
        drop(reader);
        drop(client_write);

        tokio::time::timeout(Duration::from_secs(5), connection)
            .await
            .expect("connection outlived its client")
            .unwrap()
            .expect("a client hanging up mid-stream must end the connection cleanly");
    }

    #[tokio::test(start_paused = true)]
    async fn write_frame_times_out_when_the_client_stops_reading() {
        let (server, _client_never_reads) = UnixStream::pair().unwrap();
        let (_server_read, mut write_half) = server.into_split();
        let event = Event::new(
            "stalled",
            EventBody::Delta {
                text: "x".repeat(1024),
            },
        );

        let err = tokio::time::timeout(EVENT_WRITE_TIMEOUT * 4, async {
            loop {
                if let Err(e) = write_frame(&mut write_half, &event).await {
                    break e;
                }
            }
        })
        .await
        .expect("write to a stalled client never failed");
        assert!(matches!(err, SocketError::WriteTimeout(_)), "got {err:?}");
    }

    /// Never returns and emits nothing; `dropped` flips when the invocation
    /// future is torn down.
    #[derive(Debug)]
    struct SilentHangingTool {
        dropped: Arc<AtomicBool>,
    }

    struct DropFlag(Arc<AtomicBool>);

    impl Drop for DropFlag {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[async_trait::async_trait]
    impl assistd_tools::Tool for SilentHangingTool {
        fn name(&self) -> &'static str {
            "hang"
        }
        fn description(&self) -> Cow<'_, str> {
            "never returns".into()
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
        async fn invoke(
            &self,
            _turn: &TurnContext,
            _args: serde_json::Value,
        ) -> Result<assistd_tools::ToolOutput, assistd_tools::ToolError> {
            let _flag = DropFlag(self.dropped.clone());
            std::future::pending::<()>().await;
            unreachable!("hanging tool must never resolve")
        }
    }

    #[tokio::test]
    async fn client_hangup_during_a_silent_tool_call_cancels_the_turn() {
        let dropped = Arc::new(AtomicBool::new(false));
        let mut tools = assistd_tools::ToolRegistry::new();
        tools.register(SilentHangingTool {
            dropped: dropped.clone(),
        });
        let backend = ScriptedBackend::new([calling("c1", "hang")]).then_replying("done");
        let state = state_with(Arc::new(backend), Arc::new(tools)).await;
        with_server(&state, |path| async move {
            let (write, mut reader) =
                open_connection(&path, &[r#"{"type":"query","id":"h1","text":"go"}"#], true).await;
            loop {
                let event = read_event(&mut reader)
                    .await
                    .expect("connection closed before the tool call started");
                if matches!(event.body, EventBody::ToolCall { .. }) {
                    break;
                }
            }
            drop(reader);
            drop(write);

            tokio::time::timeout(Duration::from_secs(5), async {
                while !dropped.load(Ordering::SeqCst) {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            })
            .await
            .expect("turn kept running its tool after the client hung up");
        })
        .await;
    }
}
