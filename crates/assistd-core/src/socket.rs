//! Unix-socket IPC server: one newline-delimited JSON request per
//! connection, answered by a stream of events.

use std::fs::{DirBuilder, File, OpenOptions, Permissions, TryLockError};
use std::future::Future;
use std::io;
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use rustix::event::{PollFd, PollFlags, Timespec, poll};
use rustix::process::geteuid;
use thiserror::Error;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::unix::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{mpsc, watch};
use tokio::task::JoinSet;
use tracing::{Instrument, debug, error, info, warn};

use assistd_ipc::{Event, Request};
use assistd_tools::{Approval, CONFIRM_ROUTER, CONFIRM_TIMEOUT, ConfirmRouter};

use crate::AppState;
use crate::recovery::drain_join_set;

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

/// Start of a serialized `Request::Query`; only a frame opening with it
/// may grow past [`MAX_CONTROL_REQUEST_BYTES`].
const QUERY_FRAME_PREFIX: &[u8] = br#"{"type":"query""#;

/// How long a new connection may take to deliver its first frame.
const INITIAL_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// Connections served at once; further ones are closed on accept.
const MAX_CONNECTIONS: usize = 128;

/// Pause after EMFILE/ENFILE from `accept()`, which would otherwise spin.
const FD_EXHAUSTION_BACKOFF: Duration = Duration::from_millis(100);

/// How long an existing socket gets to accept a probe before it is
/// treated as stale; one that hangs past this is treated as live.
const PROBE_TIMEOUT: Duration = Duration::from_secs(1);

/// Mode for a socket directory the daemon has to create itself.
const SOCKET_DIR_MODE: u32 = 0o700;

/// Mode for the socket itself: only the daemon's user may connect.
const SOCKET_MODE: u32 = 0o600;

/// Mode for lock files beside the socket.
const LOCK_FILE_MODE: u32 = 0o600;

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

    #[error("failed to open lock file at {path}: {source}")]
    LockFile {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

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

    #[error("failed to prepare socket directory {path}: {source}")]
    SocketDir {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("socket directory {path} is {problem}; refusing to listen there")]
    UnsafeSocketDir {
        path: PathBuf,
        problem: &'static str,
    },

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

    /// Take the lock beside `socket_path`, creating the socket directory
    /// owner-only if needed. Fails with `AlreadyStarting` when another
    /// process holds it; the lock is released when the guard drops or the
    /// process exits.
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

/// Serve the IPC socket at `path` until `shutdown` resolves, then drain
/// in-flight connections for up to `daemon.shutdown_grace_secs`. The
/// socket's directory must belong to this user and not be writable by
/// others, and the socket is owner-only. A stale socket file at `path` is
/// removed first; a live one is an error.
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

/// Open (creating owner-only if needed) a lock file in the socket
/// directory without following a symlink at `lock_path`. Fails when the
/// directory is not a private directory owned by this user.
pub fn open_private_lock_file(lock_path: &Path) -> Result<File, SocketError> {
    if let Some(dir) = lock_path.parent() {
        ensure_private_socket_dir(dir)?;
    }
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(LOCK_FILE_MODE)
        .custom_flags(libc::O_NOFOLLOW)
        .open(lock_path)
        .map_err(|source| SocketError::LockFile {
            path: lock_path.to_path_buf(),
            source,
        })
}

fn ensure_private_socket_dir(dir: &Path) -> Result<(), SocketError> {
    let dir_error = |source| SocketError::SocketDir {
        path: dir.to_path_buf(),
        source,
    };
    DirBuilder::new()
        .recursive(true)
        .mode(SOCKET_DIR_MODE)
        .create(dir)
        .map_err(dir_error)?;
    let metadata = std::fs::symlink_metadata(dir).map_err(dir_error)?;
    let problem = if !metadata.is_dir() {
        Some("not a directory")
    } else if metadata.uid() != geteuid().as_raw() {
        Some("owned by another user")
    } else if metadata.mode() & 0o022 != 0 {
        Some("writable by other users")
    } else {
        None
    };
    problem.map_or(Ok(()), |problem| {
        Err(SocketError::UnsafeSocketDir {
            path: dir.to_path_buf(),
            problem,
        })
    })
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
    let grace = Duration::from_secs(state.config.daemon.shutdown_grace_secs);
    let mut connections: JoinSet<()> = JoinSet::new();
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
                        } else if peer_is_daemon_user(&stream) {
                            spawn_connection(
                                &mut connections,
                                stream,
                                state.clone(),
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
    drain_join_set(&mut connections, grace, "connection").await;
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

fn peer_pid(stream: &UnixStream) -> Option<u32> {
    let pid = stream.peer_cred().ok()?.pid()?;
    u32::try_from(pid).ok()
}

fn spawn_connection(
    connections: &mut JoinSet<()>,
    stream: UnixStream,
    state: Arc<AppState>,
    drain: watch::Receiver<bool>,
) {
    connections.spawn(async move {
        if let Err(e) = handle_connection(stream, state, drain).await {
            error!("connection error: {e}");
        }
    });
}

/// Warns only on the first failure of a run, then sleeps.
async fn back_off_from_fd_exhaustion(err: &io::Error, fd_exhausted: &mut bool) {
    if !*fd_exhausted {
        warn!(
            error = %err,
            backoff_ms = u64::try_from(FD_EXHAUSTION_BACKOFF.as_millis()).unwrap_or(u64::MAX),
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
    matches!(err.raw_os_error(), Some(libc::EMFILE | libc::ENFILE))
}

async fn handle_connection(
    stream: UnixStream,
    state: Arc<AppState>,
    mut drain: watch::Receiver<bool>,
) -> Result<(), SocketError> {
    let peer_pid = peer_pid(&stream);
    let (read_half, mut write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);
    let Some(req) = read_initial_request(&mut reader, &mut write_half).await? else {
        return Ok(());
    };

    let (tx, rx) = mpsc::channel::<Event>(EVENT_CHANNEL_CAPACITY);
    let span = tracing::info_span!("ipc", id = %req.id(), req = req.kind());
    let router = ConfirmRouter::new(req.id().to_string(), tx.clone(), CONFIRM_TIMEOUT);
    let is_subscribe = matches!(req, Request::Subscribe { .. });

    let dispatch_state = state.clone();
    let router_for_dispatch = router.clone();
    let dispatch_fut = async move {
        CONFIRM_ROUTER
            .scope(
                router_for_dispatch,
                Box::pin(dispatch_state.dispatch(req, peer_pid, tx)),
            )
            .await
    };
    let forward_fut = forward_events(rx, write_half, state, is_subscribe);
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

/// Read and parse the connection's first frame. `None` means the client
/// left, missed [`INITIAL_REQUEST_TIMEOUT`], or was already sent an error
/// and closed.
async fn read_initial_request(
    reader: &mut BufReader<OwnedReadHalf>,
    write_half: &mut OwnedWriteHalf,
) -> Result<Option<Request>, SocketError> {
    let mut line = Vec::new();
    let Ok(limit) = tokio::time::timeout(
        INITIAL_REQUEST_TIMEOUT,
        read_request_frame(reader, &mut line),
    )
    .await
    else {
        warn!(
            timeout_secs = INITIAL_REQUEST_TIMEOUT.as_secs(),
            "client sent no complete request in time; closing"
        );
        return Ok(None);
    };
    let limit = limit?;
    if line.is_empty() {
        debug!("client disconnected without sending a request");
        return Ok(None);
    }
    let rejection = if !line.ends_with(b"\n") {
        format!("request exceeded {limit}-byte limit")
    } else {
        match serde_json::from_slice::<Request>(line.trim_ascii()) {
            Ok(req) => return Ok(Some(req)),
            Err(e) => format!("invalid request: {e}"),
        }
    };
    let err = Event::Error {
        id: String::new(),
        message: rejection,
    };
    write_event(write_half, &err).await?;
    write_half.shutdown().await?;
    Ok(None)
}

/// Read one frame into `line`, stopping at the cap its prefix allows,
/// which is returned.
async fn read_request_frame(
    reader: &mut BufReader<OwnedReadHalf>,
    line: &mut Vec<u8>,
) -> io::Result<u64> {
    (&mut *reader)
        .take(MAX_CONTROL_REQUEST_BYTES)
        .read_until(b'\n', line)
        .await?;
    let may_grow =
        !line.ends_with(b"\n") && line.trim_ascii_start().starts_with(QUERY_FRAME_PREFIX);
    if !may_grow {
        return Ok(MAX_CONTROL_REQUEST_BYTES);
    }
    reader
        .take(MAX_REQUEST_BYTES - MAX_CONTROL_REQUEST_BYTES)
        .read_until(b'\n', line)
        .await?;
    Ok(MAX_REQUEST_BYTES)
}

/// Write every dispatched event to the client, teeing it onto the bus
/// unless this connection is itself a bus subscriber. Returns early,
/// dropping the event receiver, once the client hangs up.
async fn forward_events(
    mut rx: mpsc::Receiver<Event>,
    mut write_half: OwnedWriteHalf,
    state: Arc<AppState>,
    is_subscribe: bool,
) -> Result<ForwardEnd, SocketError> {
    let mut hangup_poll = tokio::time::interval(PEER_HANGUP_POLL_INTERVAL);
    loop {
        tokio::select! {
            received = rx.recv() => {
                let Some(event) = received else {
                    return Ok(ForwardEnd::Drained(write_half));
                };
                if !is_subscribe {
                    state.runtime.publish(&event);
                }
                write_event(&mut write_half, &event).await?;
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

/// Route mid-stream `ConfirmResponse` frames to `router` until the client
/// closes its write side, then close the router.
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
    match serde_json::from_str::<Request>(line) {
        Ok(Request::ConfirmResponse {
            confirm_id,
            allow,
            always,
            ..
        }) => {
            let approval = Approval::from_answer(allow, always);
            if let Err(e) = router.route_response(&confirm_id, approval) {
                warn!(confirm_id = %confirm_id, reason = %e, "unmatched ConfirmResponse");
            }
        }
        Ok(other) => {
            warn!(
                kind = other.kind(),
                "unexpected mid-stream request; only ConfirmResponse is honored after the \
                 initial request"
            );
        }
        Err(e) => {
            warn!(error = %e, "invalid mid-stream JSON; ignoring line");
        }
    }
}

async fn write_event(write_half: &mut OwnedWriteHalf, event: &Event) -> Result<(), SocketError> {
    let mut out = serde_json::to_string(event)?;
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
mod tests;
