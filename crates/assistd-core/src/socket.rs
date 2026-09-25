//! Unix-socket IPC server: one newline-delimited JSON request per
//! connection, answered by a stream of events.

use std::future::Future;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

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

/// Cap on one request frame: a 32 MiB image base64-encoded plus JSON
/// overhead, so a runaway client cannot OOM the daemon.
const MAX_REQUEST_BYTES: u64 = 64 * 1024 * 1024;

/// Pause after EMFILE/ENFILE from `accept()`, which would otherwise spin.
const FD_EXHAUSTION_BACKOFF: Duration = Duration::from_millis(100);

/// How long an existing socket gets to accept a probe before it is
/// treated as stale; one that hangs past this is treated as live.
const PROBE_TIMEOUT: Duration = Duration::from_secs(1);

/// Errors produced by the socket listener and per-connection handlers.
#[derive(Debug, Error)]
pub enum SocketError {
    #[error(
        "another assistd daemon is already accepting connections at {path}; refusing to clobber \
         its socket"
    )]
    AlreadyRunning { path: PathBuf },

    #[error("failed to remove stale socket file at {path}: {source}")]
    StaleCleanup {
        path: PathBuf,
        #[source]
        source: io::Error,
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

/// [`serve_at`] on the default path from [`assistd_ipc::socket_path`].
pub async fn serve<F>(state: Arc<AppState>, shutdown: F) -> Result<(), SocketError>
where
    F: Future<Output = ()>,
{
    let path = assistd_ipc::socket_path();
    serve_at(&path, state, shutdown).await
}

/// Serve the IPC socket at `path` until `shutdown` resolves, then drain
/// in-flight connections for up to `daemon.shutdown_grace_secs`. A stale
/// socket file at `path` is removed first; a live one is an error.
pub async fn serve_at<F>(path: &Path, state: Arc<AppState>, shutdown: F) -> Result<(), SocketError>
where
    F: Future<Output = ()>,
{
    prepare_socket_path(path).await?;

    let listener = UnixListener::bind(path).map_err(|source| SocketError::Bind {
        path: path.to_path_buf(),
        source,
    })?;
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
    let grace = Duration::from_secs(state.config.daemon.shutdown_grace_secs);
    let mut connections: JoinSet<()> = JoinSet::new();
    let mut fd_exhausted = false;
    let (drain_tx, drain_rx) = watch::channel(false);

    tokio::pin!(shutdown);
    loop {
        tokio::select! {
            _ = &mut shutdown => {
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
                        spawn_connection(&mut connections, stream, state.clone(), drain_rx.clone());
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
            backoff_ms = FD_EXHAUSTION_BACKOFF.as_millis() as u64,
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
    matches!(err.raw_os_error(), Some(libc::EMFILE) | Some(libc::ENFILE))
}

async fn handle_connection(
    stream: UnixStream,
    state: Arc<AppState>,
    mut drain: watch::Receiver<bool>,
) -> Result<(), SocketError> {
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
            .scope(router_for_dispatch, dispatch_state.dispatch(req, tx))
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
    let mut write_half = forward_res?;
    write_half.shutdown().await?;
    Ok(())
}

/// Read and parse the connection's first frame. `None` means the client
/// left or was already sent an error and closed.
async fn read_initial_request(
    reader: &mut BufReader<OwnedReadHalf>,
    write_half: &mut OwnedWriteHalf,
) -> Result<Option<Request>, SocketError> {
    let mut line = String::new();
    let bytes_read = reader.take(MAX_REQUEST_BYTES).read_line(&mut line).await?;
    if bytes_read == 0 {
        debug!("client disconnected without sending a request");
        return Ok(None);
    }
    let rejection = if !line.ends_with('\n') {
        format!("request exceeded {MAX_REQUEST_BYTES}-byte limit")
    } else {
        match serde_json::from_str::<Request>(line.trim()) {
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

/// Write every dispatched event to the client, teeing it onto the bus
/// unless this connection is itself a bus subscriber.
async fn forward_events(
    mut rx: mpsc::Receiver<Event>,
    mut write_half: OwnedWriteHalf,
    state: Arc<AppState>,
    is_subscribe: bool,
) -> Result<OwnedWriteHalf, SocketError> {
    while let Some(event) = rx.recv().await {
        if !is_subscribe {
            state.runtime.publish(&event);
        }
        write_event(&mut write_half, &event).await?;
    }
    Ok(write_half)
}

/// Route mid-stream `ConfirmResponse` frames to `router` until the client
/// closes its write side, then close the router.
async fn route_confirm_responses(mut reader: BufReader<OwnedReadHalf>, router: Arc<ConfirmRouter>) {
    let mut buf = String::new();
    loop {
        buf.clear();
        match (&mut reader)
            .take(MAX_REQUEST_BYTES)
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
                "mid-stream request exceeded {MAX_REQUEST_BYTES}-byte limit; closing read \
                 side"
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
