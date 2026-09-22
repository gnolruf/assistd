use crate::AppState;
use assistd_ipc::{Event, Request};
use assistd_tools::{CONFIRM_ROUTER, CONFIRM_TIMEOUT, ConfirmRouter};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::mpsc;
use tokio::task::JoinSet;
use tracing::{Instrument, debug, error, info, warn};

const EVENT_CHANNEL_CAPACITY: usize = 32;

/// Hard cap on a single newline-delimited request frame. `read_line` is
/// otherwise unbounded; a runaway client streaming a multi-GB prompt
/// would OOM the daemon. 64 MiB fits a single 32 MiB image attachment
/// (`MAX_IMAGE_BYTES`) base64-encoded plus JSON overhead with margin.
const MAX_REQUEST_BYTES: u64 = 64 * 1024 * 1024;

/// Backoff applied when `accept()` returns EMFILE/ENFILE. Without it the
/// select! arm spins, because the error is returned synchronously and
/// nothing else changes to clear the condition.
const FD_EXHAUSTION_BACKOFF: Duration = Duration::from_millis(100);

/// Matches the raw errno because EMFILE maps to the unstable
/// `io::ErrorKind::Uncategorized` on current stable Rust.
fn is_fd_exhaustion(err: &std::io::Error) -> bool {
    matches!(err.raw_os_error(), Some(libc::EMFILE) | Some(libc::ENFILE))
}

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
        source: std::io::Error,
    },

    #[error("failed to bind unix socket at {path}: {source}")]
    Bind {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("socket I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),
}

/// How long an existing socket gets to accept a probe connection before
/// it is treated as stale. A socket that hangs past this is treated as
/// live to avoid clobbering it.
const PROBE_TIMEOUT: Duration = Duration::from_secs(1);

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

    let owned_path = PathBuf::from(path);
    let result = accept_loop(listener, state, shutdown).await;

    if let Err(e) = std::fs::remove_file(&owned_path) {
        warn!(
            "failed to remove socket file at {}: {}",
            owned_path.display(),
            e
        );
    }

    result
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
    let (drain_tx, drain_rx) = tokio::sync::watch::channel(false);

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
                        let conn_state = state.clone();
                        let conn_drain = drain_rx.clone();
                        connections.spawn(async move {
                            if let Err(e) = handle_connection(stream, conn_state, conn_drain).await
                            {
                                error!("connection error: {e}");
                            }
                        });
                    }
                    Err(e) if is_fd_exhaustion(&e) => {
                        if !fd_exhausted {
                            warn!(
                                error = %e,
                                backoff_ms = FD_EXHAUSTION_BACKOFF.as_millis() as u64,
                                "accept failed: file-descriptor limit reached; backing \
                                 off until in-flight connections release descriptors. \
                                 Repeat occurrences suppressed until recovery."
                            );
                            fd_exhausted = true;
                        }
                        tokio::time::sleep(FD_EXHAUSTION_BACKOFF).await;
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
    crate::recovery::drain_join_set(&mut connections, grace, "connection").await;
    Ok(())
}

async fn handle_connection(
    stream: UnixStream,
    state: Arc<AppState>,
    drain: tokio::sync::watch::Receiver<bool>,
) -> Result<(), SocketError> {
    let (read_half, mut write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);
    let mut line = String::new();
    let n = (&mut reader)
        .take(MAX_REQUEST_BYTES)
        .read_line(&mut line)
        .await?;
    if n == 0 {
        debug!("client disconnected without sending a request");
        return Ok(());
    }
    if !line.ends_with('\n') {
        let err = Event::Error {
            id: String::new(),
            message: format!("request exceeded {MAX_REQUEST_BYTES}-byte limit"),
        };
        write_event(&mut write_half, &err).await?;
        write_half.shutdown().await?;
        return Ok(());
    }

    let req = match serde_json::from_str::<Request>(line.trim()) {
        Ok(req) => req,
        Err(e) => {
            let err = Event::Error {
                id: String::new(),
                message: format!("invalid request: {e}"),
            };
            write_event(&mut write_half, &err).await?;
            write_half.shutdown().await?;
            return Ok(());
        }
    };

    let (tx, mut rx) = mpsc::channel::<Event>(EVENT_CHANNEL_CAPACITY);
    let dispatch_state = state.clone();

    let span = tracing::info_span!("ipc", id = %req.id(), req = req.kind());

    let router = ConfirmRouter::new(req.id().to_string(), tx.clone(), CONFIRM_TIMEOUT);

    let is_subscribe = matches!(req, Request::Subscribe { .. });
    let events_bus = state.runtime.events_bus().clone();

    let router_for_dispatch = router.clone();
    let dispatch_fut = async move {
        CONFIRM_ROUTER
            .scope(router_for_dispatch, dispatch_state.dispatch(req, tx))
            .await
    };
    let forward_fut = async {
        while let Some(event) = rx.recv().await {
            // Subscribe forwarders read from the bus; teeing back
            // onto it would loop.
            if !is_subscribe && event.kind().is_some() {
                let _ = events_bus.send(event.clone());
            }
            write_event(&mut write_half, &event).await?;
        }
        Ok::<_, SocketError>(write_half)
    };

    let read_fut = async move {
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
            if trimmed.is_empty() {
                continue;
            }
            match serde_json::from_str::<Request>(trimmed) {
                Ok(Request::ConfirmResponse {
                    confirm_id, allow, ..
                }) => {
                    if let Err(reason) = router.route_response(&confirm_id, allow) {
                        warn!(confirm_id = %confirm_id, reason, "unmatched ConfirmResponse");
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
        router.close();
    };

    let dispatch_and_read = async {
        tokio::pin!(dispatch_fut);
        tokio::pin!(read_fut);
        tokio::select! {
            res = &mut dispatch_fut => res,
            () = &mut read_fut => dispatch_fut.await,
        }
    };

    let mut drain = drain;
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
        error!("dispatch error: {e:#}");
    }
    let mut write_half = forward_res?;
    write_half.shutdown().await?;
    Ok(())
}

async fn write_event(
    write_half: &mut tokio::net::unix::OwnedWriteHalf,
    event: &Event,
) -> Result<(), SocketError> {
    let mut out = serde_json::to_string(event)?;
    out.push('\n');
    write_half.write_all(out.as_bytes()).await?;
    write_half.flush().await?;
    Ok(())
}

#[cfg(test)]
mod tests;
