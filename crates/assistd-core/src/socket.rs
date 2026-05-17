use crate::AppState;
use assistd_ipc::{Event, Request};
use assistd_tools::{CONFIRM_ROUTER, ConfirmRouter};
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
/// select! arm spins as fast as the runtime can poll because the error
/// is returned synchronously from the syscall and nothing else changes
/// to clear the condition. 100 ms gives in-flight connections time to
/// finish and release file descriptors, while keeping the daemon
/// responsive once recovery happens.
const FD_EXHAUSTION_BACKOFF: Duration = Duration::from_millis(100);

/// Returns true when an accept error reflects a file-descriptor limit
/// (`EMFILE` — per-process limit, the common case; `ENFILE` —
/// system-wide). We can't rely on `io::ErrorKind` here because EMFILE
/// maps to the unstable `Uncategorized` variant on current stable Rust,
/// so we match the raw errno directly.
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

/// How long to wait for an existing socket to accept a probe connection
/// before deciding it's stale. A live daemon's accept loop is reactive
/// enough that 1s is generous; a hung socket past this point is treated
/// as live to avoid clobbering it.
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

/// Serve the IPC socket at the default path from [`assistd_ipc::socket_path`].
///
/// Resolves the socket path and delegates to [`serve_at`]. Runs until `shutdown`
/// resolves, then drains in-flight connections within the configured grace period.
///
/// # Errors
///
/// Returns [`SocketError`] if the stale socket file cannot be removed or the
/// listener cannot bind.
pub async fn serve<F>(state: Arc<AppState>, shutdown: F) -> Result<(), SocketError>
where
    F: Future<Output = ()>,
{
    let path = assistd_ipc::socket_path();
    serve_at(&path, state, shutdown).await
}

/// Serve the IPC socket at `path`, using `state` to dispatch requests.
///
/// Removes any stale socket file at `path`, binds a new [`UnixListener`],
/// and runs the accept loop until `shutdown` resolves. After shutdown, in-flight
/// connections are drained for up to `config.daemon.shutdown_grace_secs` before
/// the listener closes. Removes the socket file on exit.
///
/// # Errors
///
/// Returns [`SocketError`] if the stale file cannot be removed, the listener
/// cannot bind, or the accept loop encounters an unrecoverable I/O error.
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
                        connections.spawn(async move {
                            if let Err(e) = handle_connection(stream, conn_state).await {
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

    let in_flight = connections.len();
    if in_flight == 0 {
        return Ok(());
    }
    info!(
        grace_secs = grace.as_secs(),
        in_flight, "draining in-flight connections"
    );

    let drained = tokio::time::timeout(grace, async {
        while let Some(res) = connections.join_next().await {
            if let Err(e) = res
                && e.is_panic()
            {
                error!("connection task panicked: {e}");
            }
        }
    })
    .await;

    if drained.is_err() {
        let remaining = connections.len();
        warn!(
            remaining,
            "shutdown grace expired; aborting remaining connections"
        );
        connections.shutdown().await;
    }

    Ok(())
}

async fn handle_connection(stream: UnixStream, state: Arc<AppState>) -> Result<(), SocketError> {
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
            // We can't read an id out of unparseable input; use empty.
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

    let router = ConfirmRouter::new(req.id().to_string(), tx.clone());

    // Tee broadcast-eligible events from this connection's outbound
    // stream onto the daemon-wide events bus so `Request::Subscribe`
    // connections can observe activity. Skipped on Subscribe
    // connections themselves to prevent a feedback loop: a Subscribe
    // forwarder's events came *from* the bus, and re-broadcasting
    // them would amplify every event across every subscriber.
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
            if !is_subscribe && let Some(_kind) = event.kind() {
                let _ = events_bus.send(event.clone());
            }
            write_event(&mut write_half, &event).await?;
        }
        Ok::<_, SocketError>(write_half)
    };

    let router_for_reader = router.clone();
    let read_fut = async move {
        let mut buf = String::new();
        loop {
            buf.clear();
            match (&mut reader)
                .take(MAX_REQUEST_BYTES)
                .read_line(&mut buf)
                .await
            {
                Ok(0) => {
                    // Half-close from the client (one-shot CLI pattern).
                    // Exit cleanly.
                    break;
                }
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
                    if let Err(reason) = router_for_reader.route_response(&confirm_id, allow) {
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
    };

    let (dispatch_res, forward_res, _) = async {
        let read_handle = tokio::spawn(read_fut);
        let (d, f) = tokio::join!(dispatch_fut, forward_fut);
        drop(router);
        read_handle.abort();
        let _ = read_handle.await;
        (d, f, ())
    }
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
mod tests {
    use super::*;
    use crate::{Config, PresenceManager, PresenceState};
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    use tokio::net::UnixStream;
    use tokio::sync::oneshot;

    #[test]
    fn fd_exhaustion_predicate_matches_emfile_and_enfile() {
        let emfile = std::io::Error::from_raw_os_error(libc::EMFILE);
        let enfile = std::io::Error::from_raw_os_error(libc::ENFILE);
        assert!(
            is_fd_exhaustion(&emfile),
            "EMFILE must classify as FD exhaustion"
        );
        assert!(
            is_fd_exhaustion(&enfile),
            "ENFILE must classify as FD exhaustion"
        );
    }

    #[test]
    fn fd_exhaustion_predicate_rejects_unrelated_errors() {
        let conn_reset = std::io::Error::from_raw_os_error(libc::ECONNRESET);
        let conn_aborted = std::io::Error::from_raw_os_error(libc::ECONNABORTED);
        let no_errno = std::io::Error::other("synthetic non-OS error");
        assert!(!is_fd_exhaustion(&conn_reset));
        assert!(!is_fd_exhaustion(&conn_aborted));
        assert!(!is_fd_exhaustion(&no_errno));
    }

    fn test_state() -> Arc<AppState> {
        Arc::new(AppState::new(
            Config::default(),
            Arc::new(assistd_llm::EchoBackend::new()),
            PresenceManager::stub(PresenceState::Active),
            Arc::new(assistd_tools::ToolRegistry::default()),
            Arc::new(assistd_voice::NoVoiceInput::new()),
            Arc::new(assistd_voice::NoContinuousListener::new()),
            assistd_voice::VoiceOutputController::new(Arc::new(assistd_voice::NoVoiceOutput), true),
        ))
    }

    async fn wait_for_listener(path: &Path) {
        for _ in 0..200 {
            if UnixStream::connect(path).await.is_ok() {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        panic!("listener at {} did not become ready", path.display());
    }

    async fn send_request_collect_events(path: &Path, body: &str) -> Vec<Event> {
        let stream = UnixStream::connect(path).await.unwrap();
        let (read, mut write) = stream.into_split();
        write.write_all(body.as_bytes()).await.unwrap();
        write.write_all(b"\n").await.unwrap();
        write.shutdown().await.unwrap();
        let mut reader = BufReader::new(read);
        let mut events = Vec::new();
        loop {
            let mut line = String::new();
            let n = reader.read_line(&mut line).await.unwrap();
            if n == 0 {
                break;
            }
            let event: Event = serde_json::from_str(line.trim()).unwrap();
            let terminal = event.is_terminal();
            events.push(event);
            if terminal {
                break;
            }
        }
        events
    }

    #[tokio::test]
    async fn echoes_query_text_as_delta_then_done() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("assistd.sock");
        let (tx, rx) = oneshot::channel::<()>();
        let server_path = path.clone();
        let state = test_state();
        let server = tokio::spawn(async move {
            serve_at(&server_path, state, async {
                let _ = rx.await;
            })
            .await
            .unwrap();
        });

        wait_for_listener(&path).await;

        let events =
            send_request_collect_events(&path, r#"{"type":"query","id":"req-1","text":"ping"}"#)
                .await;
        assert_eq!(events.len(), 2);
        match &events[0] {
            Event::Delta { id, text } => {
                assert_eq!(id, "req-1");
                assert_eq!(text, "ping");
            }
            other => panic!("expected Delta, got {other:?}"),
        }
        match &events[1] {
            Event::Done { id } => assert_eq!(id, "req-1"),
            other => panic!("expected Done, got {other:?}"),
        }

        tx.send(()).unwrap();
        server.await.unwrap();
        assert!(!path.exists(), "socket file should be removed on shutdown");
    }

    #[tokio::test]
    async fn handles_concurrent_connections() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("assistd.sock");
        let (tx, rx) = oneshot::channel::<()>();
        let server_path = path.clone();
        let state = test_state();
        let server = tokio::spawn(async move {
            serve_at(&server_path, state, async {
                let _ = rx.await;
            })
            .await
            .unwrap();
        });

        wait_for_listener(&path).await;

        let mut handles = Vec::new();
        for i in 0..16 {
            let p = path.clone();
            handles.push(tokio::spawn(async move {
                let id = format!("req-{i}");
                let body = format!(r#"{{"type":"query","id":"{id}","text":"msg-{i}"}}"#);
                let events = send_request_collect_events(&p, &body).await;
                assert_eq!(events.len(), 2, "expected Delta+Done, got {events:?}");
                assert!(matches!(events[0], Event::Delta { .. }));
                assert!(matches!(events[1], Event::Done { .. }));
                assert_eq!(events[0].id(), id);
                assert_eq!(events[1].id(), id);
            }));
        }
        for h in handles {
            h.await.unwrap();
        }

        tx.send(()).unwrap();
        server.await.unwrap();
    }

    #[tokio::test]
    async fn oversize_request_is_rejected_without_oom() {
        with_server(test_state(), |path| async move {
            let stream = UnixStream::connect(&path).await.unwrap();
            let (read, mut write) = stream.into_split();

            let chunk = vec![b'a'; 1024 * 1024];
            let mut remaining = MAX_REQUEST_BYTES as usize + 1;
            while remaining > 0 {
                let n = remaining.min(chunk.len());
                if write.write_all(&chunk[..n]).await.is_err() {
                    // Daemon may close the read side once the cap
                    // is hit; that's a valid outcome too.
                    break;
                }
                remaining -= n;
            }
            let _ = write.shutdown().await;

            let mut reader = BufReader::new(read);
            let mut line = String::new();
            let n = reader.read_line(&mut line).await.unwrap();
            assert!(n > 0, "expected an Error event before EOF");
            let event: Event = serde_json::from_str(line.trim()).unwrap();
            match event {
                Event::Error { id, message } => {
                    assert_eq!(id, "");
                    assert!(
                        message.contains("limit"),
                        "expected limit error, got {message}"
                    );
                }
                other => panic!("expected Error event, got {other:?}"),
            }
        })
        .await;
    }

    #[tokio::test]
    async fn malformed_request_returns_error_event() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("assistd.sock");
        let (tx, rx) = oneshot::channel::<()>();
        let server_path = path.clone();
        let state = test_state();
        let server = tokio::spawn(async move {
            serve_at(&server_path, state, async {
                let _ = rx.await;
            })
            .await
            .unwrap();
        });

        wait_for_listener(&path).await;

        let events = send_request_collect_events(&path, "not json").await;
        assert_eq!(events.len(), 1);
        match &events[0] {
            Event::Error { .. } => {}
            other => panic!("expected Error event, got {other:?}"),
        }

        tx.send(()).unwrap();
        server.await.unwrap();
    }

    #[tokio::test]
    async fn get_listen_state_reports_inactive_by_default() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("assistd.sock");
        let (tx, rx) = oneshot::channel::<()>();
        let server_path = path.clone();
        let state = test_state();
        let server = tokio::spawn(async move {
            serve_at(&server_path, state, async {
                let _ = rx.await;
            })
            .await
            .unwrap();
        });
        wait_for_listener(&path).await;

        let events =
            send_request_collect_events(&path, r#"{"type":"get_listen_state","id":"ls-1"}"#).await;
        assert!(matches!(
            events.first(),
            Some(Event::ListenState { id, active: false }) if id == "ls-1"
        ));
        assert!(matches!(events.last(), Some(Event::Done { .. })));

        tx.send(()).unwrap();
        server.await.unwrap();
    }

    #[tokio::test]
    async fn listen_start_errors_with_no_continuous_listener_stub() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("assistd.sock");
        let (tx, rx) = oneshot::channel::<()>();
        let server_path = path.clone();
        let state = test_state();
        let server = tokio::spawn(async move {
            serve_at(&server_path, state, async {
                let _ = rx.await;
            })
            .await
            .unwrap();
        });
        wait_for_listener(&path).await;

        let events =
            send_request_collect_events(&path, r#"{"type":"listen_start","id":"ls-start"}"#).await;
        assert!(
            matches!(events.last(), Some(Event::Error { message, .. }) if message.contains("not enabled")),
            "expected Error with 'not enabled' message; got {events:?}"
        );

        tx.send(()).unwrap();
        server.await.unwrap();
    }

    #[tokio::test]
    async fn second_serve_refuses_to_clobber_live_socket() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("assistd.sock");
        let (tx, rx) = oneshot::channel::<()>();
        let server_path = path.clone();
        let state = test_state();
        let server = tokio::spawn(async move {
            serve_at(&server_path, state, async {
                let _ = rx.await;
            })
            .await
            .unwrap();
        });
        wait_for_listener(&path).await;

        let second_state = test_state();
        let err = serve_at(&path, second_state, std::future::pending::<()>())
            .await
            .expect_err("second serve_at must fail while first is alive");
        match err {
            SocketError::AlreadyRunning { path: p } => assert_eq!(p, path),
            other => panic!("expected AlreadyRunning, got {other:?}"),
        }
        assert!(
            path.exists(),
            "live socket must remain after a refused second-bind attempt"
        );

        let events =
            send_request_collect_events(&path, r#"{"type":"query","id":"q","text":"ok"}"#).await;
        assert!(matches!(events.last(), Some(Event::Done { .. })));

        tx.send(()).unwrap();
        server.await.unwrap();
    }

    #[tokio::test]
    async fn removes_stale_socket_file_on_bind() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("assistd.sock");
        std::fs::write(&path, b"stale").unwrap();
        assert!(path.exists());

        let (tx, rx) = oneshot::channel::<()>();
        let server_path = path.clone();
        let state = test_state();
        let server = tokio::spawn(async move {
            serve_at(&server_path, state, async {
                let _ = rx.await;
            })
            .await
            .unwrap();
        });

        wait_for_listener(&path).await;

        let events =
            send_request_collect_events(&path, r#"{"type":"query","id":"req-ok","text":"ok"}"#)
                .await;
        assert!(matches!(events.last(), Some(Event::Done { .. })));

        tx.send(()).unwrap();
        server.await.unwrap();
    }

    /// Backend that emits N deltas with a fixed pause between each, then
    /// Done. Used to exercise graceful shutdown of in-flight streams.
    struct SlowBackend {
        deltas: usize,
        pause: std::time::Duration,
    }

    #[async_trait::async_trait]
    impl assistd_llm::LlmBackend for SlowBackend {
        async fn generate(
            &self,
            _prompt: String,
            tx: tokio::sync::mpsc::Sender<assistd_llm::LlmEvent>,
        ) -> assistd_llm::LlmResult<()> {
            for i in 0..self.deltas {
                let _ = tx
                    .send(assistd_llm::LlmEvent::Delta {
                        text: format!("d{i}"),
                    })
                    .await;
                tokio::time::sleep(self.pause).await;
            }
            let _ = tx.send(assistd_llm::LlmEvent::Done).await;
            Ok(())
        }

        async fn push_user(
            &self,
            _text: String,
            _attachments: Vec<assistd_tools::Attachment>,
        ) -> assistd_llm::LlmResult<()> {
            Ok(())
        }

        async fn push_tool_results(
            &self,
            _results: Vec<assistd_llm::ToolResultPayload>,
        ) -> assistd_llm::LlmResult<()> {
            Ok(())
        }

        async fn step(
            &self,
            _tools: Vec<serde_json::Value>,
            tx: tokio::sync::mpsc::Sender<assistd_llm::LlmEvent>,
        ) -> assistd_llm::LlmResult<assistd_llm::StepOutcome> {
            for i in 0..self.deltas {
                let _ = tx
                    .send(assistd_llm::LlmEvent::Delta {
                        text: format!("d{i}"),
                    })
                    .await;
                tokio::time::sleep(self.pause).await;
            }
            Ok(assistd_llm::StepOutcome::Final)
        }
    }

    /// Backend that emits a single delta then blocks indefinitely on an
    /// un-awoken channel. Used to exercise the shutdown grace timeout.
    struct StuckBackend;

    #[async_trait::async_trait]
    impl assistd_llm::LlmBackend for StuckBackend {
        async fn generate(
            &self,
            _prompt: String,
            tx: tokio::sync::mpsc::Sender<assistd_llm::LlmEvent>,
        ) -> assistd_llm::LlmResult<()> {
            let _ = tx
                .send(assistd_llm::LlmEvent::Delta {
                    text: "stuck".into(),
                })
                .await;
            std::future::pending::<()>().await;
            Ok(())
        }

        async fn push_user(
            &self,
            _text: String,
            _attachments: Vec<assistd_tools::Attachment>,
        ) -> assistd_llm::LlmResult<()> {
            Ok(())
        }

        async fn push_tool_results(
            &self,
            _results: Vec<assistd_llm::ToolResultPayload>,
        ) -> assistd_llm::LlmResult<()> {
            Ok(())
        }

        async fn step(
            &self,
            _tools: Vec<serde_json::Value>,
            tx: tokio::sync::mpsc::Sender<assistd_llm::LlmEvent>,
        ) -> assistd_llm::LlmResult<assistd_llm::StepOutcome> {
            let _ = tx
                .send(assistd_llm::LlmEvent::Delta {
                    text: "stuck".into(),
                })
                .await;
            std::future::pending::<()>().await;
            Ok(assistd_llm::StepOutcome::Final)
        }
    }

    fn state_with_backend_and_grace(
        backend: Arc<dyn assistd_llm::LlmBackend>,
        grace_secs: u64,
    ) -> Arc<AppState> {
        let mut config = Config::default();
        config.daemon.shutdown_grace_secs = grace_secs;
        Arc::new(AppState::new(
            config,
            backend,
            PresenceManager::stub(PresenceState::Active),
            Arc::new(assistd_tools::ToolRegistry::default()),
            Arc::new(assistd_voice::NoVoiceInput::new()),
            Arc::new(assistd_voice::NoContinuousListener::new()),
            assistd_voice::VoiceOutputController::new(Arc::new(assistd_voice::NoVoiceOutput), true),
        ))
    }

    #[tokio::test]
    async fn graceful_shutdown_waits_for_in_flight_stream() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("assistd.sock");
        let (tx, rx) = oneshot::channel::<()>();
        let server_path = path.clone();
        let state = state_with_backend_and_grace(
            Arc::new(SlowBackend {
                deltas: 3,
                pause: std::time::Duration::from_millis(50),
            }),
            5,
        );
        let server = tokio::spawn(async move {
            serve_at(&server_path, state, async {
                let _ = rx.await;
            })
            .await
            .unwrap();
        });

        wait_for_listener(&path).await;

        let stream = UnixStream::connect(&path).await.unwrap();
        let (read, mut write) = stream.into_split();
        write
            .write_all(br#"{"type":"query","id":"s1","text":"go"}"#)
            .await
            .unwrap();
        write.write_all(b"\n").await.unwrap();
        write.shutdown().await.unwrap();

        let mut reader = BufReader::new(read);

        // Wait for the first Delta, then trigger shutdown mid-stream.
        let mut first = String::new();
        let n = reader.read_line(&mut first).await.unwrap();
        assert!(n > 0, "expected first Delta to arrive");
        let ev: Event = serde_json::from_str(first.trim()).unwrap();
        assert!(matches!(ev, Event::Delta { .. }));

        tx.send(()).unwrap();

        let mut seen_delta = 1;
        let mut seen_done = false;
        loop {
            let mut line = String::new();
            let n = reader.read_line(&mut line).await.unwrap();
            if n == 0 {
                break;
            }
            let event: Event = serde_json::from_str(line.trim()).unwrap();
            match event {
                Event::Delta { .. } => seen_delta += 1,
                Event::Done { .. } => {
                    seen_done = true;
                    break;
                }
                other => panic!("unexpected event during graceful shutdown: {other:?}"),
            }
        }
        assert_eq!(seen_delta, 3, "expected all 3 deltas to arrive");
        assert!(seen_done, "expected terminal Done before connection close");

        server.await.unwrap();
    }

    #[tokio::test]
    async fn graceful_shutdown_aborts_after_grace_timeout() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("assistd.sock");
        let (tx, rx) = oneshot::channel::<()>();
        let server_path = path.clone();
        // Grace = 0 (no wait). Shutdown aborts the stuck connection
        // immediately after the accept loop exits.
        let state = state_with_backend_and_grace(Arc::new(StuckBackend), 0);
        let server = tokio::spawn(async move {
            serve_at(&server_path, state, async {
                let _ = rx.await;
            })
            .await
            .unwrap();
        });

        wait_for_listener(&path).await;

        let stream = UnixStream::connect(&path).await.unwrap();
        let (read, mut write) = stream.into_split();
        write
            .write_all(br#"{"type":"query","id":"s2","text":"hang"}"#)
            .await
            .unwrap();
        write.write_all(b"\n").await.unwrap();
        write.shutdown().await.unwrap();

        let mut reader = BufReader::new(read);
        let mut first = String::new();
        let n = reader.read_line(&mut first).await.unwrap();
        assert!(n > 0, "expected first Delta from StuckBackend");

        tx.send(()).unwrap();

        tokio::time::timeout(std::time::Duration::from_secs(2), server)
            .await
            .expect("server did not exit within 2s of shutdown")
            .unwrap();
    }

    async fn with_server<F, Fut, R>(state: Arc<AppState>, body: F) -> R
    where
        F: FnOnce(PathBuf) -> Fut,
        Fut: std::future::Future<Output = R>,
    {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("assistd.sock");
        let (tx, rx) = oneshot::channel::<()>();
        let server_path = path.clone();
        let server = tokio::spawn(async move {
            serve_at(&server_path, state, async {
                let _ = rx.await;
            })
            .await
            .unwrap();
        });
        wait_for_listener(&path).await;
        let out = body(path).await;
        tx.send(()).unwrap();
        server.await.unwrap();
        out
    }

    #[tokio::test]
    async fn set_presence_to_sleeping_returns_presence_then_done() {
        with_server(test_state(), |path| async move {
            let events = send_request_collect_events(
                &path,
                r#"{"type":"set_presence","id":"sp-1","target":"sleeping"}"#,
            )
            .await;
            assert!(matches!(
                events.first(),
                Some(Event::Presence { id, state }) if id == "sp-1" && *state == PresenceState::Sleeping
            ), "got {events:?}");
            assert!(matches!(events.last(), Some(Event::Done { id }) if id == "sp-1"));
        })
        .await;
    }

    #[tokio::test]
    async fn set_presence_to_active_when_active_is_idempotent() {
        with_server(test_state(), |path| async move {
            let events = send_request_collect_events(
                &path,
                r#"{"type":"set_presence","id":"sp-2","target":"active"}"#,
            )
            .await;
            assert!(matches!(
                events.first(),
                Some(Event::Presence { id, state }) if id == "sp-2" && *state == PresenceState::Active
            ), "got {events:?}");
            assert!(matches!(events.last(), Some(Event::Done { .. })));
        })
        .await;
    }

    #[tokio::test]
    async fn get_presence_returns_active_for_active_stub() {
        with_server(test_state(), |path| async move {
            let events = send_request_collect_events(
                &path,
                r#"{"type":"get_presence","id":"gp-1"}"#,
            )
            .await;
            assert!(matches!(
                events.first(),
                Some(Event::Presence { id, state }) if id == "gp-1" && *state == PresenceState::Active
            ), "got {events:?}");
            assert!(matches!(events.last(), Some(Event::Done { .. })));
        })
        .await;
    }

    #[tokio::test]
    async fn cycle_from_active_attempts_drowse_and_errors_on_stub() {
        with_server(test_state(), |path| async move {
            let events =
                send_request_collect_events(&path, r#"{"type":"cycle","id":"cy-1"}"#).await;
            assert!(
                matches!(
                    events.last(),
                    Some(Event::Error { id, .. }) if id == "cy-1"
                ),
                "got {events:?}"
            );
        })
        .await;
    }

    #[tokio::test]
    async fn ptt_start_errors_with_no_voice_input_stub() {
        with_server(test_state(), |path| async move {
            let events =
                send_request_collect_events(&path, r#"{"type":"ptt_start","id":"ptt-1"}"#).await;
            assert!(
                matches!(
                    events.last(),
                    Some(Event::Error { id, message })
                        if id == "ptt-1" && message.contains("not enabled")
                ),
                "got {events:?}"
            );
        })
        .await;
    }

    #[tokio::test]
    async fn ptt_stop_errors_with_no_voice_input_stub() {
        with_server(test_state(), |path| async move {
            let events =
                send_request_collect_events(&path, r#"{"type":"ptt_stop","id":"ptt-2"}"#).await;
            assert!(
                matches!(
                    events.last(),
                    Some(Event::Error { id, .. }) if id == "ptt-2"
                ),
                "got {events:?}"
            );
        })
        .await;
    }

    #[tokio::test]
    async fn listen_stop_when_inactive_returns_listenstate_then_done() {
        with_server(test_state(), |path| async move {
            let events =
                send_request_collect_events(&path, r#"{"type":"listen_stop","id":"ls-stop"}"#)
                    .await;
            assert!(
                matches!(
                    events.first(),
                    Some(Event::ListenState { id, active: false }) if id == "ls-stop"
                ),
                "got {events:?}"
            );
            assert!(matches!(events.last(), Some(Event::Done { id }) if id == "ls-stop"));
        })
        .await;
    }

    #[tokio::test]
    async fn listen_toggle_from_inactive_attempts_start_and_errors_on_stub() {
        with_server(test_state(), |path| async move {
            let events =
                send_request_collect_events(&path, r#"{"type":"listen_toggle","id":"lt-1"}"#).await;
            assert!(
                matches!(
                    events.last(),
                    Some(Event::Error { id, message })
                        if id == "lt-1" && message.contains("not enabled")
                ),
                "got {events:?}"
            );
        })
        .await;
    }

    #[tokio::test]
    async fn voice_toggle_flips_enabled_and_returns_voiceoutputstate_then_done() {
        // Default test_state builds the controller with enabled=true.
        with_server(test_state(), |path| async move {
            let events =
                send_request_collect_events(&path, r#"{"type":"voice_toggle","id":"vt-1"}"#).await;
            assert!(
                matches!(
                    events.first(),
                    Some(Event::VoiceOutputState { id, enabled: false }) if id == "vt-1"
                ),
                "got {events:?}"
            );
            assert!(matches!(events.last(), Some(Event::Done { id }) if id == "vt-1"));
        })
        .await;
    }

    #[tokio::test]
    async fn voice_skip_returns_voiceoutputstate_with_enabled_unchanged_then_done() {
        with_server(test_state(), |path| async move {
            let events =
                send_request_collect_events(&path, r#"{"type":"voice_skip","id":"vs-1"}"#).await;
            // Skip drops queued audio but leaves the enabled flag.
            assert!(
                matches!(
                    events.first(),
                    Some(Event::VoiceOutputState { id, enabled: true }) if id == "vs-1"
                ),
                "got {events:?}"
            );
            assert!(matches!(events.last(), Some(Event::Done { .. })));
        })
        .await;
    }

    #[tokio::test]
    async fn confirm_response_as_initial_request_returns_error() {
        with_server(test_state(), |path| async move {
            let events = send_request_collect_events(
                &path,
                r#"{"type":"confirm_response","id":"cr-1","confirm_id":"x","allow":true}"#,
            )
            .await;
            assert!(
                matches!(
                    events.last(),
                    Some(Event::Error { id, message })
                        if id == "cr-1" && message.contains("ConfirmResponse")
                ),
                "got {events:?}"
            );
        })
        .await;
    }

    #[tokio::test]
    async fn unmatched_mid_stream_confirm_response_does_not_crash() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("assistd.sock");
        let (tx, rx) = oneshot::channel::<()>();
        let server_path = path.clone();
        let state = test_state();
        let server = tokio::spawn(async move {
            serve_at(&server_path, state, async {
                let _ = rx.await;
            })
            .await
            .unwrap();
        });
        wait_for_listener(&path).await;

        let stream = UnixStream::connect(&path).await.unwrap();
        let (read, mut write) = stream.into_split();
        write
            .write_all(br#"{"type":"query","id":"q1","text":"ping"}"#)
            .await
            .unwrap();
        write.write_all(b"\n").await.unwrap();
        write
            .write_all(
                br#"{"type":"confirm_response","id":"cr-x","confirm_id":"missing","allow":false}"#,
            )
            .await
            .unwrap();
        write.write_all(b"\n").await.unwrap();
        write.shutdown().await.unwrap();

        let mut reader = BufReader::new(read);
        let mut events = Vec::new();
        loop {
            let mut line = String::new();
            let n = reader.read_line(&mut line).await.unwrap();
            if n == 0 {
                break;
            }
            let event: Event = serde_json::from_str(line.trim()).unwrap();
            let terminal = event.is_terminal();
            events.push(event);
            if terminal {
                break;
            }
        }
        assert!(
            matches!(events.last(), Some(Event::Done { id }) if id == "q1"),
            "got {events:?}"
        );

        tx.send(()).unwrap();
        server.await.unwrap();
    }

    #[tokio::test]
    async fn get_voice_state_reports_default_enabled_true() {
        with_server(test_state(), |path| async move {
            let events =
                send_request_collect_events(&path, r#"{"type":"get_voice_state","id":"gvs-1"}"#)
                    .await;
            assert!(
                matches!(
                    events.first(),
                    Some(Event::VoiceOutputState { id, enabled: true }) if id == "gvs-1"
                ),
                "got {events:?}"
            );
            assert!(matches!(events.last(), Some(Event::Done { .. })));
        })
        .await;
    }

    // ---------------------------------------------------------------
    // Concurrent IPC sessions: state isolation, protocol-version
    // tolerance, mixed read/write traffic. Complements the existing
    // `handles_concurrent_connections` (which is 16-way EchoBackend
    // basic concurrency) by stressing presence transitions and
    // protocol-evolution paths.
    // ---------------------------------------------------------------

    #[tokio::test]
    async fn concurrent_set_presence_to_sleeping_is_idempotent_under_fanout() {
        with_server(test_state(), |path| async move {
            let mut handles = Vec::new();
            for i in 0..32 {
                let p = path.clone();
                handles.push(tokio::spawn(async move {
                    let id = format!("sp-{i}");
                    let body = format!(
                        r#"{{"type":"set_presence","id":"{id}","target":"sleeping"}}"#
                    );
                    let events = send_request_collect_events(&p, &body).await;
                    assert!(
                        matches!(
                            events.first(),
                            Some(Event::Presence { id: pid, state }) if pid == &id && *state == PresenceState::Sleeping
                        ),
                        "conn {i}: missing initial Presence{{Sleeping}}: {events:?}"
                    );
                    assert!(
                        matches!(events.last(), Some(Event::Done { id: did }) if did == &id),
                        "conn {i}: missing terminal Done: {events:?}"
                    );
                    assert!(
                        !events.iter().any(|e| matches!(e, Event::Error { .. })),
                        "conn {i}: unexpected Error event in {events:?}"
                    );
                }));
            }
            for h in handles {
                h.await.unwrap();
            }
        })
        .await;
    }

    #[tokio::test]
    async fn mixed_query_and_set_presence_does_not_drop_events() {
        let state = state_with_backend_and_grace(
            Arc::new(SlowBackend {
                deltas: 3,
                pause: std::time::Duration::from_millis(50),
            }),
            5,
        );
        with_server(state, |path| async move {
            let mut handles = Vec::new();
            for i in 0..16 {
                let p = path.clone();
                handles.push(tokio::spawn(async move {
                    let id = format!("mix-{i}");
                    let body = if i % 2 == 0 {
                        format!(r#"{{"type":"query","id":"{id}","text":"q{i}"}}"#)
                    } else {
                        format!(r#"{{"type":"set_presence","id":"{id}","target":"active"}}"#)
                    };
                    let events = send_request_collect_events(&p, &body).await;
                    assert!(
                        matches!(events.last(), Some(Event::Done { id: did }) if did == &id),
                        "conn {i}: expected Done, got {events:?}"
                    );
                    assert!(
                        !events.iter().any(|e| matches!(e, Event::Error { .. })),
                        "conn {i}: unexpected Error in {events:?}"
                    );
                }));
            }
            for h in handles {
                h.await.unwrap();
            }
        })
        .await;
    }

    #[tokio::test]
    async fn duplicate_request_id_across_connections_does_not_leak_events() {
        with_server(test_state(), |path| async move {
            let mut handles = Vec::new();
            for i in 0..4 {
                let p = path.clone();
                let text = format!("t{i}");
                handles.push(tokio::spawn(async move {
                    let body = format!(r#"{{"type":"query","id":"shared","text":"{text}"}}"#);
                    let events = send_request_collect_events(&p, &body).await;
                    assert_eq!(
                        events.len(),
                        2,
                        "conn {i}: expected 2 events, got {events:?}"
                    );
                    match &events[0] {
                        Event::Delta { id, text: t } => {
                            assert_eq!(id, "shared");
                            assert_eq!(t, &text);
                        }
                        other => panic!("conn {i}: expected Delta, got {other:?}"),
                    }
                    match &events[1] {
                        Event::Done { id } => assert_eq!(id, "shared"),
                        other => panic!("conn {i}: expected Done, got {other:?}"),
                    }
                }));
            }
            for h in handles {
                h.await.unwrap();
            }
        })
        .await;
    }

    #[tokio::test]
    async fn query_ignores_unknown_top_level_fields() {
        // Extra unrecognized fields on a Query payload (forward-compat
        // hook for future additive evolution) must not break parsing.
        with_server(test_state(), |path| async move {
            let events = send_request_collect_events(
                &path,
                r#"{"type":"query","id":"future","text":"hi","extra":"ignored"}"#,
            )
            .await;
            assert!(
                matches!(events.first(), Some(Event::Delta { id, .. }) if id == "future"),
                "expected Delta, got {events:?}"
            );
            assert!(matches!(events.last(), Some(Event::Done { id }) if id == "future"));
        })
        .await;
    }

    /// Open a Subscribe connection without closing the write half (the
    /// daemon keeps the stream open for the lifetime of the
    /// subscription). Returns the write half (so the test can drop it
    /// to signal shutdown) plus a buffered reader for events.
    async fn open_subscribe_stream(
        path: &Path,
        body: &str,
    ) -> (
        tokio::net::unix::OwnedWriteHalf,
        BufReader<tokio::net::unix::OwnedReadHalf>,
    ) {
        let stream = UnixStream::connect(path).await.unwrap();
        let (read, mut write) = stream.into_split();
        write.write_all(body.as_bytes()).await.unwrap();
        write.write_all(b"\n").await.unwrap();
        write.flush().await.unwrap();
        (write, BufReader::new(read))
    }

    /// Drain a Subscribe stream until either the per-line timeout
    /// fires (the stream is idle) or the connection closes. Used by
    /// integration tests to capture every event a subscriber received
    /// up to a steady state.
    async fn drain_subscribe(
        reader: &mut BufReader<tokio::net::unix::OwnedReadHalf>,
        idle: std::time::Duration,
    ) -> Vec<Event> {
        let mut out = Vec::new();
        loop {
            let mut line = String::new();
            match tokio::time::timeout(idle, reader.read_line(&mut line)).await {
                Ok(Ok(0)) => break,
                Ok(Ok(_)) => match serde_json::from_str::<Event>(line.trim()) {
                    Ok(ev) => out.push(ev),
                    Err(e) => panic!("subscriber received invalid event {line:?}: {e}"),
                },
                Ok(Err(_)) | Err(_) => break,
            }
        }
        out
    }

    #[tokio::test]
    async fn subscribe_receives_query_events_from_other_client() {
        with_server(test_state(), |path| async move {
            // Open client A as a passive subscriber with the default
            // (empty) filter — it should see every broadcast-eligible
            // event the daemon emits.
            let (_write_a, mut read_a) = open_subscribe_stream(
                &path,
                r#"{"type":"subscribe","id":"sub-1","filter":{"kinds":[]}}"#,
            )
            .await;

            // Give the Subscribe handler a moment to register on the
            // broadcast bus before client B starts emitting events.
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;

            // Client B runs a normal Query. Per-request flow must be
            // unchanged (regression check).
            let b_events = send_request_collect_events(
                &path,
                r#"{"type":"query","id":"q-1","text":"hello"}"#,
            )
            .await;
            assert!(
                matches!(b_events.first(), Some(Event::Delta { id, text }) if id == "q-1" && text == "hello"),
                "client B regression: expected Delta(q-1, hello) first, got {b_events:?}"
            );
            assert!(
                matches!(b_events.last(), Some(Event::Done { id }) if id == "q-1"),
                "client B regression: expected terminal Done(q-1), got {b_events:?}"
            );

            // Client A should have observed the Query's Delta and
            // Done events on the bus, plus at least one LastDelta
            // (the on-Done flush is mandatory; the in-stream
            // debounced emission may or may not fire depending on
            // timing, but EchoBackend emits one Delta which sits
            // ≥100 ms after the initial Instant::now() - DEBOUNCE,
            // so the in-stream emission also fires here).
            let a_events =
                drain_subscribe(&mut read_a, std::time::Duration::from_millis(200)).await;
            assert!(
                a_events
                    .iter()
                    .any(|e| matches!(e, Event::Delta { id, text } if id == "q-1" && text == "hello")),
                "subscriber missed Delta from q-1; got {a_events:?}"
            );
            assert!(
                a_events
                    .iter()
                    .any(|e| matches!(e, Event::Done { id } if id == "q-1")),
                "subscriber missed Done from q-1; got {a_events:?}"
            );
            assert!(
                a_events
                    .iter()
                    .any(|e| matches!(e, Event::LastDelta { id, text } if id == "q-1" && text == "hello")),
                "subscriber missed LastDelta(q-1, 'hello'); got {a_events:?}"
            );
        })
        .await;
    }

    #[tokio::test]
    async fn subscribe_filter_rejects_unmatched_kinds() {
        with_server(test_state(), |path| async move {
            // Client A subscribes with a Presence-only filter.
            let (_write_a, mut read_a) = open_subscribe_stream(
                &path,
                r#"{"type":"subscribe","id":"sub-2","filter":{"kinds":["presence"]}}"#,
            )
            .await;
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;

            // A Query should produce Delta/Done on the bus, but the
            // Presence-only filter must drop them.
            let _ =
                send_request_collect_events(&path, r#"{"type":"query","id":"q-2","text":"hi"}"#)
                    .await;
            // Now flip presence — that *does* match the filter.
            let _ = send_request_collect_events(
                &path,
                r#"{"type":"set_presence","id":"sp-1","target":"sleeping"}"#,
            )
            .await;

            let a_events =
                drain_subscribe(&mut read_a, std::time::Duration::from_millis(200)).await;
            // Nothing from the Query: neither Delta(q-2) nor Done(q-2).
            assert!(
                a_events.iter().all(|e| !matches!(
                    e,
                    Event::Delta { id, .. } | Event::Done { id, .. } if id == "q-2"
                )),
                "Presence-only filter leaked Query events: {a_events:?}"
            );
            // But the Presence event passes.
            assert!(
                a_events.iter().any(|e| matches!(
                    e,
                    Event::Presence { id, .. } if id == "sp-1"
                )),
                "expected Presence(sp-1) on subscriber, got {a_events:?}"
            );
        })
        .await;
    }

    #[tokio::test]
    async fn subscribe_does_not_self_loop() {
        // A subscriber's own forward task must not re-broadcast the
        // events it writes (those events came from the bus in the
        // first place). Verify by attaching a subscriber and waiting
        // — no other client is connected, so the only way A could
        // see an event is via a feedback loop.
        with_server(test_state(), |path| async move {
            let (_write_a, mut read_a) = open_subscribe_stream(
                &path,
                r#"{"type":"subscribe","id":"sub-3","filter":{"kinds":[]}}"#,
            )
            .await;

            let a_events =
                drain_subscribe(&mut read_a, std::time::Duration::from_millis(150)).await;
            assert!(
                a_events.is_empty(),
                "idle subscriber emitted unexpected events: {a_events:?}"
            );
        })
        .await;
    }
}
