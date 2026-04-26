use crate::AppState;
use assistd_ipc::{Event, Request};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::mpsc;
use tokio::task::JoinSet;
use tracing::{Instrument, debug, error, info, warn};

const EVENT_CHANNEL_CAPACITY: usize = 32;

/// Errors produced by the socket listener and per-connection handlers.
#[derive(Debug, Error)]
pub enum SocketError {
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

pub async fn serve<F>(state: Arc<AppState>, shutdown: F) -> Result<(), SocketError>
where
    F: Future<Output = ()>,
{
    let path = assistd_ipc::socket_path();
    serve_at(&path, state, shutdown).await
}

pub async fn serve_at<F>(path: &Path, state: Arc<AppState>, shutdown: F) -> Result<(), SocketError>
where
    F: Future<Output = ()>,
{
    if path.exists() {
        warn!("removing stale socket file at {}", path.display());
        std::fs::remove_file(path).map_err(|source| SocketError::StaleCleanup {
            path: path.to_path_buf(),
            source,
        })?;
    }

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
                        let conn_state = state.clone();
                        connections.spawn(async move {
                            if let Err(e) = handle_connection(stream, conn_state).await {
                                error!("connection error: {e}");
                            }
                        });
                    }
                    Err(e) => {
                        error!("accept error: {e}");
                    }
                }
            }
            // Reap finished connections eagerly so the set doesn't grow
            // unbounded. `join_next` yields `None` when the set is empty,
            // which would busy-loop; gate on `len()` to avoid that.
            Some(res) = connections.join_next(), if !connections.is_empty() => {
                if let Err(e) = res
                    && e.is_panic()
                {
                    error!("connection task panicked: {e}");
                }
            }
        }
    }

    // Stop accepting and drain in-flight handlers with bounded grace so
    // streaming responses can emit their final `Done` event before the
    // daemon exits.
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
    let n = reader.read_line(&mut line).await?;
    if n == 0 {
        debug!("client disconnected without sending a request");
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

    // Span carries the request id and kind so all child events emitted
    // by `dispatch` (LLM stream, tool calls, TTS sentences) and by the
    // forward loop are correlatable in `RUST_LOG=assistd=debug` output
    // when several requests are in flight on different connections.
    let span = tracing::info_span!("ipc", id = %req.id(), req = req.kind());

    // Dispatch and event-forwarding run concurrently on this task rather
    // than on a spawned child. If the task is cancelled (e.g. the outer
    // accept loop hits its shutdown grace and calls `JoinSet::shutdown`),
    // both halves are torn down together — the dispatcher can't be
    // orphaned.
    let dispatch_fut = async move { dispatch_state.dispatch(req, tx).await };
    let forward_fut = async {
        while let Some(event) = rx.recv().await {
            write_event(&mut write_half, &event).await?;
        }
        Ok::<(), SocketError>(())
    };

    let (dispatch_res, forward_res) = async { tokio::join!(dispatch_fut, forward_fut) }
        .instrument(span)
        .await;

    if let Err(e) = dispatch_res {
        error!("dispatch error: {e:#}");
    }
    forward_res?;

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

    /// Send a raw request line and collect every event the daemon emits
    /// until the stream terminates.
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
        // The default `test_state` builds with a `NoContinuousListener`
        // whose `start()` is a hard error — this asserts the error
        // propagation path. A real MicContinuousListener has different
        // semantics (it actually starts) but needs cpal + whisper to
        // exercise end-to-end.
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
        ) -> anyhow::Result<()> {
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
        ) -> anyhow::Result<()> {
            Ok(())
        }

        async fn push_tool_results(
            &self,
            _results: Vec<assistd_llm::ToolResultPayload>,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        async fn step(
            &self,
            _tools: Vec<serde_json::Value>,
            tx: tokio::sync::mpsc::Sender<assistd_llm::LlmEvent>,
        ) -> anyhow::Result<assistd_llm::StepOutcome> {
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
        ) -> anyhow::Result<()> {
            let _ = tx
                .send(assistd_llm::LlmEvent::Delta {
                    text: "stuck".into(),
                })
                .await;
            // Park forever; this future is cancelled when the parent
            // task is aborted via JoinSet::shutdown.
            std::future::pending::<()>().await;
            Ok(())
        }

        async fn push_user(
            &self,
            _text: String,
            _attachments: Vec<assistd_tools::Attachment>,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        async fn push_tool_results(
            &self,
            _results: Vec<assistd_llm::ToolResultPayload>,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        async fn step(
            &self,
            _tools: Vec<serde_json::Value>,
            tx: tokio::sync::mpsc::Sender<assistd_llm::LlmEvent>,
        ) -> anyhow::Result<assistd_llm::StepOutcome> {
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

        // All remaining Deltas and the terminal Done must still arrive
        // before the stream closes, because grace (5s) comfortably
        // exceeds the remaining ~100ms of backend work.
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

        // Server task must exit promptly — the grace is 0, so the
        // connection is aborted as soon as shutdown fires. Bound with
        // a 2s timeout so a regression doesn't hang the test forever.
        tokio::time::timeout(std::time::Duration::from_secs(2), server)
            .await
            .expect("server did not exit within 2s of shutdown")
            .unwrap();
    }

    // ---------------------------------------------------------------
    // Wire-protocol contract tests: one happy (or expected-failure)
    // path per `assistd_ipc::Request` variant. Each variant must
    // produce a sensible event sequence terminated by `Done` or
    // `Error`. Adding/renaming a variant should fail at least one of
    // these tests, so a protocol regression is caught at CI time
    // rather than at runtime in a TUI session.
    // ---------------------------------------------------------------

    /// Spin up `serve_at` against the given state, hand the socket
    /// path to the body, then trigger shutdown and join the server.
    /// Removes the boilerplate from per-variant tests below.
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
        // Active → Sleeping is the stub-friendly path: it drops the
        // (None) llama service and flips the state field. drowse()
        // and Drowsy-bound wake() require a live HTTP server.
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
        // Cycle Active → Drowsy invokes `drowse()`, which calls
        // `control.unload_model()` against the stub's bogus
        // 127.0.0.1:1 endpoint. We don't care about the exact error
        // text, only that the IPC layer surfaces it as an `Error`
        // event with the matching id.
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
            // ptt_stop emits VoiceState::Transcribing first, then
            // surfaces the NoVoiceInput error.
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
        // NoContinuousListener::stop is idempotent — the daemon
        // emits ListenState{active:false} + Done even when nothing
        // was running.
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
        // Toggle delegates to start/stop based on `is_active()`. With
        // NoContinuousListener inactive, toggle routes to start,
        // which then errors with "not enabled".
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
}
