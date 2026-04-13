use crate::AppState;
use assistd_ipc::{Event, Request};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

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
    tokio::pin!(shutdown);
    loop {
        tokio::select! {
            _ = &mut shutdown => {
                info!("shutting down socket listener");
                return Ok(());
            }
            accepted = listener.accept() => {
                match accepted {
                    Ok((stream, _addr)) => {
                        let conn_state = state.clone();
                        tokio::spawn(async move {
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
        }
    }
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
    let dispatcher = tokio::spawn(async move { dispatch_state.dispatch(req, tx).await });

    while let Some(event) = rx.recv().await {
        write_event(&mut write_half, &event).await?;
    }

    match dispatcher.await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => error!("dispatch error: {e:#}"),
        Err(e) => error!("dispatch task panicked: {e}"),
    }

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
            Arc::new(assistd_llm::EchoBackend),
            PresenceManager::stub(PresenceState::Active),
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
}
