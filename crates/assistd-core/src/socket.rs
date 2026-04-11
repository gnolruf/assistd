use anyhow::{Context, Result};
use std::future::Future;
use std::path::{Path, PathBuf};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tracing::{debug, error, info, warn};

use crate::ipc::{self, Request, Response};

pub async fn serve<F>(shutdown: F) -> Result<()>
where
    F: Future<Output = ()>,
{
    let path = ipc::socket_path();
    serve_at(&path, shutdown).await
}

pub async fn serve_at<F>(path: &Path, shutdown: F) -> Result<()>
where
    F: Future<Output = ()>,
{
    if path.exists() {
        warn!("removing stale socket file at {}", path.display());
        std::fs::remove_file(path)
            .with_context(|| format!("failed to remove stale socket file at {}", path.display()))?;
    }

    let listener = UnixListener::bind(path)
        .with_context(|| format!("failed to bind unix socket at {}", path.display()))?;
    info!("listening on {}", path.display());

    let owned_path = PathBuf::from(path);
    let result = accept_loop(listener, shutdown).await;

    if let Err(e) = std::fs::remove_file(&owned_path) {
        warn!(
            "failed to remove socket file at {}: {}",
            owned_path.display(),
            e
        );
    }

    result
}

async fn accept_loop<F>(listener: UnixListener, shutdown: F) -> Result<()>
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
                        tokio::spawn(async move {
                            if let Err(e) = handle_connection(stream).await {
                                error!("connection error: {e:#}");
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

async fn handle_connection(stream: UnixStream) -> Result<()> {
    let (read_half, mut write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);
    let mut line = String::new();
    let n = reader.read_line(&mut line).await?;
    if n == 0 {
        debug!("client disconnected without sending a request");
        return Ok(());
    }

    let response = match serde_json::from_str::<Request>(line.trim()) {
        Ok(req) => dispatch(req),
        Err(e) => Response::Error {
            message: format!("invalid request: {e}"),
        },
    };

    let mut out = serde_json::to_string(&response)?;
    out.push('\n');
    write_half.write_all(out.as_bytes()).await?;
    write_half.flush().await?;
    write_half.shutdown().await?;
    Ok(())
}

fn dispatch(req: Request) -> Response {
    match req {
        // TODO: route to assistd-llm once the LLM subsystem is implemented.
        Request::Query { text } => Response::Response { text },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    use tokio::net::UnixStream;
    use tokio::sync::oneshot;

    async fn wait_for_listener(path: &Path) {
        for _ in 0..200 {
            if UnixStream::connect(path).await.is_ok() {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        panic!("listener at {} did not become ready", path.display());
    }

    async fn send_request(path: &Path, body: &str) -> String {
        let stream = UnixStream::connect(path).await.unwrap();
        let (read, mut write) = stream.into_split();
        write.write_all(body.as_bytes()).await.unwrap();
        write.write_all(b"\n").await.unwrap();
        write.shutdown().await.unwrap();
        let mut reader = BufReader::new(read);
        let mut line = String::new();
        reader.read_line(&mut line).await.unwrap();
        line
    }

    #[tokio::test]
    async fn echoes_query_text() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("assistd.sock");
        let (tx, rx) = oneshot::channel::<()>();
        let server_path = path.clone();
        let server = tokio::spawn(async move {
            serve_at(&server_path, async {
                let _ = rx.await;
            })
            .await
            .unwrap();
        });

        wait_for_listener(&path).await;

        let resp = send_request(&path, r#"{"type":"query","text":"ping"}"#).await;
        assert_eq!(resp.trim(), r#"{"type":"response","text":"ping"}"#);

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
        let server = tokio::spawn(async move {
            serve_at(&server_path, async {
                let _ = rx.await;
            })
            .await
            .unwrap();
        });

        for _ in 0..50 {
            if path.exists() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }

        let mut handles = Vec::new();
        for i in 0..16 {
            let p = path.clone();
            handles.push(tokio::spawn(async move {
                let body = format!(r#"{{"type":"query","text":"msg-{i}"}}"#);
                let resp = send_request(&p, &body).await;
                let expected = format!(r#"{{"type":"response","text":"msg-{i}"}}"#);
                assert_eq!(resp.trim(), expected);
            }));
        }
        for h in handles {
            h.await.unwrap();
        }

        tx.send(()).unwrap();
        server.await.unwrap();
    }

    #[tokio::test]
    async fn malformed_request_returns_error_response() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("assistd.sock");
        let (tx, rx) = oneshot::channel::<()>();
        let server_path = path.clone();
        let server = tokio::spawn(async move {
            serve_at(&server_path, async {
                let _ = rx.await;
            })
            .await
            .unwrap();
        });

        for _ in 0..50 {
            if path.exists() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }

        let resp = send_request(&path, "not json").await;
        let parsed: Response = serde_json::from_str(resp.trim()).unwrap();
        match parsed {
            Response::Error { .. } => {}
            other => panic!("expected error response, got {other:?}"),
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
        let server = tokio::spawn(async move {
            serve_at(&server_path, async {
                let _ = rx.await;
            })
            .await
            .unwrap();
        });

        wait_for_listener(&path).await;

        let resp = send_request(&path, r#"{"type":"query","text":"ok"}"#).await;
        assert_eq!(resp.trim(), r#"{"type":"response","text":"ok"}"#);

        tx.send(()).unwrap();
        server.await.unwrap();
    }
}
