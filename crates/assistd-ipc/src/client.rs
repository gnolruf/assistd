//! Unix-socket client for the IPC protocol: [`IpcClient::one_shot`]
//! for request/stream calls and [`IpcClient::open_dialog`] when the
//! client must answer mid-stream prompts on the same connection.

use std::path::{Path, PathBuf};

use thiserror::Error;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::net::unix::{OwnedReadHalf, OwnedWriteHalf};

use crate::{Event, Request, socket_path};

/// Errors produced by the IPC client.
#[derive(Debug, Error)]
pub enum IpcClientError {
    /// The daemon socket couldn't be reached, usually because the
    /// daemon isn't running.
    #[error("daemon not reachable at {path}: {source}")]
    NotReachable {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("ipc i/o error: {0}")]
    Io(#[from] std::io::Error),
    #[error("ipc json error: {0}")]
    Json(#[from] serde_json::Error),
    /// Daemon closed the connection before emitting `Done` or `Error`.
    #[error("daemon closed connection mid-stream")]
    DaemonClosed,
}

pub type Result<T> = std::result::Result<T, IpcClientError>;

/// Connection factory bound to one socket path.
#[derive(Debug, Clone)]
pub struct IpcClient {
    socket_path: PathBuf,
}

impl IpcClient {
    /// Client for the per-user default [`socket_path`].
    pub fn new() -> Self {
        Self {
            socket_path: socket_path(),
        }
    }

    /// Client for the socket at `p`.
    pub fn with_path(p: impl Into<PathBuf>) -> Self {
        Self {
            socket_path: p.into(),
        }
    }

    /// The socket path this client connects to.
    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    /// Send one request, close the write half, and stream events until
    /// the daemon emits `Done` or `Error`.
    pub async fn one_shot(&self, req: Request) -> Result<EventStream> {
        let stream = self.connect().await?;
        let (read, mut write) = stream.into_split();
        write_frame(&mut write, &req).await?;
        write.shutdown().await?;
        Ok(EventStream::new(read))
    }

    /// Send the initial request but keep the write half open so further
    /// requests, such as a [`Request::ConfirmResponse`], can follow.
    pub async fn open_dialog(&self, initial: Request) -> Result<DialogConnection> {
        let stream = self.connect().await?;
        let (read, mut write) = stream.into_split();
        write_frame(&mut write, &initial).await?;
        Ok(DialogConnection {
            write,
            events: EventStream::new(read),
        })
    }

    async fn connect(&self) -> Result<UnixStream> {
        UnixStream::connect(&self.socket_path)
            .await
            .map_err(|source| IpcClientError::NotReachable {
                path: self.socket_path.clone(),
                source,
            })
    }
}

impl Default for IpcClient {
    fn default() -> Self {
        Self::new()
    }
}

async fn write_frame(write: &mut OwnedWriteHalf, req: &Request) -> Result<()> {
    let mut payload = serde_json::to_string(req)?;
    payload.push('\n');
    write.write_all(payload.as_bytes()).await?;
    write.flush().await?;
    Ok(())
}

/// Stream of [`Event`]s read from a daemon connection.
pub struct EventStream {
    inner: tokio::io::Lines<BufReader<OwnedReadHalf>>,
}

impl std::fmt::Debug for EventStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventStream").finish_non_exhaustive()
    }
}

impl EventStream {
    fn new(read: OwnedReadHalf) -> Self {
        Self {
            inner: BufReader::new(read).lines(),
        }
    }

    /// The next event, or `Ok(None)` when the daemon closed the stream
    /// without a terminal event.
    pub async fn next_event(&mut self) -> Result<Option<Event>> {
        match self.inner.next_line().await? {
            None => Ok(None),
            Some(line) => Ok(Some(serde_json::from_str(&line)?)),
        }
    }

    /// Collect events up to and including the terminal one. A stream
    /// closed early is [`IpcClientError::DaemonClosed`].
    pub async fn collect(mut self) -> Result<Vec<Event>> {
        let mut out = Vec::new();
        loop {
            match self.next_event().await? {
                Some(ev) => {
                    let terminal = ev.is_terminal();
                    out.push(ev);
                    if terminal {
                        return Ok(out);
                    }
                }
                None => return Err(IpcClientError::DaemonClosed),
            }
        }
    }
}

/// Bidirectional connection: read events as they arrive, send further
/// requests at any time.
pub struct DialogConnection {
    write: OwnedWriteHalf,
    events: EventStream,
}

impl DialogConnection {
    /// See [`EventStream::next_event`].
    pub async fn next_event(&mut self) -> Result<Option<Event>> {
        self.events.next_event().await
    }

    /// Send a further request, such as a [`Request::ConfirmResponse`],
    /// on this connection.
    pub async fn send(&mut self, req: Request) -> Result<()> {
        write_frame(&mut self.write, &req).await
    }

    /// Close the write half. Events remain readable until the daemon
    /// emits its terminal event.
    pub async fn close_write(&mut self) -> Result<()> {
        self.write.shutdown().await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    use tokio::net::UnixListener;

    /// Mock daemon that accepts one connection, reads and parses one
    /// request line, writes `responses`, then closes its write half.
    /// The returned `TempDir` owns the socket and must outlive the test.
    fn mock_server(
        responses: Vec<Event>,
    ) -> (tempfile::TempDir, PathBuf, tokio::task::JoinHandle<()>) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mock.sock");
        let listener = UnixListener::bind(&path).unwrap();
        let h = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let (read, mut write) = stream.into_split();
            let mut reader = BufReader::new(read);
            let mut line = String::new();
            reader.read_line(&mut line).await.unwrap();
            let _: Request = serde_json::from_str(line.trim()).unwrap();
            for ev in responses {
                let mut out = serde_json::to_string(&ev).unwrap();
                out.push('\n');
                write.write_all(out.as_bytes()).await.unwrap();
            }
            write.flush().await.unwrap();
            write.shutdown().await.unwrap();
        });
        (dir, path, h)
    }

    #[tokio::test]
    async fn one_shot_collects_events_until_done() {
        let events = vec![
            Event::Delta {
                id: "r".into(),
                text: "hello".into(),
            },
            Event::Done { id: "r".into() },
        ];
        let (_dir, path, h) = mock_server(events.clone());

        let client = IpcClient::with_path(path);
        let stream = client
            .one_shot(Request::query("r", "hi"))
            .await
            .expect("one_shot");
        assert_eq!(stream.collect().await.expect("collect"), events);
        h.await.unwrap();
    }

    #[tokio::test]
    async fn one_shot_reports_not_reachable_when_no_listener() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing.sock");
        let client = IpcClient::with_path(path);
        let err = client
            .one_shot(Request::query("r", "x"))
            .await
            .expect_err("expected NotReachable");
        assert!(
            matches!(err, IpcClientError::NotReachable { .. }),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn collect_errors_on_premature_close() {
        let (_dir, path, h) = mock_server(vec![Event::Delta {
            id: "r".into(),
            text: "incomplete".into(),
        }]);
        let client = IpcClient::with_path(path);
        let stream = client.one_shot(Request::query("r", "x")).await.unwrap();
        let err = stream.collect().await.expect_err("expected DaemonClosed");
        assert!(matches!(err, IpcClientError::DaemonClosed), "{err:?}");
        h.await.unwrap();
    }

    #[tokio::test]
    async fn dialog_can_send_after_initial_request() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dialog.sock");
        let listener = UnixListener::bind(&path).unwrap();
        let h = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let (read, mut write) = stream.into_split();
            let mut reader = BufReader::new(read);

            let mut first = String::new();
            reader.read_line(&mut first).await.unwrap();
            let _: Request = serde_json::from_str(first.trim()).unwrap();

            let cr = Event::ConfirmRequest {
                id: "r".into(),
                confirm_id: "c1".into(),
                tool: "bash".into(),
                script: "rm -rf /tmp/x".into(),
                matched_pattern: "rm -rf".into(),
            };
            let mut out = serde_json::to_string(&cr).unwrap();
            out.push('\n');
            write.write_all(out.as_bytes()).await.unwrap();
            write.flush().await.unwrap();

            let mut second = String::new();
            reader.read_line(&mut second).await.unwrap();
            let req: Request = serde_json::from_str(second.trim()).unwrap();
            assert!(matches!(req, Request::ConfirmResponse { allow: true, .. }));

            let mut done = serde_json::to_string(&Event::Done { id: "r".into() }).unwrap();
            done.push('\n');
            write.write_all(done.as_bytes()).await.unwrap();
            write.flush().await.unwrap();
        });

        let client = IpcClient::with_path(&path);
        let mut conn = client.open_dialog(Request::query("r", "go")).await.unwrap();

        let ev = conn.next_event().await.unwrap().expect("ConfirmRequest");
        match ev {
            Event::ConfirmRequest { confirm_id, .. } => {
                conn.send(Request::ConfirmResponse {
                    id: "r".into(),
                    confirm_id,
                    allow: true,
                })
                .await
                .unwrap();
            }
            other => panic!("expected ConfirmRequest, got {other:?}"),
        }

        let done = conn.next_event().await.unwrap().expect("Done");
        assert_eq!(done, Event::Done { id: "r".into() });
        h.await.unwrap();
    }
}
