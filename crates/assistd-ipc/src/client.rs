//! Unix-socket client for the IPC protocol: [`IpcClient::request`] to run one request to
//! its end, [`IpcClient::one_shot`] to read its events as a stream, and
//! [`IpcClient::open_dialog`] when the client must answer mid-stream prompts.

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

use thiserror::Error;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::net::unix::{OwnedReadHalf, OwnedWriteHalf};

use crate::{Event, EventBody, Hello, Request, VERSION, socket_path};

/// Errors produced by the IPC client.
#[derive(Debug, Error)]
pub enum IpcClientError {
    /// The socket couldn't be reached, usually because the daemon isn't running.
    #[error("daemon not reachable at {path}: {source}")]
    NotReachable {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    /// The socket is served by another user's process, so it may be squatted.
    #[error("socket at {path} is served by uid {peer_uid}, not this user; refusing to talk to it")]
    ForeignDaemon { path: PathBuf, peer_uid: u32 },
    /// The daemon is from another assistd version, so the two may not share a protocol;
    /// `daemon` is `None` when it is too old to say which.
    #[error(
        "the running daemon is {} but this client is assistd {VERSION}; restart the daemon",
        describe_daemon(.daemon.as_deref())
    )]
    VersionMismatch { daemon: Option<String> },
    #[error("ipc i/o error: {0}")]
    Io(#[from] io::Error),
    #[error("ipc json error: {0}")]
    Json(#[from] serde_json::Error),
    /// Daemon closed the connection before emitting `Done` or `Error`.
    #[error("daemon closed connection mid-stream")]
    DaemonClosed,
    /// The daemon ended the request with [`EventBody::Error`]; holds its message.
    #[error("daemon error: {0}")]
    Daemon(String),
}

/// Result of an IPC client operation.
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

    /// Client for the socket at `path`.
    pub fn with_path(path: impl Into<PathBuf>) -> Self {
        Self {
            socket_path: path.into(),
        }
    }

    /// The socket path this client connects to.
    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    /// Send one request, close the write half, and stream events until `Done` or `Error`.
    /// Fails with [`IpcClientError::VersionMismatch`] when the daemon is from another version.
    pub async fn one_shot(&self, req: Request) -> Result<EventStream> {
        let stream = self.connect().await?;
        let (read, mut write) = stream.into_split();
        if write_initial_frame(&mut write, &req).await? {
            write.shutdown().await?;
        }
        EventStream::greeted(read).await
    }

    /// Send one request and hand every event, the terminal one included, to `on_event`
    /// until `Done`. Fails with [`IpcClientError::Daemon`] on `Error`, with
    /// [`IpcClientError::DaemonClosed`] when the stream closes first, and with the first
    /// error `on_event` returns.
    pub async fn request<E>(
        &self,
        req: Request,
        mut on_event: impl FnMut(&Event) -> std::result::Result<(), E>,
    ) -> std::result::Result<(), E>
    where
        E: From<IpcClientError>,
    {
        let mut stream = self.one_shot(req).await?;
        loop {
            let event = stream
                .next_event()
                .await?
                .ok_or(IpcClientError::DaemonClosed)?;
            on_event(&event)?;
            match event.body {
                EventBody::Done => return Ok(()),
                EventBody::Error { message } => {
                    return Err(IpcClientError::Daemon(message).into());
                }
                _ => {}
            }
        }
    }

    /// [`request`](Self::request) for a caller with no use for the events.
    pub async fn send(&self, req: Request) -> Result<()> {
        self.request(req, |_| Ok(())).await
    }

    /// Send `initial` but keep the write half open for further requests, such as a
    /// [`crate::RequestBody::ConfirmResponse`]. Fails like [`one_shot`](Self::one_shot).
    pub async fn open_dialog(&self, initial: Request) -> Result<DialogConnection> {
        let stream = self.connect().await?;
        let (read, mut write) = stream.into_split();
        write_initial_frame(&mut write, &initial).await?;
        Ok(DialogConnection {
            write,
            events: EventStream::greeted(read).await?,
        })
    }

    async fn connect(&self) -> Result<UnixStream> {
        let stream = UnixStream::connect(&self.socket_path)
            .await
            .map_err(|source| IpcClientError::NotReachable {
                path: self.socket_path.clone(),
                source,
            })?;
        let peer_uid = stream.peer_cred()?.uid();
        if peer_uid != rustix::process::geteuid().as_raw() {
            return Err(IpcClientError::ForeignDaemon {
                path: self.socket_path.clone(),
                peer_uid,
            });
        }
        Ok(stream)
    }
}

impl Default for IpcClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Write the opening request; `false` when the daemon had already closed the connection,
/// leaving any [`EventBody::Error`] saying why for the event stream to report.
async fn write_initial_frame(write: &mut OwnedWriteHalf, req: &Request) -> Result<bool> {
    match write_frame(write, req).await {
        Ok(()) => Ok(true),
        Err(IpcClientError::Io(e))
            if matches!(
                e.kind(),
                io::ErrorKind::BrokenPipe | io::ErrorKind::ConnectionReset
            ) =>
        {
            Ok(false)
        }
        Err(e) => Err(e),
    }
}

fn describe_daemon(version: Option<&str>) -> String {
    version.map_or_else(
        || "an older assistd".to_owned(),
        |version| format!("assistd {version}"),
    )
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

impl fmt::Debug for EventStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EventStream").finish_non_exhaustive()
    }
}

impl EventStream {
    /// The stream behind the daemon's [`Hello`], which must come first and name this
    /// client's version.
    async fn greeted(read: OwnedReadHalf) -> Result<Self> {
        let mut inner = BufReader::new(read).lines();
        let greeting = inner
            .next_line()
            .await?
            .ok_or(IpcClientError::DaemonClosed)?;
        match serde_json::from_str::<Hello>(&greeting) {
            Ok(hello) if hello.daemon_version == VERSION => Ok(Self { inner }),
            Ok(hello) => Err(IpcClientError::VersionMismatch {
                daemon: Some(hello.daemon_version),
            }),
            Err(_) => Err(IpcClientError::VersionMismatch { daemon: None }),
        }
    }

    /// The next event; `Ok(None)` when the daemon closed the stream, and
    /// [`IpcClientError::Json`] for a line that is not an event this client knows.
    pub async fn next_event(&mut self) -> Result<Option<Event>> {
        let line = self.inner.next_line().await?;
        Ok(line.as_deref().map(serde_json::from_str).transpose()?)
    }

    /// Collect events through the terminal one; errors with [`IpcClientError::DaemonClosed`] if
    /// the stream closes first.
    pub async fn collect(mut self) -> Result<Vec<Event>> {
        let mut events = Vec::new();
        loop {
            match self.next_event().await? {
                Some(event) => {
                    let terminal = event.is_terminal();
                    events.push(event);
                    if terminal {
                        return Ok(events);
                    }
                }
                None => return Err(IpcClientError::DaemonClosed),
            }
        }
    }
}

/// Bidirectional connection: read events as they arrive and send further requests at any time.
#[derive(Debug)]
pub struct DialogConnection {
    write: OwnedWriteHalf,
    events: EventStream,
}

impl DialogConnection {
    /// See [`EventStream::next_event`].
    pub async fn next_event(&mut self) -> Result<Option<Event>> {
        self.events.next_event().await
    }

    /// Send a further request, such as a [`crate::RequestBody::ConfirmResponse`], on this connection.
    pub async fn send(&mut self, req: Request) -> Result<()> {
        write_frame(&mut self.write, &req).await
    }

    /// Close the write half; events stay readable until the terminal one.
    pub async fn close_write(&mut self) -> Result<()> {
        self.write.shutdown().await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RequestBody;
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    use tokio::net::UnixListener;

    /// Mock daemon: accepts one connection, parses one request, writes `responses`, then closes.
    /// The returned `TempDir` owns the socket and must outlive the test.
    fn mock_server(
        responses: &[Event],
    ) -> (tempfile::TempDir, PathBuf, tokio::task::JoinHandle<()>) {
        mock_server_raw(
            std::iter::once(hello_line())
                .chain(
                    responses
                        .iter()
                        .map(|event| serde_json::to_string(event).unwrap()),
                )
                .collect(),
        )
    }

    fn hello_line() -> String {
        serde_json::to_string(&Hello::current()).unwrap()
    }

    /// [`mock_server`] that writes each of `lines` verbatim.
    fn mock_server_raw(
        lines: Vec<String>,
    ) -> (tempfile::TempDir, PathBuf, tokio::task::JoinHandle<()>) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mock.sock");
        let listener = UnixListener::bind(&path).unwrap();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let (read, mut write) = stream.into_split();
            let mut reader = BufReader::new(read);
            let mut line = String::new();
            reader.read_line(&mut line).await.unwrap();
            let _: Request = serde_json::from_str(line.trim()).unwrap();
            for mut out in lines {
                out.push('\n');
                write.write_all(out.as_bytes()).await.unwrap();
            }
            write.flush().await.unwrap();
            write.shutdown().await.unwrap();
        });
        (dir, path, server)
    }

    #[tokio::test]
    async fn daemon_of_another_version_is_refused() {
        let connect = async |first_line: &str| {
            let (_dir, path, server) = mock_server_raw(vec![
                first_line.to_owned(),
                r#"{"type":"done","id":"r"}"#.into(),
            ]);
            let outcome = IpcClient::with_path(path)
                .one_shot(Request::query("r", "x"))
                .await;
            server.await.unwrap();
            outcome.map(|_| ())
        };
        let outcome = connect(r#"{"daemon_version":"0.0.1"}"#).await;
        assert!(
            matches!(
                &outcome,
                Err(IpcClientError::VersionMismatch { daemon: Some(version) }) if version == "0.0.1"
            ),
            "{outcome:?}"
        );
        let outcome = connect(r#"{"type":"delta","id":"r","text":"hello"}"#).await;
        assert!(
            matches!(
                outcome,
                Err(IpcClientError::VersionMismatch { daemon: None })
            ),
            "{outcome:?}"
        );
    }

    #[tokio::test]
    async fn undecodable_event_is_an_error() {
        let (_dir, path, server) = mock_server_raw(vec![
            hello_line(),
            r#"{"type":"confirm_request","id":"r","confirm_id":7}"#.into(),
            r#"{"type":"done","id":"r"}"#.into(),
        ]);
        let client = IpcClient::with_path(path);
        let stream = client.one_shot(Request::query("r", "x")).await.unwrap();
        let outcome = stream.collect().await;
        assert!(
            matches!(outcome, Err(IpcClientError::Json(_))),
            "{outcome:?}"
        );
        server.await.unwrap();
    }

    #[tokio::test]
    async fn collect_errors_on_premature_close() {
        let (_dir, path, server) = mock_server(&[Event::new(
            "r",
            EventBody::Delta {
                text: "incomplete".into(),
            },
        )]);
        let client = IpcClient::with_path(path);
        let stream = client.one_shot(Request::query("r", "x")).await.unwrap();
        let err = stream.collect().await.expect_err("expected DaemonClosed");
        assert!(matches!(err, IpcClientError::DaemonClosed), "{err:?}");
        server.await.unwrap();
    }

    #[tokio::test]
    async fn request_ends_on_done_error_or_early_close() {
        let delta = Event::new("r", EventBody::Delta { text: "hi".into() });
        let done = Event::new("r", EventBody::Done);
        let error = Event::new(
            "r",
            EventBody::Error {
                message: "boom".into(),
            },
        );
        let run = async |events: &[Event]| {
            let (_dir, path, server) = mock_server(events);
            let mut seen = 0;
            let outcome = IpcClient::with_path(path)
                .request(Request::query("r", "x"), |_| {
                    seen += 1;
                    Ok::<(), IpcClientError>(())
                })
                .await;
            server.await.unwrap();
            (seen, outcome)
        };
        assert!(matches!(run(&[delta.clone(), done]).await, (2, Ok(()))));
        let (seen, outcome) = run(&[delta.clone(), error]).await;
        assert!(
            matches!(&outcome, Err(IpcClientError::Daemon(message)) if message == "boom"),
            "{outcome:?}"
        );
        assert_eq!(seen, 2);
        let (_, outcome) = run(&[delta]).await;
        assert!(
            matches!(outcome, Err(IpcClientError::DaemonClosed)),
            "{outcome:?}"
        );
    }

    #[tokio::test]
    async fn dialog_can_send_after_initial_request() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dialog.sock");
        let listener = UnixListener::bind(&path).unwrap();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let (read, mut write) = stream.into_split();
            let mut reader = BufReader::new(read);

            let mut first = String::new();
            reader.read_line(&mut first).await.unwrap();
            let _: Request = serde_json::from_str(first.trim()).unwrap();

            let confirm = Event::new(
                "r",
                EventBody::ConfirmRequest {
                    confirm_id: "c1".into(),
                    tool: "bash".into(),
                    script: "rm -rf /tmp/x".into(),
                    matched_pattern: "rm -rf".into(),
                    always_allow: Vec::new(),
                    source: crate::ConfirmSource::Main,
                },
            );
            let mut out = hello_line() + "\n" + &serde_json::to_string(&confirm).unwrap();
            out.push('\n');
            write.write_all(out.as_bytes()).await.unwrap();
            write.flush().await.unwrap();

            let mut second = String::new();
            reader.read_line(&mut second).await.unwrap();
            let req: Request = serde_json::from_str(second.trim()).unwrap();
            assert!(matches!(
                req.body,
                RequestBody::ConfirmResponse { allow: true, .. }
            ));

            let mut done = serde_json::to_string(&Event::new("r", EventBody::Done)).unwrap();
            done.push('\n');
            write.write_all(done.as_bytes()).await.unwrap();
            write.flush().await.unwrap();
        });

        let client = IpcClient::with_path(&path);
        let mut conn = client.open_dialog(Request::query("r", "go")).await.unwrap();

        let event = conn.next_event().await.unwrap().expect("ConfirmRequest");
        match event.body {
            EventBody::ConfirmRequest { confirm_id, .. } => {
                conn.send(Request::new(
                    "r",
                    RequestBody::ConfirmResponse {
                        confirm_id,
                        allow: true,
                        always: false,
                    },
                ))
                .await
                .unwrap();
            }
            other => panic!("expected ConfirmRequest, got {other:?}"),
        }

        let done = conn.next_event().await.unwrap().expect("Done");
        assert_eq!(done, Event::new("r", EventBody::Done));
        server.await.unwrap();
    }
}
