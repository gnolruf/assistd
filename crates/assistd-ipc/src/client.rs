//! Typed Unix-socket client wrapper for the assistd IPC protocol.
//!
//! Eight CLI subcommands and the chat TUI all open the same Unix socket,
//! send a [`Request`] line, and pump [`Event`] lines back. This module
//! collapses that plumbing into one type so callers don't reinvent the
//! framing each time.
//!
//! Two connection shapes are supported:
//!
//! - [`IpcClient::one_shot`]: legacy "send one Request, shutdown write
//!   half, read until terminal Event". The vast majority of CLI calls
//!   take this path.
//! - [`IpcClient::open_dialog`]: bidirectional. The connection stays
//!   write-open after the initial Request so the client can answer
//!   mid-stream prompts (e.g. [`Request::ConfirmResponse`]) on the same
//!   socket. Used by the chat TUI's Query and PttStop flows.
//!
//! Available only with the `client` cargo feature.

use std::path::{Path, PathBuf};

use thiserror::Error;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::net::unix::{OwnedReadHalf, OwnedWriteHalf};

use crate::{Event, Request, socket_path};

/// Errors produced by the IPC client.
#[derive(Debug, Error)]
pub enum IpcClientError {
    /// The daemon socket couldn't be reached. Most likely "daemon not
    /// running"; callers can use this signal to drive an auto-spawn
    /// path.
    #[error("daemon not reachable at {path}: {source}")]
    NotReachable {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// Generic I/O failure on a previously-opened connection.
    #[error("ipc i/o error: {0}")]
    Io(#[from] std::io::Error),
    /// JSON serialization or deserialization error.
    #[error("ipc json error: {0}")]
    Json(#[from] serde_json::Error),
    /// Daemon closed the connection before emitting `Done` or `Error`.
    #[error("daemon closed connection mid-stream")]
    DaemonClosed,
}

/// Convenience alias for `Result<T, `[`IpcClientError`]`>`.
pub type Result<T> = std::result::Result<T, IpcClientError>;

/// Connection factory. Construct once, reuse across calls; `Clone` is
/// cheap (single `PathBuf`).
#[derive(Debug, Clone)]
pub struct IpcClient {
    socket_path: PathBuf,
}

impl IpcClient {
    /// Use the default socket path (`$XDG_RUNTIME_DIR/assistd.sock` or
    /// `/tmp/assistd-$USER.sock`).
    pub fn new() -> Self {
        Self {
            socket_path: socket_path(),
        }
    }

    /// Use a specific socket path. Primarily for tests.
    pub fn with_path(p: impl Into<PathBuf>) -> Self {
        Self {
            socket_path: p.into(),
        }
    }

    /// The configured socket path. Useful for diagnostics and for
    /// callers that want to probe `path.exists()` before connecting.
    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    /// Open the socket, send a single Request, shut down the write
    /// half, and return an [`EventStream`] that yields events until
    /// the daemon emits `Done` or `Error` (or closes mid-stream).
    pub async fn one_shot(&self, req: Request) -> Result<EventStream> {
        let stream = self.connect().await?;
        let (read, mut write) = stream.into_split();
        let mut payload = serde_json::to_string(&req)?;
        payload.push('\n');
        write.write_all(payload.as_bytes()).await?;
        write.flush().await?;
        write.shutdown().await?;
        Ok(EventStream::new(read))
    }

    /// Open the socket and send the initial Request, but keep the
    /// write half open. The returned [`DialogConnection`] can read
    /// streamed events and write additional Requests (e.g. a
    /// [`Request::ConfirmResponse`] in reply to a daemon-issued
    /// [`Event::ConfirmRequest`]) on the same connection.
    pub async fn open_dialog(&self, initial: Request) -> Result<DialogConnection> {
        let stream = self.connect().await?;
        let (read, mut write) = stream.into_split();
        let mut payload = serde_json::to_string(&initial)?;
        payload.push('\n');
        write.write_all(payload.as_bytes()).await?;
        write.flush().await?;
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

/// Stream of [`Event`]s read from a daemon connection.
///
/// `next_event` returns `Ok(Some(_))` for each event, `Ok(None)` when
/// the daemon closes the connection cleanly, and `Err(_)` for I/O or
/// JSON failures. Most callers want to loop until they see a terminal
/// event ([`Event::is_terminal`]).
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

    /// Read the next event. `Ok(None)` means the daemon closed the
    /// stream without an explicit terminal event; callers should
    /// usually map this to [`IpcClientError::DaemonClosed`] when they
    /// haven't already seen `Done`/`Error`.
    pub async fn next_event(&mut self) -> Result<Option<Event>> {
        match self.inner.next_line().await? {
            None => Ok(None),
            Some(line) => Ok(Some(serde_json::from_str(&line)?)),
        }
    }

    /// Drain the stream into a `Vec`. Reads until either a terminal
    /// event or `Ok(None)`. Convenient for one-shot CLI calls that
    /// don't need to react to events as they arrive.
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

/// Bidirectional connection: read events as they arrive, send
/// additional Requests at any time. Used by the chat TUI to answer
/// mid-stream confirmation prompts on the same connection that's
/// streaming the LLM response.
pub struct DialogConnection {
    write: OwnedWriteHalf,
    events: EventStream,
}

impl DialogConnection {
    /// Receive the next event from the daemon. Same semantics as
    /// [`EventStream::next_event`].
    pub async fn next_event(&mut self) -> Result<Option<Event>> {
        self.events.next_event().await
    }

    /// Send a Request on the open connection. The most common case is
    /// a [`Request::ConfirmResponse`] in reply to an inbound
    /// [`Event::ConfirmRequest`]; the daemon also accepts arbitrary
    /// additional Requests but most flows don't use that.
    pub async fn send(&mut self, req: Request) -> Result<()> {
        let mut payload = serde_json::to_string(&req)?;
        payload.push('\n');
        self.write.write_all(payload.as_bytes()).await?;
        self.write.flush().await?;
        Ok(())
    }

    /// Close the write half, signalling end-of-input to the daemon.
    /// The event stream remains readable until the daemon emits its
    /// terminal event (or closes its end).
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

    /// Spin up a tiny mock daemon that reads one line, validates it
    /// parses as a Request, then replies with the events from
    /// `responses` (one per line) and shuts down the write half so
    /// the client sees a clean EOF. Returns the socket path and a
    /// JoinHandle the test should await once it's done.
    async fn mock_server(responses: Vec<Event>) -> (PathBuf, tokio::task::JoinHandle<()>) {
        let dir = tempfile::tempdir().unwrap();
        // Leak the tempdir so the socket file survives until process
        // exit; tests don't bother to clean up explicitly and this
        // avoids the dir being dropped before the connection completes.
        let path = dir.path().join("mock.sock");
        std::mem::forget(dir);

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
            // Half-close so the client's read returns Ok(None) instead
            // of "connection reset by peer" on the next poll.
            write.shutdown().await.unwrap();
            drop(listener);
        });

        // Yield once so the spawned task gets a chance to bind+accept.
        // The path already exists from `bind`; an explicit connect
        // would consume the only `accept()` on this listener.
        tokio::task::yield_now().await;
        (path, h)
    }

    #[tokio::test]
    async fn one_shot_collects_events_until_done() {
        let (path, _h) = mock_server(vec![
            Event::Delta {
                id: "r".into(),
                text: "hello".into(),
            },
            Event::Done { id: "r".into() },
        ])
        .await;

        let client = IpcClient::with_path(path);
        let stream = client
            .one_shot(Request::query("r", "hi"))
            .await
            .expect("one_shot");
        let events = stream.collect().await.expect("collect");
        assert_eq!(events.len(), 2);
        assert!(matches!(events[0], Event::Delta { .. }));
        assert!(matches!(events[1], Event::Done { .. }));
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
        assert!(matches!(err, IpcClientError::NotReachable { .. }));
    }

    #[tokio::test]
    async fn collect_errors_on_premature_close() {
        // Server sends one delta and closes; no terminal event.
        let (path, _h) = mock_server(vec![Event::Delta {
            id: "r".into(),
            text: "incomplete".into(),
        }])
        .await;
        let client = IpcClient::with_path(path);
        let stream = client.one_shot(Request::query("r", "x")).await.unwrap();
        let err = stream.collect().await.expect_err("expected DaemonClosed");
        assert!(matches!(err, IpcClientError::DaemonClosed));
    }

    #[tokio::test]
    async fn dialog_can_send_after_initial_request() {
        // Server: read initial request, send a ConfirmRequest event,
        // then read a ConfirmResponse and echo Done.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dialog.sock");
        let listener = UnixListener::bind(&path).unwrap();
        let h = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let (read, mut write) = stream.into_split();
            let mut reader = BufReader::new(read);
            // Read initial Request::Query.
            let mut first = String::new();
            reader.read_line(&mut first).await.unwrap();
            let _: Request = serde_json::from_str(first.trim()).unwrap();

            // Emit a ConfirmRequest event.
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

            // Read the client's ConfirmResponse.
            let mut second = String::new();
            reader.read_line(&mut second).await.unwrap();
            let req: Request = serde_json::from_str(second.trim()).unwrap();
            assert!(matches!(req, Request::ConfirmResponse { allow: true, .. }));

            // Emit Done.
            let mut done = serde_json::to_string(&Event::Done { id: "r".into() }).unwrap();
            done.push('\n');
            write.write_all(done.as_bytes()).await.unwrap();
            write.flush().await.unwrap();
        });

        // Yield so the spawned listener task starts before we connect.
        tokio::task::yield_now().await;

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
        assert!(matches!(done, Event::Done { .. }));
        h.await.unwrap();
    }
}
