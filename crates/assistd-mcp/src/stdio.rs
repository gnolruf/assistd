//! Newline-delimited JSON-RPC over a child process's stdin/stdout;
//! stderr is forwarded to tracing.

use std::collections::{BTreeSet, HashMap};
use std::ffi::OsString;
use std::fmt;
use std::io;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use rustix::process::Signal;
use serde_json::{Value, json};
use tokio::io::{
    AsyncBufRead, AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader,
};
use tokio::process::{Child, ChildStderr, Command};
use tokio::sync::{mpsc, oneshot};
use tokio_util::task::AbortOnDropHandle;
use tracing::{debug, info, warn};

use assistd_utils::log_lines::forward_lines;
use assistd_utils::process_group::ProcessGroup;

use crate::error::McpError;
use crate::jsonrpc::{Correlator, Incoming, notification_line, reply_line};
use crate::response_id::ResponseIdScanner;
use crate::{McpClient, ToolResult, ToolSchema, protocol};

/// Longest line the reader buffers. A longer line is discarded as it
/// streams in, failing only the call it answers.
const MAX_LINE_BYTES: usize = 32 * 1024 * 1024;

/// Line buffer capacity kept between lines, so one large reply does not
/// pin [`MAX_LINE_BYTES`] of memory for the life of the connection.
const RETAINED_LINE_CAPACITY: usize = 64 * 1024;

/// Daemon variables a server inherits besides `LC_*`; anything else,
/// credentials included, must come from [`StdioConfig::env`].
const INHERITED_ENV: &[&str] = &[
    "HOME", "LANG", "LANGUAGE", "LOGNAME", "PATH", "SHELL", "TERM", "TMPDIR", "TZ", "USER",
];

/// Per-server stdio transport configuration. `Debug` lists env var names,
/// never values.
#[derive(Clone)]
pub struct StdioConfig {
    pub command: String,
    pub args: Vec<String>,
    /// Set on top of the few daemon variables every server inherits.
    pub env: HashMap<String, String>,
    pub request_timeout: Duration,
    /// Server name used in tracing logs.
    pub label: String,
}

impl StdioConfig {
    /// Config that runs `command` with no args or extra env and a 30s
    /// request timeout.
    pub fn new(label: impl Into<String>, command: impl Into<String>) -> Self {
        Self {
            command: command.into(),
            args: Vec::new(),
            env: HashMap::new(),
            request_timeout: Duration::from_secs(30),
            label: label.into(),
        }
    }
}

impl fmt::Debug for StdioConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StdioConfig")
            .field("command", &self.command)
            .field("args", &self.args)
            .field("env_names", &self.env.keys().collect::<BTreeSet<_>>())
            .field("request_timeout", &self.request_timeout)
            .field("label", &self.label)
            .finish()
    }
}

/// [`McpClient`] over a child process's pipes.
#[derive(Debug)]
pub struct StdioMcpClient {
    label: String,
    correlator: Arc<Correlator>,
    write_tx: mpsc::Sender<Vec<u8>>,
    request_timeout: Duration,
}

impl StdioMcpClient {
    /// Spawn the server in its own process group and run the initialize
    /// handshake. Errors if either fails; a failed handshake kills the child.
    pub async fn spawn(cfg: StdioConfig) -> Result<(Arc<Self>, ChildLifeline), McpError> {
        let mut command = Command::new(&cfg.command);
        command
            .args(&cfg.args)
            .env_clear()
            .envs(inherited_env(std::env::vars_os()))
            .envs(cfg.env.iter())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        #[cfg(unix)]
        {
            command.process_group(0);
        }

        let mut child = command.spawn().map_err(|e| McpError::Spawn {
            path: cfg.command.clone(),
            source: e,
        })?;
        let group = ProcessGroup::led_by(&child).ok_or_else(|| McpError::Spawn {
            path: cfg.command.clone(),
            source: io::Error::other("spawned child reported no pid"),
        })?;

        let stdout = child.stdout.take().expect("stdout piped");
        let stdin = child.stdin.take().expect("stdin piped");
        let stderr = child.stderr.take().expect("stderr piped");

        let stderr_task =
            AbortOnDropHandle::new(tokio::spawn(forward_stderr(stderr, cfg.label.clone())));

        let (client, transport_handles) =
            Self::from_streams(stdout, stdin, cfg.label.clone(), cfg.request_timeout)?;
        let lifeline = ChildLifeline {
            label: cfg.label.clone(),
            child,
            group,
            transport: transport_handles,
            stderr_task,
        };

        if let Err(e) = client.initialize().await {
            warn!(
                target: "assistd::mcp",
                server = %cfg.label,
                error = %e,
                "MCP initialize failed; tearing down transport",
            );
            lifeline.kill().await;
            return Err(e);
        }

        info!(
            target: "assistd::mcp",
            server = %cfg.label,
            pid = lifeline.group.id().as_raw_nonzero(),
            "MCP stdio server initialized",
        );
        Ok((client, lifeline))
    }

    /// Wire the transport over arbitrary streams without running the
    /// initialize handshake.
    pub fn from_streams<R, W>(
        read: R,
        write: W,
        label: String,
        request_timeout: Duration,
    ) -> Result<(Arc<Self>, TransportHandles), McpError>
    where
        R: AsyncRead + Send + Unpin + 'static,
        W: AsyncWrite + Send + Unpin + 'static,
    {
        let correlator = Arc::new(Correlator::new());
        let (write_tx, write_rx) = mpsc::channel::<Vec<u8>>(128);

        let (read_done_tx, read_done_rx) = oneshot::channel::<()>();
        let read_task = AbortOnDropHandle::new(tokio::spawn(read_loop(
            read,
            correlator.clone(),
            write_tx.clone(),
            label.clone(),
            read_done_tx,
        )));
        let write_task =
            AbortOnDropHandle::new(tokio::spawn(write_loop(write, write_rx, label.clone())));

        let client = Arc::new(Self {
            label,
            correlator,
            write_tx,
            request_timeout,
        });

        Ok((
            client,
            TransportHandles {
                read_task,
                write_task,
                read_done: read_done_rx,
            },
        ))
    }

    /// Run the `initialize` handshake. No other request may be issued
    /// before this completes.
    pub async fn initialize(&self) -> Result<(), McpError> {
        let result = self
            .call("initialize", protocol::initialize_params())
            .await?;
        protocol::warn_on_version_mismatch(&self.label, &result);
        let bytes = notification_line("notifications/initialized", json!({}))?;
        self.write_tx
            .send(bytes)
            .await
            .map_err(|_| McpError::TransportClosed)?;
        Ok(())
    }

    async fn call(&self, method: &'static str, params: Value) -> Result<Value, McpError> {
        let mut pending = self.correlator.next_request(method, params)?;
        let bytes = pending.frame_line()?;
        self.write_tx
            .send(bytes)
            .await
            .map_err(|_| McpError::TransportClosed)?;

        protocol::await_reply(&mut pending.rx, self.request_timeout).await
    }
}

#[async_trait]
impl McpClient for StdioMcpClient {
    async fn list_tools(&self) -> Result<Vec<ToolSchema>, McpError> {
        let result = self.call("tools/list", json!({})).await?;
        protocol::parse_tools_list(&result)
    }

    async fn invoke(&self, name: &str, arguments: Value) -> Result<ToolResult, McpError> {
        let result = self
            .call("tools/call", protocol::tool_call_params(name, &arguments))
            .await?;
        protocol::parse_tool_call(&result)
    }
}

/// The spawned child plus its I/O tasks. Dropping it SIGKILLs the
/// child's whole process group and aborts the tasks.
#[derive(Debug)]
pub struct ChildLifeline {
    label: String,
    child: Child,
    group: ProcessGroup,
    transport: TransportHandles,
    stderr_task: AbortOnDropHandle<()>,
}

impl ChildLifeline {
    /// Resolves when the child exits or its stdout read loop ends; a
    /// child that can no longer answer counts as dead.
    pub async fn wait_for_death(&mut self) {
        tokio::select! {
            status = self.child.wait() => debug!(
                target: "assistd::mcp",
                server = %self.label,
                "MCP child exited: {status:?}",
            ),
            _ = &mut self.transport.read_done => debug!(
                target: "assistd::mcp",
                server = %self.label,
                "MCP read loop ended while the child was still alive",
            ),
        }
    }

    /// SIGTERM the process group, wait `term_timeout` for the child,
    /// then SIGKILL whatever is left of the group.
    pub async fn shutdown(self, term_timeout: Duration) {
        let Self {
            label,
            mut child,
            group,
            transport,
            stderr_task,
        } = self;
        group.signal(Signal::TERM);
        match tokio::time::timeout(term_timeout, child.wait()).await {
            Ok(Ok(status)) => {
                info!(
                    target: "assistd::mcp",
                    server = %label,
                    "MCP server exited after SIGTERM: {status}",
                );
            }
            Ok(Err(e)) => {
                warn!(
                    target: "assistd::mcp",
                    server = %label,
                    "MCP server wait error: {e}",
                );
            }
            Err(_) => {
                warn!(
                    target: "assistd::mcp",
                    server = %label,
                    "MCP server did not exit within {term_timeout:?}; sending SIGKILL",
                );
                group.signal(Signal::KILL);
                let _ = child.wait().await;
            }
        }
        drop(group);
        join_io_tasks(transport, stderr_task).await;
    }

    /// SIGKILL the process group and reap the child at once.
    async fn kill(self) {
        let Self {
            mut child,
            group,
            transport,
            stderr_task,
            ..
        } = self;
        drop(group);
        let _ = child.wait().await;
        join_io_tasks(transport, stderr_task).await;
    }
}

async fn join_io_tasks(transport: TransportHandles, stderr_task: AbortOnDropHandle<()>) {
    transport.shutdown_and_join().await;
    let _ = tokio::time::timeout(Duration::from_millis(500), stderr_task).await;
}

/// The read and write tasks of one transport. Dropping it aborts both.
#[derive(Debug)]
pub struct TransportHandles {
    read_task: AbortOnDropHandle<()>,
    write_task: AbortOnDropHandle<()>,
    /// Fires when the read loop terminates.
    pub read_done: oneshot::Receiver<()>,
}

impl TransportHandles {
    /// Abort both tasks and wait briefly for each to finish.
    pub async fn shutdown_and_join(self) {
        for task in [self.read_task, self.write_task] {
            task.abort();
            let _ = tokio::time::timeout(Duration::from_millis(500), task).await;
        }
    }
}

/// How one line read from the server's stdout ended.
enum LineRead {
    /// A line of at most [`MAX_LINE_BYTES`] is in the buffer.
    Complete,
    /// A longer line was discarded; `response_id` names the call it answered.
    Oversize {
        response_id: Option<u64>,
    },
    Eof,
}

async fn read_loop<R: AsyncRead + Unpin>(
    stream: R,
    correlator: Arc<Correlator>,
    write_tx: mpsc::Sender<Vec<u8>>,
    label: String,
    done_tx: oneshot::Sender<()>,
) {
    let mut reader = BufReader::new(stream);
    let mut line = Vec::new();
    loop {
        line.clear();
        line.shrink_to(RETAINED_LINE_CAPACITY);
        match read_line(&mut reader, &mut line).await {
            Ok(LineRead::Complete) => dispatch_line(&line, &correlator, &write_tx, &label).await,
            Ok(LineRead::Oversize { response_id }) => {
                reject_oversize_reply(&correlator, &label, response_id);
            }
            Ok(LineRead::Eof) => {
                debug!(target: "assistd::mcp", server = %label, "MCP stdout EOF");
                break;
            }
            Err(e) => {
                warn!(
                    target: "assistd::mcp",
                    server = %label,
                    "MCP stdout read error: {e}",
                );
                break;
            }
        }
    }
    correlator.fail_all();
    let _ = done_tx.send(());
}

async fn read_line<R: AsyncBufRead + Unpin>(
    reader: &mut R,
    line: &mut Vec<u8>,
) -> io::Result<LineRead> {
    let bytes_read = (&mut *reader)
        .take(MAX_LINE_BYTES as u64 + 1)
        .read_until(b'\n', line)
        .await?;
    if bytes_read == 0 {
        return Ok(LineRead::Eof);
    }
    if bytes_read <= MAX_LINE_BYTES {
        return Ok(LineRead::Complete);
    }
    let mut scanner = ResponseIdScanner::default();
    scanner.feed(line);
    if line.last() != Some(&b'\n') {
        skip_rest_of_line(reader, &mut scanner).await?;
    }
    Ok(LineRead::Oversize {
        response_id: scanner.response_id(),
    })
}

async fn skip_rest_of_line<R: AsyncBufRead + Unpin>(
    reader: &mut R,
    scanner: &mut ResponseIdScanner,
) -> io::Result<()> {
    loop {
        let buffered = reader.fill_buf().await?;
        if buffered.is_empty() {
            return Ok(());
        }
        let newline = buffered.iter().position(|&byte| byte == b'\n');
        let consumed = newline.map_or(buffered.len(), |at| at + 1);
        scanner.feed(&buffered[..consumed]);
        reader.consume(consumed);
        if newline.is_some() {
            return Ok(());
        }
    }
}

async fn dispatch_line(
    line: &[u8],
    correlator: &Correlator,
    write_tx: &mpsc::Sender<Vec<u8>>,
    label: &str,
) {
    if line.iter().all(u8::is_ascii_whitespace) {
        return;
    }
    match Incoming::parse(line) {
        Ok(Incoming::Response(response)) => correlator.deliver(response),
        Ok(Incoming::Request { id, method }) => {
            answer_server_request(write_tx, label, id, &method).await;
        }
        Ok(Incoming::Notification { method }) => {
            debug!(target: "assistd::mcp", server = %label, method, "ignoring server notification");
        }
        Err(e) => {
            warn!(
                target: "assistd::mcp",
                server = %label,
                "MCP stdout JSON parse error: {e}; line: {}",
                String::from_utf8_lossy(line).trim(),
            );
        }
    }
}

fn reject_oversize_reply(correlator: &Correlator, label: &str, response_id: Option<u64>) {
    warn!(
        target: "assistd::mcp",
        server = %label,
        ?response_id,
        "MCP stdout line over {MAX_LINE_BYTES} bytes; discarded",
    );
    if let Some(id) = response_id {
        correlator.fail(
            id,
            McpError::ReplyTooLarge {
                limit: MAX_LINE_BYTES,
            },
        );
    }
}

async fn answer_server_request(
    write_tx: &mpsc::Sender<Vec<u8>>,
    label: &str,
    id: Value,
    method: &str,
) {
    debug!(target: "assistd::mcp", server = %label, method, "answering server request");
    let line = match reply_line(&id, &protocol::answer_server_request(method)) {
        Ok(line) => line,
        Err(e) => {
            warn!(target: "assistd::mcp", server = %label, "failed to encode reply to `{method}`: {e}");
            return;
        }
    };
    if write_tx.send(line).await.is_err() {
        debug!(target: "assistd::mcp", server = %label, "write loop gone; reply to `{method}` dropped");
    }
}

async fn write_loop<W: AsyncWrite + Unpin>(
    mut stream: W,
    mut write_rx: mpsc::Receiver<Vec<u8>>,
    label: String,
) {
    while let Some(bytes) = write_rx.recv().await {
        if let Err(e) = stream.write_all(&bytes).await {
            warn!(
                target: "assistd::mcp",
                server = %label,
                "MCP stdin write error: {e}",
            );
            break;
        }
        if let Err(e) = stream.flush().await {
            warn!(
                target: "assistd::mcp",
                server = %label,
                "MCP stdin flush error: {e}",
            );
            break;
        }
    }
}

fn inherited_env(
    vars: impl IntoIterator<Item = (OsString, OsString)>,
) -> impl Iterator<Item = (OsString, OsString)> {
    let inherited = |name: &str| INHERITED_ENV.contains(&name) || name.starts_with("LC_");
    vars.into_iter()
        .filter(move |(name, _)| name.to_str().is_some_and(inherited))
}

async fn forward_stderr(stream: ChildStderr, label: String) {
    let forwarded = forward_lines(
        stream,
        |line| warn!(target: "assistd::mcp", server = %label, "{line}"),
    )
    .await;
    if let Err(e) = forwarded {
        warn!(target: "assistd::mcp", server = %label, "stderr read error: {e}");
    }
}

#[cfg(test)]
mod tests {
    use tokio::io::{DuplexStream, duplex};
    use tokio::task::JoinHandle;

    use super::*;

    /// Pretend MCP server: answers every request read from
    /// `client_to_server` with `handler(request)` on `server_to_client`.
    fn fake_server<F, Fut>(
        client_to_server: DuplexStream,
        server_to_client: DuplexStream,
        handler: F,
    ) -> JoinHandle<()>
    where
        F: Fn(Value) -> Fut + Send + 'static,
        Fut: Future<Output = Value> + Send,
    {
        tokio::spawn(async move {
            let mut reader = BufReader::new(client_to_server);
            let mut writer = server_to_client;
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line).await {
                    Ok(0) | Err(_) => return,
                    Ok(_) => {}
                }
                let Ok(req) = serde_json::from_str::<Value>(line.trim()) else {
                    continue;
                };
                if req.get("id").is_none() {
                    continue;
                }
                let resp = handler(req).await;
                let mut bytes = serde_json::to_vec(&resp).unwrap();
                bytes.push(b'\n');
                if writer.write_all(&bytes).await.is_err() {
                    return;
                }
                if writer.flush().await.is_err() {
                    return;
                }
            }
        })
    }

    fn make_client_with_handler<F, Fut>(
        handler: F,
    ) -> (Arc<StdioMcpClient>, TransportHandles, JoinHandle<()>)
    where
        F: Fn(Value) -> Fut + Send + Clone + 'static,
        Fut: Future<Output = Value> + Send,
    {
        let (client_write, server_read) = duplex(8192);
        let (server_write, client_read) = duplex(8192);

        let server_task = fake_server(server_read, server_write, handler);

        let (client, handles) = StdioMcpClient::from_streams(
            client_read,
            client_write,
            "test".into(),
            Duration::from_secs(2),
        )
        .unwrap();
        (client, handles, server_task)
    }

    #[tokio::test]
    async fn list_tools_round_trip() {
        let handler = |req: Value| async move {
            assert_eq!(req["method"], "tools/list");
            json!({
                "jsonrpc": "2.0",
                "id": req["id"],
                "result": {
                    "tools": [
                        {
                            "name": "echo",
                            "description": "echoes its input",
                            "inputSchema": {"type": "object", "properties": {"x": {"type": "string"}}}
                        }
                    ]
                }
            })
        };
        let (client, handles, server) = make_client_with_handler(handler);

        let tools = client.list_tools().await.unwrap();
        let [tool] = tools.as_slice() else {
            panic!("expected one tool, got {tools:?}");
        };
        assert_eq!(tool.name, "echo");
        assert_eq!(tool.description, "echoes its input");
        assert_eq!(
            tool.input_schema,
            json!({"type": "object", "properties": {"x": {"type": "string"}}})
        );

        handles.shutdown_and_join().await;
        let _ = server.await;
    }

    #[tokio::test]
    async fn invoke_text_response() {
        let handler = |req: Value| async move {
            assert_eq!(req["method"], "tools/call");
            assert_eq!(
                req["params"],
                json!({"name": "echo", "arguments": {"x": "hi"}})
            );
            json!({
                "jsonrpc": "2.0",
                "id": req["id"],
                "result": {
                    "content": [{"type": "text", "text": "hello"}],
                    "isError": false
                }
            })
        };
        let (client, handles, server) = make_client_with_handler(handler);

        let result = client.invoke("echo", json!({"x": "hi"})).await.unwrap();
        match result {
            ToolResult::Text(text) => assert_eq!(text, "hello"),
            other => panic!("expected Text, got {other:?}"),
        }
        handles.shutdown_and_join().await;
        let _ = server.await;
    }

    #[tokio::test]
    async fn invoke_image_response_decodes_base64() {
        let handler = |req: Value| async move {
            json!({
                "jsonrpc": "2.0",
                "id": req["id"],
                "result": {
                    "content": [{
                        "type": "image",
                        "mimeType": "image/png",
                        "data": "3q2+7w=="
                    }],
                    "isError": false
                }
            })
        };
        let (client, handles, server) = make_client_with_handler(handler);
        let result = client.invoke("snap", json!({})).await.unwrap();
        match result {
            ToolResult::Image { mime, bytes } => {
                assert_eq!(mime, "image/png");
                assert_eq!(bytes, [0xDE, 0xAD, 0xBE, 0xEF]);
            }
            other => panic!("expected Image, got {other:?}"),
        }
        handles.shutdown_and_join().await;
        let _ = server.await;
    }

    #[tokio::test]
    async fn rpc_error_surfaces_as_error() {
        let handler = |req: Value| async move {
            json!({
                "jsonrpc": "2.0",
                "id": req["id"],
                "error": {"code": -32601, "message": "method not found"}
            })
        };
        let (client, handles, server) = make_client_with_handler(handler);
        let err = client.list_tools().await.unwrap_err();
        assert!(
            matches!(
                &err,
                McpError::RpcError { code: -32601, message, .. } if message == "method not found"
            ),
            "{err:?}"
        );
        handles.shutdown_and_join().await;
        let _ = server.await;
    }

    #[tokio::test]
    async fn request_timeout_fires_when_server_never_answers() {
        let (client_write, _server_read) = duplex(8192);
        let (_server_write, client_read) = duplex(8192);

        let (client, handles) = StdioMcpClient::from_streams(
            client_read,
            client_write,
            "silent".into(),
            Duration::from_millis(150),
        )
        .unwrap();

        let err = client.list_tools().await.unwrap_err();
        assert!(
            matches!(err, McpError::RequestTimeout(timeout) if timeout == Duration::from_millis(150)),
            "{err:?}"
        );
        assert_eq!(client.correlator.in_flight(), 0);
        handles.shutdown_and_join().await;
    }

    fn spawn_list_tools(
        client: &Arc<StdioMcpClient>,
    ) -> JoinHandle<Result<Vec<ToolSchema>, McpError>> {
        let client = client.clone();
        tokio::spawn(async move { client.list_tools().await })
    }

    async fn next_request_id(from_client: &mut BufReader<DuplexStream>) -> Value {
        let mut request = String::new();
        from_client.read_line(&mut request).await.unwrap();
        serde_json::from_str::<Value>(&request).unwrap()["id"].clone()
    }

    fn empty_tools_reply(id: &Value) -> String {
        format!(
            "{}\n",
            json!({"jsonrpc": "2.0", "id": id, "result": {"tools": []}})
        )
    }

    #[tokio::test]
    async fn oversize_reply_fails_only_its_own_call() {
        let (client_write, server_read) = duplex(64 * 1024);
        let (mut server_write, client_read) = duplex(64 * 1024);
        let (client, mut handles) = StdioMcpClient::from_streams(
            client_read,
            client_write,
            "big".into(),
            Duration::from_secs(5),
        )
        .unwrap();
        let mut from_client = BufReader::new(server_read);

        let oversized = spawn_list_tools(&client);
        let oversized_id = next_request_id(&mut from_client).await;
        let normal = spawn_list_tools(&client);
        let normal_id = next_request_id(&mut from_client).await;

        let padding = "x".repeat(MAX_LINE_BYTES);
        let reply = format!(
            "{{\"jsonrpc\":\"2.0\",\"result\":{{\"tools\":[],\"padding\":\"{padding}\"}},\"id\":{oversized_id}}}\n"
        );
        server_write.write_all(reply.as_bytes()).await.unwrap();
        server_write
            .write_all(empty_tools_reply(&normal_id).as_bytes())
            .await
            .unwrap();

        let err = oversized.await.unwrap().unwrap_err();
        assert!(
            matches!(
                err,
                McpError::ReplyTooLarge {
                    limit: MAX_LINE_BYTES
                }
            ),
            "{err:?}"
        );
        assert!(normal.await.unwrap().unwrap().is_empty());
        assert!(
            matches!(
                handles.read_done.try_recv(),
                Err(oneshot::error::TryRecvError::Empty)
            ),
            "an oversize reply must not end the read loop"
        );
        handles.shutdown_and_join().await;
    }

    #[tokio::test]
    async fn oversize_line_ending_at_the_read_limit_leaves_the_next_line_intact() {
        let (client_write, server_read) = duplex(64 * 1024);
        let (mut server_write, client_read) = duplex(64 * 1024);
        let (client, handles) = StdioMcpClient::from_streams(
            client_read,
            client_write,
            "edge".into(),
            Duration::from_secs(5),
        )
        .unwrap();
        let mut from_client = BufReader::new(server_read);

        let call = spawn_list_tools(&client);
        let id = next_request_id(&mut from_client).await;

        let mut junk = vec![b'x'; MAX_LINE_BYTES];
        junk.push(b'\n');
        server_write.write_all(&junk).await.unwrap();
        server_write
            .write_all(empty_tools_reply(&id).as_bytes())
            .await
            .unwrap();

        assert!(call.await.unwrap().unwrap().is_empty());
        handles.shutdown_and_join().await;
    }

    #[tokio::test]
    async fn server_ping_is_answered_and_leaves_the_pending_call_waiting() {
        let (client_write, server_read) = duplex(8192);
        let (mut server_write, client_read) = duplex(8192);
        let (client, handles) = StdioMcpClient::from_streams(
            client_read,
            client_write,
            "pinger".into(),
            Duration::from_secs(5),
        )
        .unwrap();

        let call = tokio::spawn({
            let client = client.clone();
            async move { client.list_tools().await }
        });

        let mut from_client = BufReader::new(server_read);
        let mut request = String::new();
        from_client.read_line(&mut request).await.unwrap();
        let request: Value = serde_json::from_str(&request).unwrap();
        assert_eq!(request["method"], "tools/list");
        let id = request["id"].clone();

        let ping = format!(
            "{}\n",
            json!({"jsonrpc": "2.0", "id": id, "method": "ping"})
        );
        server_write.write_all(ping.as_bytes()).await.unwrap();

        let mut reply = String::new();
        from_client.read_line(&mut reply).await.unwrap();
        let reply: Value = serde_json::from_str(&reply).unwrap();
        assert_eq!(reply, json!({"jsonrpc": "2.0", "id": id, "result": {}}));
        assert!(
            !call.is_finished(),
            "a server request must not complete our call"
        );

        let response = format!(
            "{}\n",
            json!({"jsonrpc": "2.0", "id": id, "result": {"tools": []}})
        );
        server_write.write_all(response.as_bytes()).await.unwrap();
        let tools = call.await.unwrap().unwrap();
        assert!(tools.is_empty());
        handles.shutdown_and_join().await;
    }

    #[tokio::test]
    async fn transport_close_wakes_pending_calls() {
        let (client_write, server_read) = duplex(8192);
        let (server_write, client_read) = duplex(8192);
        let (client, handles) = StdioMcpClient::from_streams(
            client_read,
            client_write,
            "drop".into(),
            Duration::from_secs(5),
        )
        .unwrap();

        let call = tokio::spawn({
            let client = client.clone();
            async move { client.list_tools().await }
        });
        // Register the request before EOF so `fail_all` sees it.
        while client.correlator.in_flight() == 0 {
            tokio::task::yield_now().await;
        }
        drop(server_read);
        drop(server_write);

        let err = call.await.unwrap().unwrap_err();
        assert!(matches!(err, McpError::TransportClosed), "{err}");
        handles.shutdown_and_join().await;
    }

    #[test]
    fn debug_lists_env_names_but_not_values() {
        let mut cfg = StdioConfig::new("local", "server");
        cfg.env.insert("API_TOKEN".into(), "hunter2".into());
        let rendered = format!("{cfg:?}");
        assert!(rendered.contains("API_TOKEN"), "{rendered}");
        assert!(!rendered.contains("hunter2"), "{rendered}");
    }

    #[test]
    fn servers_inherit_only_basic_session_and_locale_variables() {
        let vars = [
            "PATH",
            "HOME",
            "LC_ALL",
            "TZ",
            "GITHUB_TOKEN",
            "AWS_SECRET_ACCESS_KEY",
            "SSH_AUTH_SOCK",
        ]
        .map(|name| (OsString::from(name), OsString::from("value")));
        let names: Vec<_> = inherited_env(vars).map(|(name, _)| name).collect();
        assert_eq!(names, ["PATH", "HOME", "LC_ALL", "TZ"]);
    }
}
