//! Newline-delimited JSON-RPC over a child process's stdin/stdout;
//! stderr is forwarded to tracing.

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStderr, Command};
use tokio::sync::{mpsc, oneshot};
use tokio_util::task::AbortOnDropHandle;
use tracing::{debug, info, warn};

use crate::error::McpError;
use crate::jsonrpc::{Correlator, Response, notification_line};
use crate::{McpClient, ToolResult, ToolSchema, protocol};

/// The reader drops the connection rather than buffer a line past
/// this, so a misbehaving server cannot exhaust memory.
const MAX_LINE_BYTES: usize = 1024 * 1024;

/// Per-server stdio transport configuration.
#[derive(Debug, Clone)]
pub struct StdioConfig {
    pub command: String,
    pub args: Vec<String>,
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

/// [`McpClient`] over a child process's pipes.
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

        let pid = child.id();
        let stdout = child.stdout.take().expect("stdout piped");
        let stdin = child.stdin.take().expect("stdin piped");
        let stderr = child.stderr.take().expect("stderr piped");

        let stderr_task =
            AbortOnDropHandle::new(tokio::spawn(forward_stderr(stderr, cfg.label.clone())));

        let (client, transport_handles) =
            Self::from_streams(stdout, stdin, cfg.label.clone(), cfg.request_timeout).await?;

        if let Err(e) = client.initialize().await {
            warn!(
                target: "assistd::mcp",
                server = %cfg.label,
                error = %e,
                "MCP initialize failed; tearing down transport",
            );
            let _ = child.kill().await;
            transport_handles.shutdown_and_join().await;
            let _ = stderr_task.await;
            return Err(e);
        }

        info!(
            target: "assistd::mcp",
            server = %cfg.label,
            pid = pid,
            "MCP stdio server initialized",
        );

        let lifeline = ChildLifeline {
            label: cfg.label,
            child,
            transport: transport_handles,
            stderr_task,
        };
        Ok((client, lifeline))
    }

    /// Wire the transport over arbitrary streams without running the
    /// initialize handshake.
    pub async fn from_streams<R, W>(
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
            .call("tools/call", protocol::tool_call_params(name, arguments))
            .await?;
        protocol::parse_tool_call(result)
    }
}

/// The spawned child plus its I/O tasks. Dropping it SIGKILLs the
/// child and aborts the tasks.
pub struct ChildLifeline {
    label: String,
    child: Child,
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

    /// SIGTERM the process group, wait `term_timeout`, then SIGKILL.
    pub async fn shutdown(self, term_timeout: Duration) {
        let Self {
            label,
            mut child,
            transport,
            stderr_task,
        } = self;
        #[cfg(unix)]
        if let Some(pid) = child.id()
            && let Some(pgid) = rustix::process::Pid::from_raw(pid as i32)
        {
            let _ = rustix::process::kill_process_group(pgid, rustix::process::Signal::TERM);
        }
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
                let _ = child.start_kill();
                let _ = child.wait().await;
            }
        }

        transport.shutdown_and_join().await;
        let _ = tokio::time::timeout(Duration::from_millis(500), stderr_task).await;
    }
}

/// The read and write tasks of one transport. Dropping it aborts both.
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

async fn read_loop<R: AsyncRead + Unpin>(
    stream: R,
    correlator: Arc<Correlator>,
    label: String,
    done_tx: oneshot::Sender<()>,
) {
    let mut reader = BufReader::new(stream);
    let mut line = Vec::new();
    loop {
        line.clear();
        let read = (&mut reader)
            .take(MAX_LINE_BYTES as u64 + 1)
            .read_until(b'\n', &mut line)
            .await;
        let bytes_read = match read {
            Ok(bytes_read) => bytes_read,
            Err(e) => {
                warn!(
                    target: "assistd::mcp",
                    server = %label,
                    "MCP stdout read error: {e}",
                );
                break;
            }
        };
        if bytes_read == 0 {
            debug!(target: "assistd::mcp", server = %label, "MCP stdout EOF");
            break;
        }
        if bytes_read > MAX_LINE_BYTES {
            warn!(
                target: "assistd::mcp",
                server = %label,
                "MCP stdout line over {MAX_LINE_BYTES} bytes; dropping connection",
            );
            break;
        }
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        match serde_json::from_slice::<Response>(&line) {
            Ok(resp) => correlator.deliver(resp),
            Err(e) => {
                warn!(
                    target: "assistd::mcp",
                    server = %label,
                    "MCP stdout JSON parse error: {e}; line: {}",
                    String::from_utf8_lossy(&line).trim(),
                );
            }
        }
    }
    correlator.fail_all();
    let _ = done_tx.send(());
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

async fn forward_stderr(stream: ChildStderr, label: String) {
    let mut lines = BufReader::new(stream).lines();
    loop {
        match lines.next_line().await {
            Ok(Some(line)) => warn!(target: "assistd::mcp", server = %label, "{line}"),
            Ok(None) => return,
            Err(e) => {
                warn!(target: "assistd::mcp", server = %label, "stderr read error: {e}");
                return;
            }
        }
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

    async fn make_client_with_handler<F, Fut>(
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
        .await
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
        let (client, handles, server) = make_client_with_handler(handler).await;

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
        let (client, handles, server) = make_client_with_handler(handler).await;

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
        let (client, handles, server) = make_client_with_handler(handler).await;
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
        let (client, handles, server) = make_client_with_handler(handler).await;
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
        .await
        .unwrap();

        let err = client.list_tools().await.unwrap_err();
        assert!(
            matches!(err, McpError::RequestTimeout(timeout) if timeout == Duration::from_millis(150)),
            "{err:?}"
        );
        assert_eq!(client.correlator.in_flight(), 0);
        handles.shutdown_and_join().await;
    }

    #[tokio::test]
    async fn oversize_line_ends_the_read_loop_without_waiting_for_a_newline() {
        let (client_write, _server_read) = duplex(8192);
        let (mut server_write, client_read) = duplex(8192);
        let (_client, mut handles) = StdioMcpClient::from_streams(
            client_read,
            client_write,
            "flood".into(),
            Duration::from_secs(5),
        )
        .await
        .unwrap();

        let flood = tokio::spawn(async move {
            let chunk = vec![b'x'; 64 * 1024];
            while server_write.write_all(&chunk).await.is_ok() {}
        });

        tokio::time::timeout(Duration::from_secs(5), &mut handles.read_done)
            .await
            .expect("read loop must stop at the line cap, not buffer until a newline")
            .expect("read loop signals termination");

        flood.abort();
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
        .await
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
}
