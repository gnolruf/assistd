//! Stdio JSON-RPC transport for MCP servers spawned as a child process.
//!
//! Wire format: newline-delimited JSON-RPC 2.0 over the child's stdin
//! (outbound) and stdout (inbound). Stderr is forwarded to tracing.
//!
//! The transport core is split from process spawning so tests can run
//! the full request/response loop against a `tokio::io::duplex` pair —
//! see [`StdioMcpClient::from_streams`].

#![allow(unsafe_code)] // libc::kill for SIGTERM-then-SIGKILL on shutdown.

use std::collections::HashMap;
use std::process::{ExitStatus, Stdio};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result as AnyResult;
use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

use crate::error::McpError;
use crate::jsonrpc::{Correlator, Response, RpcError, notification_line};
use crate::{McpClient, ToolResult, ToolSchema};

const PROTOCOL_VERSION: &str = "2024-11-05";
const CLIENT_NAME: &str = "assistd";
const CLIENT_VERSION: &str = env!("CARGO_PKG_VERSION");
/// MCP servers that emit a single line larger than this are protocol-violating.
/// Capping prevents an upstream bug or a hostile server from leaking memory.
const MAX_LINE_BYTES: usize = 1024 * 1024;

/// Per-server stdio transport configuration.
#[derive(Debug, Clone)]
pub struct StdioConfig {
    pub command: String,
    pub args: Vec<String>,
    pub env: HashMap<String, String>,
    pub request_timeout: Duration,
    /// Label used in tracing logs (typically the server's config name).
    pub label: String,
}

impl StdioConfig {
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

/// `McpClient` impl that owns the writer-task channel and shares the
/// `Correlator` with the transport's reader task.
pub struct StdioMcpClient {
    label: String,
    correlator: Arc<Correlator>,
    write_tx: mpsc::Sender<Vec<u8>>,
    request_timeout: Duration,
}

impl StdioMcpClient {
    /// Spawn an MCP server child process and bring up the transport.
    /// Returns the client (an `Arc<dyn McpClient>` once cast) and a
    /// `ChildLifeline` whose `wait_for_exit()` future fires when the
    /// child exits — used by the per-server supervisor.
    pub async fn spawn(cfg: StdioConfig) -> Result<(Arc<Self>, ChildLifeline), McpError> {
        let mut cmd = Command::new(&cfg.command);
        cmd.args(&cfg.args)
            .envs(cfg.env.iter())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        #[cfg(unix)]
        {
            // Process group so SIGTERM to -pid takes down any grandchildren
            // (e.g., npx → node → server-filesystem).
            cmd.process_group(0);
        }

        let mut child = cmd.spawn().map_err(|e| McpError::Spawn {
            path: cfg.command.clone(),
            source: e,
        })?;

        let pid = child.id();
        let stdout = child.stdout.take().expect("stdout piped");
        let stdin = child.stdin.take().expect("stdin piped");
        let stderr = child.stderr.take().expect("stderr piped");

        let stderr_task = {
            let label = cfg.label.clone();
            tokio::spawn(forward_stderr(stderr, label))
        };

        let (client, transport_handles) =
            Self::from_streams(stdout, stdin, cfg.label.clone(), cfg.request_timeout).await?;

        // Initialize handshake. Failure here brings down the whole transport
        // since a server that won't agree on the protocol version cannot
        // serve tools/list / tools/call.
        if let Err(e) = client.initialize().await {
            warn!(
                target: "assistd::mcp",
                server = %cfg.label,
                error = %e,
                "MCP initialize failed; tearing down transport",
            );
            // Kill child, await background tasks, propagate error.
            let _ = child.kill().await;
            let _ = transport_handles.shutdown_and_join().await;
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
            label: cfg.label.clone(),
            child: Some(child),
            transport: Some(transport_handles),
            stderr_task: Some(stderr_task),
        };
        Ok((client, lifeline))
    }

    /// Wire the transport up over arbitrary `AsyncRead` / `AsyncWrite`.
    /// Used by [`Self::spawn`] (with the child's stdout/stdin) and by
    /// tests (with `tokio::io::duplex`). Does NOT perform the initialize
    /// handshake — call [`Self::initialize`] separately.
    pub async fn from_streams<R, W>(
        read: R,
        write: W,
        label: String,
        request_timeout: Duration,
    ) -> Result<(Arc<Self>, TransportHandles), McpError>
    where
        R: AsyncRead + Send + Unpin + 'static,
        W: tokio::io::AsyncWrite + Send + Unpin + 'static,
    {
        let correlator = Arc::new(Correlator::new());
        let (write_tx, write_rx) = mpsc::channel::<Vec<u8>>(128);

        let read_label = label.clone();
        let read_correlator = correlator.clone();
        let (read_done_tx, read_done_rx) = oneshot::channel::<()>();
        let read_task = tokio::spawn(read_loop(read, read_correlator, read_label, read_done_tx));

        let write_label = label.clone();
        let write_task = tokio::spawn(write_loop(write, write_rx, write_label));

        let client = Arc::new(Self {
            label,
            correlator,
            write_tx,
            request_timeout,
        });

        Ok((
            client,
            TransportHandles {
                read_task: Some(read_task),
                write_task: Some(write_task),
                read_done: Some(read_done_rx),
            },
        ))
    }

    /// Send the `initialize` request and the follow-up
    /// `notifications/initialized` notification. Per the MCP spec, no
    /// other request may be issued before this completes.
    pub async fn initialize(&self) -> Result<(), McpError> {
        let params = json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": { "tools": {} },
            "clientInfo": { "name": CLIENT_NAME, "version": CLIENT_VERSION },
        });
        let result = self.call("initialize", params).await?;

        // Some servers return their negotiated protocolVersion; we don't
        // hard-fail on mismatch because servers commonly advertise multiple
        // versions and tools/call still works. We log it for the operator.
        if let Some(server_pv) = result.get("protocolVersion").and_then(Value::as_str)
            && server_pv != PROTOCOL_VERSION
        {
            warn!(
                target: "assistd::mcp",
                server = %self.label,
                client_version = PROTOCOL_VERSION,
                server_version = server_pv,
                "MCP protocol version mismatch (continuing optimistically)",
            );
        }

        let bytes = notification_line("notifications/initialized", json!({}))?;
        self.write_tx
            .send(bytes)
            .await
            .map_err(|_| McpError::TransportClosed)?;
        Ok(())
    }

    /// Issue a JSON-RPC request; await response with [`Self::request_timeout`].
    async fn call(&self, method: &'static str, params: Value) -> Result<Value, McpError> {
        let pending = self.correlator.next_request(method, params)?;
        let bytes = pending.frame_line()?;
        self.write_tx
            .send(bytes)
            .await
            .map_err(|_| McpError::TransportClosed)?;

        let reply = match tokio::time::timeout(self.request_timeout, pending.rx).await {
            Ok(Ok(reply)) => reply,
            Ok(Err(_)) => return Err(McpError::TransportClosed),
            Err(_) => return Err(McpError::RequestTimeout(self.request_timeout)),
        };
        match reply {
            Ok(value) => Ok(value),
            Err(RpcError {
                code,
                message,
                data,
            }) => Err(McpError::RpcError {
                code,
                message,
                data,
            }),
        }
    }
}

#[async_trait]
impl McpClient for StdioMcpClient {
    async fn list_tools(&self) -> AnyResult<Vec<ToolSchema>> {
        let result = self.call("tools/list", json!({})).await?;
        let tools_arr = result
            .get("tools")
            .and_then(Value::as_array)
            .ok_or_else(|| McpError::Protocol("tools/list missing `tools` array".into()))?;
        let mut out = Vec::with_capacity(tools_arr.len());
        for entry in tools_arr {
            let name = entry
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| McpError::Protocol("tool entry missing `name`".into()))?
                .to_string();
            let description = entry
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let input_schema = entry
                .get("inputSchema")
                .cloned()
                .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
            out.push(ToolSchema {
                name,
                description,
                input_schema,
            });
        }
        Ok(out)
    }

    async fn invoke(&self, name: &str, arguments: Value) -> AnyResult<ToolResult> {
        let params = json!({ "name": name, "arguments": arguments });
        let result = self.call("tools/call", params).await?;

        // The MCP spec says tools/call returns:
        //   { "content": [{"type":"text","text":...}|{"type":"image",...}|...],
        //     "isError": bool }
        // We map the first content entry into our ToolResult shape. Multiple
        // content entries are uncommon enough in practice that taking the
        // first is the right v1 trade-off; downstream code can still surface
        // a helpful message.
        let content_arr = result
            .get("content")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let is_error = result
            .get("isError")
            .and_then(Value::as_bool)
            .unwrap_or(false);

        let first = content_arr.into_iter().next();
        let parsed = match first {
            None => ToolResult::Text(String::new()),
            Some(entry) => parse_content_entry(entry)?,
        };

        if is_error {
            // Wrap text errors so the model sees the failure mode in the
            // result body. Image errors keep their image; the model can
            // still react.
            let parsed = match parsed {
                ToolResult::Text(t) => ToolResult::Text(format!("[mcp tool error] {t}")),
                other => other,
            };
            return Ok(parsed);
        }
        Ok(parsed)
    }
}

fn parse_content_entry(entry: Value) -> Result<ToolResult, McpError> {
    let kind = entry.get("type").and_then(Value::as_str).unwrap_or("");
    match kind {
        "text" => {
            let text = entry
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            Ok(ToolResult::Text(text))
        }
        "image" => {
            let mime = entry
                .get("mimeType")
                .and_then(Value::as_str)
                .ok_or_else(|| McpError::Protocol("image content missing `mimeType`".into()))?
                .to_string();
            let data_b64 = entry
                .get("data")
                .and_then(Value::as_str)
                .ok_or_else(|| McpError::Protocol("image content missing `data`".into()))?;
            use base64::Engine;
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(data_b64)
                .map_err(|e| McpError::Protocol(format!("image base64 decode failed: {e}")))?;
            Ok(ToolResult::Image { mime, bytes })
        }
        // resource / structured → JSON passthrough so the model can read it.
        _ => Ok(ToolResult::Json(entry)),
    }
}

/// Owns the spawned child plus its background I/O tasks. Awaiting
/// [`Self::wait_for_exit`] returns when the child terminates.
pub struct ChildLifeline {
    pub label: String,
    child: Option<tokio::process::Child>,
    transport: Option<TransportHandles>,
    stderr_task: Option<JoinHandle<()>>,
}

impl ChildLifeline {
    pub fn pid(&self) -> Option<u32> {
        self.child.as_ref().and_then(|c| c.id())
    }

    /// Block until the child exits naturally. Returns whatever
    /// `Child::wait` returns. The supervisor races this against the
    /// shared shutdown signal.
    pub async fn wait_for_exit(&mut self) -> std::io::Result<ExitStatus> {
        let child = self
            .child
            .as_mut()
            .expect("wait_for_exit called after shutdown");
        child.wait().await
    }

    /// Send SIGTERM to the child's process group, wait `term_timeout`,
    /// then SIGKILL if still alive. Always best-effort — we don't fail
    /// daemon shutdown on a stuck child.
    pub async fn shutdown(mut self, term_timeout: Duration) {
        let label = self.label.clone();
        if let Some(mut child) = self.child.take() {
            #[cfg(unix)]
            if let Some(pid) = child.id() {
                let gid: i32 = -(pid as i32);
                // SAFETY: libc::kill is thread-safe; failures (ESRCH for an
                // already-exited process) are surfaced via errno and ignored.
                unsafe {
                    libc::kill(gid, libc::SIGTERM);
                }
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
        }

        if let Some(handles) = self.transport.take() {
            handles.shutdown_and_join().await;
        }
        if let Some(task) = self.stderr_task.take() {
            let _ = tokio::time::timeout(Duration::from_millis(500), task).await;
        }
    }
}

/// JoinHandles for the read+write tasks of a single transport. Held by
/// `ChildLifeline` (production) or by tests for direct cleanup.
pub struct TransportHandles {
    read_task: Option<JoinHandle<()>>,
    write_task: Option<JoinHandle<()>>,
    /// Fires when the read loop terminates (EOF or error). Lets a test
    /// or the supervisor detect transport death without owning the child.
    pub read_done: Option<oneshot::Receiver<()>>,
}

impl TransportHandles {
    pub async fn shutdown_and_join(mut self) {
        if let Some(t) = self.read_task.take() {
            t.abort();
            let _ = tokio::time::timeout(Duration::from_millis(500), t).await;
        }
        if let Some(t) = self.write_task.take() {
            t.abort();
            let _ = tokio::time::timeout(Duration::from_millis(500), t).await;
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
    let mut line = String::new();
    loop {
        line.clear();
        let n = match reader.read_line(&mut line).await {
            Ok(n) => n,
            Err(e) => {
                warn!(
                    target: "assistd::mcp",
                    server = %label,
                    "MCP stdout read error: {e}",
                );
                break;
            }
        };
        if n == 0 {
            debug!(target: "assistd::mcp", server = %label, "MCP stdout EOF");
            break;
        }
        if line.len() > MAX_LINE_BYTES {
            warn!(
                target: "assistd::mcp",
                server = %label,
                bytes = line.len(),
                "MCP stdout line over {MAX_LINE_BYTES} bytes; dropping connection",
            );
            break;
        }
        let trimmed = line.trim_end_matches(['\n', '\r']);
        if trimmed.is_empty() {
            continue;
        }
        match serde_json::from_str::<Response>(trimmed) {
            Ok(resp) => correlator.deliver(resp),
            Err(e) => {
                warn!(
                    target: "assistd::mcp",
                    server = %label,
                    "MCP stdout JSON parse error: {e}; line: {trimmed}",
                );
            }
        }
    }
    correlator.fail_all(|| RpcError {
        code: -32603,
        message: "MCP transport closed".into(),
        data: None,
    });
    let _ = done_tx.send(());
}

async fn write_loop<W: tokio::io::AsyncWrite + Unpin>(
    mut stream: W,
    mut rx: mpsc::Receiver<Vec<u8>>,
    label: String,
) {
    while let Some(bytes) = rx.recv().await {
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

async fn forward_stderr(stream: tokio::process::ChildStderr, label: String) {
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
    use super::*;
    use serde_json::json;
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, duplex};

    /// Pretend MCP server: reads requests from `client_to_server`, replies
    /// on `server_to_client`. Hands back the spawned task so callers can
    /// `await` it after they're done.
    fn fake_server<F, Fut>(
        client_to_server: tokio::io::DuplexStream,
        server_to_client: tokio::io::DuplexStream,
        handler: F,
    ) -> JoinHandle<()>
    where
        F: Fn(serde_json::Value) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = serde_json::Value> + Send,
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
                let req: serde_json::Value = match serde_json::from_str(line.trim()) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if req.get("id").is_none() {
                    // Notification (e.g. notifications/initialized). Ignore.
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

    /// Helper: standard mock that replies to every request id with the
    /// given closure-built `result`.
    async fn make_client_with_handler<F, Fut>(
        handler: F,
    ) -> (
        Arc<StdioMcpClient>,
        TransportHandles,
        JoinHandle<()>,
        // Keep the server-side half alive so writes don't fail.
        tokio::task::JoinHandle<()>,
    )
    where
        F: Fn(serde_json::Value) -> Fut + Send + Clone + 'static,
        Fut: std::future::Future<Output = serde_json::Value> + Send,
    {
        let (client_write, server_read) = duplex(8192); // client -> server
        let (server_write, client_read) = duplex(8192); // server -> client

        let server_task = fake_server(server_read, server_write, handler);

        let (client, handles) = StdioMcpClient::from_streams(
            client_read,
            client_write,
            "test".into(),
            Duration::from_secs(2),
        )
        .await
        .unwrap();
        let dummy = tokio::spawn(async {});
        (client, handles, server_task, dummy)
    }

    #[tokio::test]
    async fn list_tools_round_trip() {
        let handler = |req: serde_json::Value| async move {
            let id = req["id"].clone();
            let method = req["method"].as_str().unwrap();
            assert_eq!(method, "tools/list");
            json!({
                "jsonrpc": "2.0",
                "id": id,
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
        let (client, handles, server, _) = make_client_with_handler(handler).await;

        let tools = client.list_tools().await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "echo");
        assert_eq!(tools[0].description, "echoes its input");
        assert_eq!(tools[0].input_schema["type"], "object");

        handles.shutdown_and_join().await;
        let _ = server.await;
    }

    #[tokio::test]
    async fn invoke_text_response() {
        let handler = |req: serde_json::Value| async move {
            let id = req["id"].clone();
            assert_eq!(req["method"], "tools/call");
            assert_eq!(req["params"]["name"], "echo");
            json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "content": [{"type": "text", "text": "hello"}],
                    "isError": false
                }
            })
        };
        let (client, handles, server, _) = make_client_with_handler(handler).await;

        let result = client.invoke("echo", json!({"x": "hi"})).await.unwrap();
        match result {
            ToolResult::Text(t) => assert_eq!(t, "hello"),
            other => panic!("expected Text, got {other:?}"),
        }
        handles.shutdown_and_join().await;
        let _ = server.await;
    }

    #[tokio::test]
    async fn invoke_image_response_decodes_base64() {
        let handler = |req: serde_json::Value| async move {
            let id = req["id"].clone();
            json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "content": [{
                        "type": "image",
                        "mimeType": "image/png",
                        "data": "3q2+7w==" // 0xDE 0xAD 0xBE 0xEF
                    }],
                    "isError": false
                }
            })
        };
        let (client, handles, server, _) = make_client_with_handler(handler).await;
        let result = client.invoke("snap", json!({})).await.unwrap();
        match result {
            ToolResult::Image { mime, bytes } => {
                assert_eq!(mime, "image/png");
                assert_eq!(bytes, vec![0xDE, 0xAD, 0xBE, 0xEF]);
            }
            other => panic!("expected Image, got {other:?}"),
        }
        handles.shutdown_and_join().await;
        let _ = server.await;
    }

    #[tokio::test]
    async fn rpc_error_surfaces_as_error() {
        let handler = |req: serde_json::Value| async move {
            let id = req["id"].clone();
            json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {"code": -32601, "message": "method not found"}
            })
        };
        let (client, handles, server, _) = make_client_with_handler(handler).await;
        let err = client.list_tools().await.unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("method not found"),
            "expected RpcError to surface; got {msg}"
        );
        handles.shutdown_and_join().await;
        let _ = server.await;
    }

    #[tokio::test]
    async fn request_timeout_fires_when_server_silent() {
        // Server that reads but never replies.
        let (client_write, server_read) = duplex(8192);
        let (server_write, client_read) = duplex(8192);
        let silent = tokio::spawn(async move {
            let mut reader = BufReader::new(server_read);
            let mut line = String::new();
            // Drain forever, never write a response.
            loop {
                line.clear();
                if reader.read_line(&mut line).await.unwrap_or(0) == 0 {
                    break;
                }
            }
            drop(server_write); // appease borrowck
        });

        let (client, handles) = StdioMcpClient::from_streams(
            client_read,
            client_write,
            "silent".into(),
            Duration::from_millis(150),
        )
        .await
        .unwrap();

        let err = client.list_tools().await.unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("timed out"), "{msg}");
        handles.shutdown_and_join().await;
        let _ = silent.await;
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

        // Issue a request, then drop the server side so the read loop EOFs.
        let call = tokio::spawn({
            let c = client.clone();
            async move { c.list_tools().await }
        });
        // Give the client a beat to send the request before EOF'ing the server.
        tokio::time::sleep(Duration::from_millis(50)).await;
        drop(server_read);
        drop(server_write);

        let err = call.await.unwrap().unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.to_lowercase().contains("transport closed") || msg.contains("MCP transport"),
            "{msg}"
        );
        handles.shutdown_and_join().await;
    }
}
