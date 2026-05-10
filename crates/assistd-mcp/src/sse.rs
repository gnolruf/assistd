//! HTTP+SSE JSON-RPC transport for remote MCP servers.
//!
//! Wire shape per the MCP HTTP+SSE binding:
//!   * Client opens `GET <base_url>` with `Accept: text/event-stream`.
//!     The server's first SSE event is `event: endpoint\ndata: <url>`,
//!     telling the client which URL to POST requests to. (Servers that
//!     don't emit this event are assumed to accept POSTs at `base_url`.)
//!   * Client POSTs JSON-RPC requests to that URL. Server returns 202.
//!   * Server replies arrive over the SSE stream as
//!     `event: message\ndata: <json-rpc-response>`.
//!
//! Two background tasks per server:
//!   * SSE reader: long-lived GET, parses event stream, hands JSON-RPC
//!     responses to the [`Correlator`]. A `read_timeout` on the underlying
//!     `reqwest::Client` flips the connection unhealthy if the server
//!     stalls (no events, no SSE comments) for that long.
//!   * Ping task: sends JSON-RPC `ping` every `ping_interval`. If the
//!     response doesn't arrive within `request_timeout`, signals the
//!     read loop to drop the connection. Catches the case where the
//!     SSE socket stays open but the server stops processing requests.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result as AnyResult;
use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde_json::{Value, json};
use tokio::sync::{RwLock, oneshot, watch};
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};
use url::Url;

use crate::error::McpError;
use crate::jsonrpc::{Correlator, Reply, Response, RpcError, notification_line};
use crate::{McpClient, ToolResult, ToolSchema};

const PROTOCOL_VERSION: &str = "2024-11-05";
const CLIENT_NAME: &str = "assistd";
const CLIENT_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Per-server SSE configuration.
#[derive(Debug, Clone)]
pub struct SseConfig {
    pub url: String,
    pub headers: HashMap<String, String>,
    pub request_timeout: Duration,
    pub read_timeout: Duration,
    pub ping_interval: Duration,
    pub label: String,
}

impl SseConfig {
    /// Create a config with default timeouts (30s request/read, 15s ping interval).
    pub fn new(label: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            headers: HashMap::new(),
            request_timeout: Duration::from_secs(30),
            read_timeout: Duration::from_secs(30),
            ping_interval: Duration::from_secs(15),
            label: label.into(),
        }
    }
}

/// [`McpClient`] implementation that speaks JSON-RPC over HTTP+SSE.
pub struct SseMcpClient {
    #[allow(dead_code)]
    label: String,
    correlator: Arc<Correlator>,
    http: reqwest::Client,
    base_url: Url,
    post_url: Arc<RwLock<Option<Url>>>,
    headers: HeaderMap,
    request_timeout: Duration,
}

impl SseMcpClient {
    /// Connect to an SSE MCP server: opens the SSE GET stream, waits
    /// for the optional `endpoint` discovery event, performs the
    /// `initialize` handshake, and starts the periodic ping task.
    pub async fn connect(cfg: SseConfig) -> Result<(Arc<Self>, SseLifeline), McpError> {
        let base_url = Url::parse(&cfg.url)
            .map_err(|e| McpError::config(format!("invalid SSE url `{}`", cfg.url), e))?;

        let mut headers = HeaderMap::new();
        for (k, v) in &cfg.headers {
            let name = HeaderName::from_bytes(k.as_bytes())
                .map_err(|e| McpError::config(format!("invalid header name `{k}`"), e))?;
            let value = HeaderValue::from_str(v)
                .map_err(|e| McpError::config(format!("invalid header value `{v}`"), e))?;
            headers.insert(name, value);
        }

        // We deliberately set `read_timeout` (per-read) rather than
        // `timeout` (whole-request); the SSE GET is meant to be long-lived,
        // so the per-read deadline is what catches a frozen server.
        let http = reqwest::Client::builder()
            .read_timeout(cfg.read_timeout)
            .timeout(cfg.request_timeout)
            .build()?;

        let correlator = Arc::new(Correlator::new());
        let post_url = Arc::new(RwLock::new(None));
        let (cancel_tx, cancel_rx) = watch::channel(false);
        let (endpoint_ready_tx, endpoint_ready_rx) = oneshot::channel::<()>();
        let (done_tx, done_rx) = oneshot::channel::<()>();

        let client = Arc::new(Self {
            label: cfg.label.clone(),
            correlator: correlator.clone(),
            http: http.clone(),
            base_url: base_url.clone(),
            post_url: post_url.clone(),
            headers: headers.clone(),
            request_timeout: cfg.request_timeout,
        });

        let reader = ReadLoop {
            http: http.clone(),
            base_url: base_url.clone(),
            headers: headers.clone(),
            correlator: correlator.clone(),
            post_url: post_url.clone(),
            label: cfg.label.clone(),
            cancel_rx: cancel_rx.clone(),
            endpoint_ready_tx: Some(endpoint_ready_tx),
            done_tx,
        };
        let stream_task = tokio::spawn(reader.run());

        // Wait briefly for the optional `endpoint` discovery event. If
        // we don't get one in 5s, fall back to assuming POSTs go to the
        // base URL. Many MCP SSE servers omit the event.
        let _ = tokio::time::timeout(Duration::from_secs(5), endpoint_ready_rx).await;
        if post_url.read().await.is_none() {
            *post_url.write().await = Some(base_url.clone());
            debug!(
                target: "assistd::mcp",
                server = %cfg.label,
                "no endpoint event received; defaulting POST URL to base URL",
            );
        }

        if let Err(e) = client.initialize().await {
            warn!(
                target: "assistd::mcp",
                server = %cfg.label,
                error = %e,
                "MCP SSE initialize failed; tearing down",
            );
            let _ = cancel_tx.send(true);
            stream_task.abort();
            return Err(e);
        }

        // Periodic ping task — separate from the SSE stream so a server
        // that holds the SSE connection open while wedged still gets
        // detected. On ping failure, we flip the cancel watch which the
        // read loop selects on; that closes the transport and trips the
        // supervisor's restart path.
        let ping_task = tokio::spawn(ping_loop(
            client.clone(),
            cfg.ping_interval,
            cfg.label.clone(),
            cancel_rx,
            cancel_tx.clone(),
        ));

        info!(
            target: "assistd::mcp",
            server = %cfg.label,
            "MCP SSE server initialized",
        );

        Ok((
            client,
            SseLifeline {
                label: cfg.label,
                cancel_tx,
                stream_task: Some(stream_task),
                ping_task: Some(ping_task),
                done_rx: Some(done_rx),
            },
        ))
    }

    async fn initialize(&self) -> Result<(), McpError> {
        let params = json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": { "tools": {} },
            "clientInfo": { "name": CLIENT_NAME, "version": CLIENT_VERSION },
        });
        let _ = self.call("initialize", params).await?;
        let bytes = notification_line("notifications/initialized", json!({}))?;
        // Strip the trailing newline — POST bodies don't need it.
        let body = &bytes[..bytes.len().saturating_sub(1)];
        let post = self
            .post_url
            .read()
            .await
            .clone()
            .unwrap_or_else(|| self.base_url.clone());
        let resp = self
            .http
            .post(post)
            .headers(self.headers.clone())
            .body(body.to_vec())
            .header("Content-Type", "application/json")
            .send()
            .await
            .map_err(McpError::from)?;
        if !resp.status().is_success() {
            return Err(McpError::Protocol(format!(
                "POST notifications/initialized failed: HTTP {}",
                resp.status()
            )));
        }
        Ok(())
    }

    async fn call(&self, method: &'static str, params: Value) -> Result<Value, McpError> {
        let pending = self.correlator.next_request(method, params)?;
        let body = pending.frame_json()?;
        let post = self
            .post_url
            .read()
            .await
            .clone()
            .unwrap_or_else(|| self.base_url.clone());

        let resp = self
            .http
            .post(post)
            .headers(self.headers.clone())
            .body(body)
            .header("Content-Type", "application/json")
            .send()
            .await
            .map_err(McpError::from)?;
        if !resp.status().is_success() {
            return Err(McpError::Protocol(format!(
                "POST {method} failed: HTTP {}",
                resp.status()
            )));
        }

        let reply = match tokio::time::timeout(self.request_timeout, pending.rx).await {
            Ok(Ok(reply)) => reply,
            Ok(Err(_)) => return Err(McpError::TransportClosed),
            Err(_) => return Err(McpError::RequestTimeout(self.request_timeout)),
        };
        reply_to_result(reply)
    }
}

fn reply_to_result(reply: Reply) -> Result<Value, McpError> {
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

#[async_trait]
impl McpClient for SseMcpClient {
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
        let result = self
            .call(
                "tools/call",
                json!({ "name": name, "arguments": arguments }),
            )
            .await?;
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
            let parsed = match parsed {
                ToolResult::Text(t) => ToolResult::Text(format!("[mcp tool error] {t}")),
                other => other,
            };
            return Ok(parsed);
        }
        Ok(parsed)
    }
}

/// Owns the SSE reader task, the ping task, and a cancellation watch
/// that the supervisor flips on shutdown. `wait_for_disconnect` resolves
/// when the read loop terminates (either naturally or via cancellation).
pub struct SseLifeline {
    pub label: String,
    cancel_tx: watch::Sender<bool>,
    stream_task: Option<JoinHandle<()>>,
    ping_task: Option<JoinHandle<()>>,
    done_rx: Option<oneshot::Receiver<()>>,
}

impl SseLifeline {
    /// Await the SSE read loop's termination signal.
    pub async fn wait_for_disconnect(&mut self) {
        if let Some(rx) = self.done_rx.take() {
            let _ = rx.await;
        }
    }

    /// Cancel the SSE and ping tasks and wait briefly for them to finish.
    pub async fn shutdown(mut self) {
        let _ = self.cancel_tx.send(true);
        if let Some(t) = self.stream_task.take() {
            let _ = tokio::time::timeout(Duration::from_millis(500), t).await;
        }
        if let Some(t) = self.ping_task.take() {
            let _ = tokio::time::timeout(Duration::from_millis(500), t).await;
        }
    }
}

struct ReadLoop {
    http: reqwest::Client,
    base_url: Url,
    headers: HeaderMap,
    correlator: Arc<Correlator>,
    post_url: Arc<RwLock<Option<Url>>>,
    label: String,
    cancel_rx: watch::Receiver<bool>,
    endpoint_ready_tx: Option<oneshot::Sender<()>>,
    done_tx: oneshot::Sender<()>,
}

impl ReadLoop {
    async fn run(self) {
        let Self {
            http,
            base_url,
            headers,
            correlator,
            post_url,
            label,
            mut cancel_rx,
            mut endpoint_ready_tx,
            done_tx,
        } = self;

        let resp = match http
            .get(base_url.clone())
            .headers(headers)
            .header("Accept", "text/event-stream")
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                warn!(
                    target: "assistd::mcp",
                    server = %label,
                    "SSE connect failed: {e}",
                );
                correlator.fail_all(closed_err);
                let _ = done_tx.send(());
                return;
            }
        };
        if !resp.status().is_success() {
            warn!(
                target: "assistd::mcp",
                server = %label,
                status = %resp.status(),
                "SSE GET returned non-success status",
            );
            correlator.fail_all(closed_err);
            let _ = done_tx.send(());
            return;
        }

        let mut stream = resp.bytes_stream();
        let mut parser = EventParser::new();

        loop {
            let chunk = tokio::select! {
                _ = cancel_rx.changed() => {
                    if *cancel_rx.borrow() {
                        debug!(target: "assistd::mcp", server = %label, "SSE read cancelled");
                        break;
                    }
                    continue;
                }
                chunk = stream.next() => chunk,
            };

            match chunk {
                Some(Ok(bytes)) => {
                    parser.push(&bytes);
                    while let Some(event) = parser.next_event() {
                        handle_event(
                            event,
                            &correlator,
                            &post_url,
                            &mut endpoint_ready_tx,
                            &label,
                        )
                        .await;
                    }
                }
                Some(Err(e)) => {
                    warn!(
                        target: "assistd::mcp",
                        server = %label,
                        "SSE stream error: {e}",
                    );
                    break;
                }
                None => {
                    debug!(target: "assistd::mcp", server = %label, "SSE stream ended");
                    break;
                }
            }
        }

        correlator.fail_all(closed_err);
        let _ = done_tx.send(());
    }
}

fn closed_err() -> RpcError {
    RpcError {
        code: -32603,
        message: "MCP transport closed".into(),
        data: None,
    }
}

async fn handle_event(
    event: SseEvent,
    correlator: &Arc<Correlator>,
    post_url: &Arc<RwLock<Option<Url>>>,
    endpoint_ready_tx: &mut Option<oneshot::Sender<()>>,
    label: &str,
) {
    match event.event_type.as_str() {
        "endpoint" => {
            // Server tells us where to POST. Spec allows either an
            // absolute URL or a path relative to the base. We treat
            // anything that doesn't parse as an absolute URL as a
            // missed event for now; the daemon falls back to the base
            // URL after a 5s timeout in `connect`.
            let resolved = Url::parse(&event.data).ok();
            if let Some(url) = resolved {
                debug!(
                    target: "assistd::mcp",
                    server = %label,
                    endpoint = %url,
                    "received SSE endpoint event",
                );
                *post_url.write().await = Some(url);
                if let Some(tx) = endpoint_ready_tx.take() {
                    let _ = tx.send(());
                }
            }
        }
        "message" | "" => {
            // Default event type per the SSE spec is "message".
            match serde_json::from_str::<Response>(&event.data) {
                Ok(resp) => correlator.deliver(resp),
                Err(e) => {
                    warn!(
                        target: "assistd::mcp",
                        server = %label,
                        "SSE message JSON parse error: {e}; data: {}",
                        event.data,
                    );
                }
            }
        }
        other => {
            debug!(
                target: "assistd::mcp",
                server = %label,
                event_type = %other,
                "ignoring SSE event of unknown type",
            );
        }
    }
}

async fn ping_loop(
    client: Arc<SseMcpClient>,
    interval: Duration,
    label: String,
    mut cancel_rx: watch::Receiver<bool>,
    cancel_tx: watch::Sender<bool>,
) {
    let mut ticker = tokio::time::interval(interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    ticker.tick().await;
    loop {
        tokio::select! {
            _ = cancel_rx.changed() => {
                if *cancel_rx.borrow() {
                    return;
                }
            }
            _ = ticker.tick() => {
                match client.call("ping", json!({})).await {
                    Ok(_) => {
                        debug!(target: "assistd::mcp", server = %label, "ping ok");
                    }
                    Err(McpError::RpcError { code: -32601, message, .. }) => {
                        // Server doesn't implement `ping`. Treat as healthy
                        // — the SSE stream itself will tell us about death.
                        debug!(
                            target: "assistd::mcp",
                            server = %label,
                            "server does not implement ping ({message}); disabling pings",
                        );
                        return;
                    }
                    Err(e) => {
                        warn!(
                            target: "assistd::mcp",
                            server = %label,
                            "ping failed: {e}; flipping transport unhealthy",
                        );
                        let _ = cancel_tx.send(true);
                        return;
                    }
                }
            }
        }
    }
}

/// A single parsed Server-Sent Event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    pub event_type: String,
    pub data: String,
    pub id: Option<String>,
}

/// Stateful parser that accepts incremental byte chunks (as they arrive
/// from the HTTP body) and yields complete SSE events one at a time.
#[derive(Default)]
pub struct EventParser {
    buf: Vec<u8>,
    cur: PartialEvent,
}

#[derive(Default)]
struct PartialEvent {
    event_type: Option<String>,
    data_lines: Vec<String>,
    id: Option<String>,
}

impl EventParser {
    /// Create a new, empty parser.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed the next chunk of bytes from the HTTP body into the parser.
    pub fn push(&mut self, chunk: &[u8]) {
        self.buf.extend_from_slice(chunk);
    }

    /// Pull the next complete event off the buffer. Returns `None` if
    /// the buffer doesn't yet contain a blank-line terminator.
    pub fn next_event(&mut self) -> Option<SseEvent> {
        loop {
            // SSE allows \r\n or \n as line terminators.
            let nl = self.buf.iter().position(|&b| b == b'\n')?;
            let mut line: Vec<u8> = self.buf.drain(..=nl).collect();
            line.pop();
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            if line.is_empty() {
                let cur = std::mem::take(&mut self.cur);
                if cur.event_type.is_none() && cur.data_lines.is_empty() && cur.id.is_none() {
                    continue;
                }
                let event_type = cur.event_type.unwrap_or_else(|| "message".to_string());
                let data = cur.data_lines.join("\n");
                return Some(SseEvent {
                    event_type,
                    data,
                    id: cur.id,
                });
            }
            if line.first() == Some(&b':') {
                continue;
            }
            // A line with no colon is treated as a field name with empty value (per spec).
            let line_str = match std::str::from_utf8(&line) {
                Ok(s) => s,
                Err(_) => continue, // drop non-UTF-8 lines
            };
            let (field, value) = match line_str.split_once(':') {
                Some((f, v)) => {
                    // Strip a single leading space from value, per spec.
                    let v = v.strip_prefix(' ').unwrap_or(v);
                    (f, v)
                }
                None => (line_str, ""),
            };
            match field {
                "event" => self.cur.event_type = Some(value.to_string()),
                "data" => self.cur.data_lines.push(value.to_string()),
                "id" => self.cur.id = Some(value.to_string()),
                _ => {} // ignore unknown fields (including "retry")
            }
        }
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
        _ => Ok(ToolResult::Json(entry)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drive(parser: &mut EventParser, chunks: &[&[u8]]) -> Vec<SseEvent> {
        let mut events = Vec::new();
        for chunk in chunks {
            parser.push(chunk);
            while let Some(e) = parser.next_event() {
                events.push(e);
            }
        }
        events
    }

    #[test]
    fn single_message_event() {
        let mut p = EventParser::new();
        let events = drive(&mut p, &[b"event: message\ndata: hello\n\n"]);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "message");
        assert_eq!(events[0].data, "hello");
    }

    #[test]
    fn default_event_type_is_message() {
        let mut p = EventParser::new();
        let events = drive(&mut p, &[b"data: hi\n\n"]);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "message");
        assert_eq!(events[0].data, "hi");
    }

    #[test]
    fn multi_line_data_concatenated_with_newlines() {
        let mut p = EventParser::new();
        let events = drive(&mut p, &[b"data: line1\ndata: line2\ndata: line3\n\n"]);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "line1\nline2\nline3");
    }

    #[test]
    fn comment_lines_ignored() {
        let mut p = EventParser::new();
        let events = drive(&mut p, &[b": keep-alive\n: another\ndata: x\n\n"]);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "x");
    }

    #[test]
    fn handles_crlf_line_endings() {
        let mut p = EventParser::new();
        let events = drive(&mut p, &[b"event: endpoint\r\ndata: /msg?s=1\r\n\r\n"]);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "endpoint");
        assert_eq!(events[0].data, "/msg?s=1");
    }

    #[test]
    fn split_across_chunks() {
        let mut p = EventParser::new();
        let events = drive(
            &mut p,
            &[b"event: ", b"message\ndata: par", b"tial\n", b"\n"],
        );
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "message");
        assert_eq!(events[0].data, "partial");
    }

    #[test]
    fn back_to_back_events() {
        let mut p = EventParser::new();
        let events = drive(&mut p, &[b"data: one\n\ndata: two\n\ndata: three\n\n"]);
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].data, "one");
        assert_eq!(events[1].data, "two");
        assert_eq!(events[2].data, "three");
    }

    #[test]
    fn empty_lines_without_pending_event_dropped() {
        let mut p = EventParser::new();
        let events = drive(&mut p, &[b"\n\n\ndata: ok\n\n"]);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "ok");
    }

    #[test]
    fn id_field_captured() {
        let mut p = EventParser::new();
        let events = drive(&mut p, &[b"id: 42\nevent: message\ndata: x\n\n"]);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].id.as_deref(), Some("42"));
    }

    #[test]
    fn line_without_colon_treated_as_field_name() {
        // SSE spec: a line without a colon is treated as a field with
        // an empty value. "data" alone means empty data line.
        let mut p = EventParser::new();
        let events = drive(&mut p, &[b"data\ndata: rest\n\n"]);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "\nrest");
    }

    #[test]
    fn leading_space_after_colon_stripped() {
        let mut p = EventParser::new();
        let events = drive(
            &mut p,
            &[b"data:nospaces\ndata: leading-space-stripped\n\n"],
        );
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "nospaces\nleading-space-stripped");
    }
}
