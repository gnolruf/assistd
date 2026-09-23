//! JSON-RPC over the MCP HTTP+SSE binding: requests are POSTed to the
//! URL the server's `endpoint` event names (or `base_url` if it emits
//! none), replies arrive on a long-lived `GET` event stream. A ping
//! task drops the connection when the server stops answering while
//! the stream stays open.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde_json::{Value, json};
use tokio::sync::{RwLock, oneshot, watch};
use tokio_util::task::AbortOnDropHandle;
use tracing::{debug, info, warn};
use url::Url;

use crate::error::McpError;
use crate::jsonrpc::{Correlator, Response, notification_line};
use crate::{McpClient, ToolResult, ToolSchema, protocol};

/// The reader drops the connection rather than buffer an event past
/// this, so a misbehaving server cannot exhaust memory.
pub const MAX_EVENT_BYTES: usize = 1024 * 1024;

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

/// [`McpClient`] over HTTP+SSE.
pub struct SseMcpClient {
    label: String,
    correlator: Arc<Correlator>,
    http: reqwest::Client,
    base_url: Url,
    post_url: Arc<RwLock<Option<Url>>>,
    headers: HeaderMap,
    request_timeout: Duration,
}

impl SseMcpClient {
    /// Open the event stream, wait briefly for an `endpoint` event, run
    /// the initialize handshake, and start the ping task.
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

        let http = reqwest::Client::builder()
            .read_timeout(cfg.read_timeout)
            .timeout(cfg.request_timeout)
            .build()?;

        let stream_http = reqwest::Client::builder()
            .read_timeout(cfg.read_timeout)
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
            http: stream_http,
            base_url: base_url.clone(),
            headers: headers.clone(),
            correlator: correlator.clone(),
            post_url: post_url.clone(),
            label: cfg.label.clone(),
            cancel_rx: cancel_rx.clone(),
            endpoint_ready_tx: Some(endpoint_ready_tx),
            done_tx,
        };
        let stream_task = AbortOnDropHandle::new(tokio::spawn(reader.run()));

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
            return Err(e);
        }

        let ping_task = AbortOnDropHandle::new(tokio::spawn(ping_loop(
            client.clone(),
            cfg.ping_interval,
            cfg.label.clone(),
            cancel_rx,
            cancel_tx.clone(),
        )));

        info!(
            target: "assistd::mcp",
            server = %cfg.label,
            "MCP SSE server initialized",
        );

        Ok((
            client,
            SseLifeline {
                cancel_tx,
                stream_task,
                ping_task,
                done_rx,
            },
        ))
    }

    async fn initialize(&self) -> Result<(), McpError> {
        let result = self
            .call("initialize", protocol::initialize_params())
            .await?;
        protocol::warn_on_version_mismatch(&self.label, &result);
        let bytes = notification_line("notifications/initialized", json!({}))?;
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
            return Err(McpError::HttpStatus {
                method: "notifications/initialized",
                status: resp.status(),
            });
        }
        Ok(())
    }

    async fn call(&self, method: &'static str, params: Value) -> Result<Value, McpError> {
        let mut pending = self.correlator.next_request(method, params)?;
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
            return Err(McpError::HttpStatus {
                method,
                status: resp.status(),
            });
        }

        protocol::await_reply(&mut pending.rx, self.request_timeout).await
    }
}

#[async_trait]
impl McpClient for SseMcpClient {
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

/// The reader and ping tasks of one SSE connection. Dropping it
/// aborts both, closing the event stream.
pub struct SseLifeline {
    cancel_tx: watch::Sender<bool>,
    stream_task: AbortOnDropHandle<()>,
    ping_task: AbortOnDropHandle<()>,
    done_rx: oneshot::Receiver<()>,
}

impl SseLifeline {
    /// Resolves when the read loop terminates.
    pub async fn wait_for_disconnect(&mut self) {
        let _ = (&mut self.done_rx).await;
    }

    /// Cancel both tasks and wait briefly for each to finish; a task
    /// still running after that is aborted.
    pub async fn shutdown(self) {
        let _ = self.cancel_tx.send(true);
        for task in [self.stream_task, self.ping_task] {
            let _ = tokio::time::timeout(Duration::from_millis(500), task).await;
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
                correlator.fail_all();
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
            correlator.fail_all();
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
                    let overflowed = loop {
                        match parser.next_event() {
                            Ok(Some(event)) => {
                                handle_event(
                                    event,
                                    &correlator,
                                    &base_url,
                                    &post_url,
                                    &mut endpoint_ready_tx,
                                    &label,
                                )
                                .await;
                            }
                            Ok(None) => break false,
                            Err(EventTooLarge) => break true,
                        }
                    };
                    if overflowed {
                        warn!(
                            target: "assistd::mcp",
                            server = %label,
                            "SSE event over {MAX_EVENT_BYTES} bytes; dropping connection",
                        );
                        break;
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

        correlator.fail_all();
        let _ = done_tx.send(());
    }
}

fn resolve_endpoint(base_url: &Url, data: &str) -> Result<Url, url::ParseError> {
    base_url.join(data.trim())
}

async fn handle_event(
    event: SseEvent,
    correlator: &Arc<Correlator>,
    base_url: &Url,
    post_url: &Arc<RwLock<Option<Url>>>,
    endpoint_ready_tx: &mut Option<oneshot::Sender<()>>,
    label: &str,
) {
    match event.event_type.as_str() {
        "endpoint" => match resolve_endpoint(base_url, &event.data) {
            Ok(url) => {
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
            Err(e) => {
                warn!(
                    target: "assistd::mcp",
                    server = %label,
                    "unusable SSE endpoint event `{}`: {e}",
                    event.data,
                );
            }
        },
        "message" | "" => match serde_json::from_str::<Response>(&event.data) {
            Ok(resp) => correlator.deliver(resp),
            Err(e) => {
                warn!(
                    target: "assistd::mcp",
                    server = %label,
                    "SSE message JSON parse error: {e}; data: {}",
                    event.data,
                );
            }
        },
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

/// The event being assembled has grown past [`MAX_EVENT_BYTES`].
#[derive(Debug, PartialEq, Eq)]
pub struct EventTooLarge;

/// Incremental SSE parser: feed body chunks, pull complete events.
#[derive(Default)]
pub struct EventParser {
    buf: Vec<u8>,
    cur: PartialEvent,
}

#[derive(Default)]
struct PartialEvent {
    event_type: Option<String>,
    data: String,
    id: Option<String>,
}

impl EventParser {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, chunk: &[u8]) {
        self.buf.extend_from_slice(chunk);
    }

    /// The next complete event, or `None` until a blank-line terminator
    /// has arrived. Errors once the event being assembled exceeds
    /// [`MAX_EVENT_BYTES`]; the parser should then be discarded.
    pub fn next_event(&mut self) -> Result<Option<SseEvent>, EventTooLarge> {
        loop {
            let Some(nl) = self.buf.iter().position(|&b| b == b'\n') else {
                return if self.buf.len() + self.cur.data.len() > MAX_EVENT_BYTES {
                    Err(EventTooLarge)
                } else {
                    Ok(None)
                };
            };
            let mut line: Vec<u8> = self.buf.drain(..=nl).collect();
            line.pop();
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            if line.is_empty() {
                let mut cur = std::mem::take(&mut self.cur);
                if cur.event_type.is_none() && cur.data.is_empty() && cur.id.is_none() {
                    continue;
                }
                cur.data.pop();
                return Ok(Some(SseEvent {
                    event_type: cur.event_type.unwrap_or_else(|| "message".to_string()),
                    data: cur.data,
                    id: cur.id,
                }));
            }
            if line.first() == Some(&b':') {
                continue;
            }
            let Ok(line_str) = std::str::from_utf8(&line) else {
                continue;
            };
            let (field, value) = match line_str.split_once(':') {
                Some((f, v)) => {
                    let v = v.strip_prefix(' ').unwrap_or(v);
                    (f, v)
                }
                None => (line_str, ""),
            };
            match field {
                "event" => self.cur.event_type = Some(value.to_string()),
                "data" => {
                    self.cur.data.push_str(value);
                    self.cur.data.push('\n');
                    if self.cur.data.len() > MAX_EVENT_BYTES {
                        return Err(EventTooLarge);
                    }
                }
                "id" => self.cur.id = Some(value.to_string()),
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drive(parser: &mut EventParser, chunks: &[&[u8]]) -> Vec<SseEvent> {
        let mut events = Vec::new();
        for chunk in chunks {
            parser.push(chunk);
            while let Some(e) = parser.next_event().unwrap() {
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
    fn unterminated_line_past_cap_is_rejected() {
        let mut p = EventParser::new();
        p.push(&vec![b'x'; MAX_EVENT_BYTES + 1]);
        assert_eq!(p.next_event(), Err(EventTooLarge));
    }

    #[test]
    fn data_lines_past_cap_are_rejected() {
        let mut p = EventParser::new();
        let line = format!("data: {}\n", "x".repeat(1024));
        let err = (0..=MAX_EVENT_BYTES / 1024).find_map(|_| {
            p.push(line.as_bytes());
            p.next_event().err()
        });
        assert_eq!(err, Some(EventTooLarge));
    }

    #[test]
    fn events_under_cap_are_not_rejected_cumulatively() {
        let mut p = EventParser::new();
        let event = format!("data: {}\n\n", "x".repeat(MAX_EVENT_BYTES / 2));
        let events = drive(
            &mut p,
            &[event.as_bytes(), event.as_bytes(), event.as_bytes()],
        );
        assert_eq!(events.len(), 3);
    }

    #[test]
    fn relative_endpoint_resolved_against_base() {
        let base = Url::parse("http://127.0.0.1:8931/sse").unwrap();
        let url = resolve_endpoint(&base, "/messages?sessionId=abc").unwrap();
        assert_eq!(url.as_str(), "http://127.0.0.1:8931/messages?sessionId=abc");
    }

    #[test]
    fn path_relative_endpoint_resolved_against_base_directory() {
        let base = Url::parse("http://example.com/mcp/sse").unwrap();
        let url = resolve_endpoint(&base, "messages/?session_id=1").unwrap();
        assert_eq!(
            url.as_str(),
            "http://example.com/mcp/messages/?session_id=1"
        );
    }

    #[test]
    fn absolute_endpoint_replaces_base() {
        let base = Url::parse("http://127.0.0.1:8931/sse").unwrap();
        let url = resolve_endpoint(&base, "https://other.example/post").unwrap();
        assert_eq!(url.as_str(), "https://other.example/post");
    }

    #[test]
    fn endpoint_payload_is_trimmed() {
        let base = Url::parse("http://127.0.0.1:8931/sse").unwrap();
        let url = resolve_endpoint(&base, " /messages\r").unwrap();
        assert_eq!(url.as_str(), "http://127.0.0.1:8931/messages");
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
