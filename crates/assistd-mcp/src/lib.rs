#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::print_stdout,
        clippy::print_stderr
    )
)]

//! MCP (Model Context Protocol) **client** for assistd.
//!
//! Two transports are supported: newline-delimited JSON-RPC over a
//! child process's stdin/stdout ([`stdio::StdioMcpClient`]) and HTTP+SSE
//! ([`sse::SseMcpClient`]). Each MCP server is wrapped in a
//! [`handle::McpServerHandle`] whose supervisor task spawns/restarts
//! the transport with exponential backoff (capped at five consecutive
//! failures, then permanently parked).
//!
//! # Architecture
//!
//! From the daemon's perspective, an MCP server is a *tool source*:
//! - On startup, the daemon connects to each configured MCP server via
//!   [`McpServerHandle::start`] and immediately calls
//!   [`adapt_handle_as_tools`] to discover the server's tool catalog.
//! - Each discovered tool is wrapped in an [`McpToolAdapter`] (which
//!   in turn is wrapped in a [`health_route::HealthRoutedTool`]) and
//!   registered with the existing `ToolRegistry` alongside native tools.
//! - When the LLM calls a wrapped tool, the health-routed adapter checks
//!   the supervisor's health watch; if the server is healthy the call
//!   forwards to [`McpClient::invoke`], otherwise the wrapper returns a
//!   synthetic tool-error JSON immediately so the model never hangs on
//!   a dead transport.
//!
//! # Why a separate trait, not a `Tool` impl?
//!
//! A single MCP server typically exposes many tools, with distinct
//! schemas and dynamic discovery. Modeling each tool as its own
//! `Tool` impl would either require generating a struct per tool at
//! compile time (impossible: schemas come from the wire) or a giant
//! match. Instead we keep `McpClient` as a connection-level handle,
//! and use the [`McpToolAdapter`] type to turn each
//! `(client, tool_name, schema)` tuple into a `Tool` registry entry.
//!
//! # Image-bearing tool results
//!
//! [`ToolResult::Image`] is rendered into the same JSON shape the
//! `RunTool` already emits: a top-level `output`/`exit_code`/`truncated`
//! envelope plus an `attachments` array carrying the image bytes as
//! base64. The agent loop's existing `parse_attachment` helper lifts
//! those into `Attachment::Image` for the next turn, same path as the
//! local `see` / `screenshot` commands.

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use assistd_tools::Tool;
use async_trait::async_trait;
use serde_json::{Value, json};

pub mod backoff;
pub mod error;
pub mod handle;
pub mod health_route;
pub mod jsonrpc;
pub mod prompt;
pub mod sse;
pub mod stdio;

pub use error::{McpError, mcp_error_line};
pub use handle::{HealthState, McpServerHandle, SwitchingClient, TransportConfig};
pub use health_route::HealthRoutedTool;
pub use sse::{SseConfig, SseLifeline, SseMcpClient};
pub use stdio::{ChildLifeline, StdioConfig, StdioMcpClient};

/// One tool exposed by an MCP server.
#[derive(Debug, Clone)]
pub struct ToolSchema {
    /// Server-native tool name (no daemon-side prefix). The daemon
    /// decorates this when registering; see [`adapt_client_as_tools`].
    pub name: String,
    /// Human-readable description the LLM sees.
    pub description: String,
    /// JSON Schema for the `arguments` object the LLM will pass.
    /// Forwarded verbatim to the underlying server.
    pub input_schema: Value,
}

/// Result of an MCP tool invocation. Mirrors the three shapes
/// llama.cpp / OpenAI tool calls already flow through:
/// strings, structured JSON, and images.
#[derive(Debug, Clone)]
pub enum ToolResult {
    Text(String),
    Image { mime: String, bytes: Vec<u8> },
    Json(Value),
}

/// A connection to a single MCP server.
#[async_trait]
pub trait McpClient: Send + Sync + 'static {
    /// List all tools the server currently exposes. Called at
    /// startup; concrete implementations may refresh on demand
    /// but must be safe to call concurrently.
    async fn list_tools(&self) -> Result<Vec<ToolSchema>>;

    /// Invoke `name` with `arguments`. The daemon strips any
    /// registry-side prefix (e.g. `mcp__server-name__`) before
    /// forwarding, so `name` is the server-native identifier.
    async fn invoke(&self, name: &str, arguments: Value) -> Result<ToolResult>;
}

/// Wraps a single (client, tool) pair as a [`Tool`] registry entry.
/// The daemon's tool-registry code is unchanged: it iterates
/// `ToolRegistry` and the adapter takes care of the dispatch hop.
pub struct McpToolAdapter {
    client: Arc<dyn McpClient>,
    schema: ToolSchema,
    /// The name as exposed to the LLM. May be a prefixed form
    /// (`mcp__foo__bar`) when the daemon namespaces multiple servers.
    /// We carry the *server-native* name in `schema.name` and the
    /// *registry* name here so `invoke` can strip the prefix.
    registry_name: String,
}

impl McpToolAdapter {
    /// Create an adapter for a single `(client, schema)` pair under the given registry name.
    pub fn new(client: Arc<dyn McpClient>, schema: ToolSchema, registry_name: String) -> Self {
        Self {
            client,
            schema,
            registry_name,
        }
    }
}

#[async_trait]
impl Tool for McpToolAdapter {
    fn name(&self) -> &str {
        &self.registry_name
    }

    fn description(&self) -> &str {
        &self.schema.description
    }

    fn parameters_schema(&self) -> Value {
        self.schema.input_schema.clone()
    }

    /// Always resolves to `Ok(json)`; failure is rendered into the
    /// envelope itself (with `exit_code: -1` and a convention-compliant
    /// `[error] …. <Hint>: <recovery>` line in `output`) instead of
    /// bubbling a Rust `Err` to the agent loop. That keeps the model on a
    /// single recovery path: read the error line, pick a different tool or
    /// retry. Bubbling the error would be caught by `dispatch_tool_call`'s
    /// generic catch-all, which collapses every variant into the same
    /// "tool invocation failed" message and loses the recovery hint.
    async fn invoke(&self, args: Value) -> Result<Value> {
        let start = Instant::now();
        let outcome = self.client.invoke(&self.schema.name, args).await;
        let duration_ms = start.elapsed().as_millis();
        match outcome {
            Ok(r) => Ok(tool_result_to_json(r, duration_ms)),
            Err(e) => Ok(error_envelope(&self.registry_name, &e, duration_ms)),
        }
    }
}

/// Build the failure-shaped JSON envelope. Mirrors the success shape so
/// the agent loop's `dispatch_tool_call` reads `output`/`exit_code`/
/// `duration_ms` straight into the tool-result message body without a
/// branch. The error line itself comes from [`mcp_error_line`] when the
/// underlying error is an [`McpError`]; for unexpected error types we
/// fall back to a generic Convention-compliant line so the body still
/// carries a `<Hint>: <recovery>` clause.
fn error_envelope(tool_name: &str, e: &anyhow::Error, duration_ms: u128) -> Value {
    let line = match e.downcast_ref::<McpError>() {
        Some(mcp_err) => mcp_error_line(tool_name, mcp_err),
        None => format!(
            "[error] {tool_name}: tool invocation failed: {e}. \
             Try: a different command\n"
        ),
    };
    json!({
        "type": "error",
        "output": line,
        "exit_code": -1,
        "duration_ms": duration_ms,
        "truncated": false,
    })
}

/// Render an [`McpClient`] [`ToolResult`] into the JSON shape the agent
/// loop already understands. The shape mirrors what `RunTool` emits via
/// `assistd_tools::PresentResult`:
///
/// ```json
/// {
///   "type": "text" | "image" | "json",
///   "output": "...",                    // model-visible body
///   "exit_code": 0,
///   "duration_ms": <real-elapsed-ms>,
///   "truncated": false,
///   "attachments": [...]                // present only for Image
/// }
/// ```
///
/// `duration_ms` is supplied by the caller so it reflects the actual MCP
/// RPC round-trip time; the TUI surfaces this in the `[exit:N | Xms]`
/// footer alongside the colored bar.
///
/// `attachments[].data` is the base64-encoded image bytes (NOT
/// `bytes_base64`); this matches `assistd_core::agent::parse_attachment`.
fn tool_result_to_json(r: ToolResult, duration_ms: u128) -> Value {
    match r {
        ToolResult::Text(s) => json!({
            "type": "text",
            "output": s,
            "exit_code": 0,
            "duration_ms": duration_ms,
            "truncated": false,
        }),
        ToolResult::Json(v) => json!({
            "type": "json",
            "output": v.to_string(),
            "value": v,
            "exit_code": 0,
            "duration_ms": duration_ms,
            "truncated": false,
        }),
        ToolResult::Image { mime, bytes } => {
            let len = bytes.len();
            let data_b64 = base64_encode(&bytes);
            json!({
                "type": "image",
                "output": format!("(image: {mime}, {len} bytes)"),
                "exit_code": 0,
                "duration_ms": duration_ms,
                "truncated": false,
                "attachments": [
                    {"type": "image", "mime": mime, "data": data_b64}
                ],
            })
        }
    }
}

/// Build [`Tool`] registry entries for every tool an MCP client
/// exposes. The daemon decorates the registry name with `name_prefix`
/// (typically `mcp__<server>`) using `__` as a separator; `:` is
/// illegal in OpenAI/Anthropic function names.
///
/// This sibling does NOT wrap entries in [`HealthRoutedTool`]; use
/// [`adapt_handle_as_tools`] for production daemon wiring.
pub async fn adapt_client_as_tools(
    client: Arc<dyn McpClient>,
    name_prefix: &str,
) -> Result<Vec<Box<dyn Tool>>> {
    let schemas = client.list_tools().await?;
    let mut out: Vec<Box<dyn Tool>> = Vec::with_capacity(schemas.len());
    for schema in schemas {
        let registry_name = registry_name(name_prefix, &schema.name);
        out.push(Box::new(McpToolAdapter::new(
            client.clone(),
            schema,
            registry_name,
        )));
    }
    Ok(out)
}

/// Production wiring: build [`Tool`] entries for every tool exposed by
/// the server, each wrapped in a [`HealthRoutedTool`] keyed off the
/// supervisor's health watch. Use this when registering MCP tools in
/// the daemon's tool registry.
pub async fn adapt_handle_as_tools(
    handle: &McpServerHandle,
    name_prefix: &str,
) -> Result<Vec<Box<dyn Tool>>> {
    let client = handle.client();
    let schemas = client.list_tools().await?;
    let health_rx = handle.watch_health();
    let server_name = handle.name.clone();

    let mut out: Vec<Box<dyn Tool>> = Vec::with_capacity(schemas.len());
    for schema in schemas {
        let registry_name = registry_name(name_prefix, &schema.name);
        let adapter = McpToolAdapter::new(client.clone(), schema, registry_name);
        let routed = HealthRoutedTool::new(adapter, server_name.clone(), health_rx.clone());
        out.push(Box::new(routed));
    }
    Ok(out)
}

fn registry_name(prefix: &str, server_native: &str) -> String {
    if prefix.is_empty() {
        server_native.to_string()
    } else {
        format!("{prefix}__{server_native}")
    }
}

fn base64_encode(bytes: &[u8]) -> String {
    use base64_dep::Engine;
    base64_dep::engine::general_purpose::STANDARD.encode(bytes)
}

// Internal alias to keep base64 a single import path. assistd-tools
// already depends on base64 transitively via assistd-ipc; declaring
// our own dep would be redundant. We re-export under a private name
// so the public surface of this crate stays minimal.
mod base64_dep {
    pub use base64::*;
}

/// Returns the crate version string from `Cargo.toml`.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trait-only fake server: returns a static tool list and echoes
    /// arguments back as a Text result. Lets us exercise the adapter
    /// without standing up real I/O.
    struct FakeMcpClient {
        schemas: Vec<ToolSchema>,
    }

    #[async_trait]
    impl McpClient for FakeMcpClient {
        async fn list_tools(&self) -> Result<Vec<ToolSchema>> {
            Ok(self.schemas.clone())
        }

        async fn invoke(&self, name: &str, arguments: Value) -> Result<ToolResult> {
            Ok(ToolResult::Text(format!("called {name} with {arguments}")))
        }
    }

    /// Fake server that always fails its `invoke` with a caller-supplied
    /// `McpError`, after an optional sleep. Used to exercise the
    /// adapter's error-envelope path and duration tracking.
    struct ErrFakeClient {
        err: parking_lot::Mutex<Option<McpError>>,
        sleep: std::time::Duration,
    }

    #[async_trait]
    impl McpClient for ErrFakeClient {
        async fn list_tools(&self) -> Result<Vec<ToolSchema>> {
            Ok(vec![ToolSchema {
                name: "search".into(),
                description: "search".into(),
                input_schema: json!({"type": "object"}),
            }])
        }

        async fn invoke(&self, _name: &str, _arguments: Value) -> Result<ToolResult> {
            tokio::time::sleep(self.sleep).await;
            let e = self.err.lock().take().expect("err pre-armed");
            Err(e.into())
        }
    }

    fn one_tool_client() -> Arc<dyn McpClient> {
        Arc::new(FakeMcpClient {
            schemas: vec![ToolSchema {
                name: "search".into(),
                description: "search the web".into(),
                input_schema: json!({"type": "object", "properties": {}}),
            }],
        })
    }

    fn err_client_with(err: McpError, sleep: std::time::Duration) -> Arc<dyn McpClient> {
        Arc::new(ErrFakeClient {
            err: parking_lot::Mutex::new(Some(err)),
            sleep,
        })
    }

    #[tokio::test]
    async fn adapter_forwards_tool_metadata() {
        let client = one_tool_client();
        let tools = adapt_client_as_tools(client, "mcp__web").await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name(), "mcp__web__search");
        assert_eq!(tools[0].description(), "search the web");
    }

    #[tokio::test]
    async fn adapter_strips_prefix_before_invoking() {
        // Registry sees `mcp__web__search` but the upstream server
        // expects bare `search`. The adapter must forward the
        // server-native name.
        let client = one_tool_client();
        let tools = adapt_client_as_tools(client, "mcp__web").await.unwrap();
        let tool = tools.into_iter().next().unwrap();
        let out = tool.invoke(json!({"q": "rust"})).await.unwrap();
        assert_eq!(out["type"], "text");
        let text = out["output"].as_str().unwrap();
        assert!(text.starts_with("called search "), "{text}");
    }

    #[tokio::test]
    async fn empty_prefix_leaves_name_unchanged() {
        let client = one_tool_client();
        let tools = adapt_client_as_tools(client, "").await.unwrap();
        assert_eq!(tools[0].name(), "search");
    }

    #[test]
    fn image_tool_result_lifts_into_attachments_array() {
        let v = tool_result_to_json(
            ToolResult::Image {
                mime: "image/png".into(),
                bytes: vec![0xDE, 0xAD, 0xBE, 0xEF],
            },
            0,
        );
        assert_eq!(v["type"], "image");
        assert_eq!(v["exit_code"], 0);
        assert_eq!(v["truncated"], false);
        let attachments = v["attachments"].as_array().expect("attachments array");
        assert_eq!(attachments.len(), 1);
        assert_eq!(attachments[0]["type"], "image");
        assert_eq!(attachments[0]["mime"], "image/png");
        assert_eq!(attachments[0]["data"], "3q2+7w==");
        let output = v["output"].as_str().unwrap();
        assert!(output.contains("image/png"));
        assert!(output.contains("4 bytes"));
    }

    #[test]
    fn text_tool_result_uses_dispatch_envelope() {
        let v = tool_result_to_json(ToolResult::Text("hello".into()), 0);
        assert_eq!(v["type"], "text");
        assert_eq!(v["output"], "hello");
        assert_eq!(v["exit_code"], 0);
        assert_eq!(v["truncated"], false);
    }

    #[test]
    fn json_tool_result_carries_value_and_string_output() {
        let v = tool_result_to_json(ToolResult::Json(json!({"answer": 42})), 0);
        assert_eq!(v["type"], "json");
        assert_eq!(v["value"], json!({"answer": 42}));
        assert!(v["output"].as_str().unwrap().contains("answer"));
    }

    #[test]
    fn tool_result_carries_duration_ms_through_envelope() {
        let v = tool_result_to_json(ToolResult::Text("hi".into()), 42);
        assert_eq!(v["duration_ms"], 42);
    }

    #[test]
    fn version_is_not_empty() {
        assert!(!version().is_empty());
    }

    /// Acceptance: when the upstream client fails with `RpcError`, the
    /// adapter must surface the failure as a tool-result envelope (not
    /// a Rust `Err`), with `exit_code: -1` and a convention-compliant
    /// `[error] <tool>: …. Check: …` line. This is what lets the agent
    /// loop's `dispatch_tool_call` thread the body straight into the
    /// model's tool-role message without losing the recovery hint.
    #[tokio::test]
    async fn adapter_returns_error_envelope_on_rpc_error() {
        let client = err_client_with(
            McpError::RpcError {
                code: -32602,
                message: "missing field 'query'".into(),
                data: None,
            },
            std::time::Duration::ZERO,
        );
        let tools = adapt_client_as_tools(client, "mcp__web").await.unwrap();
        let tool = tools.into_iter().next().unwrap();
        let out = tool.invoke(json!({})).await.unwrap();
        assert_eq!(out["type"], "error");
        assert_eq!(out["exit_code"], -1);
        assert_eq!(out["truncated"], false);
        let body = out["output"].as_str().unwrap();
        assert!(
            body.starts_with("[error] mcp__web__search: "),
            "missing convention prefix: {body}"
        );
        assert!(body.contains("-32602"), "missing rpc code: {body}");
        assert!(body.contains("missing field"), "missing message: {body}");
        assert!(body.contains("Check:"), "missing recovery hint: {body}");
    }

    /// Acceptance: timeouts get a `Try:` recovery hint that suggests
    /// retrying or shrinking the request, distinct from `RpcError`'s
    /// `Check:` (which says "the input is wrong"). The model needs the
    /// distinction to pick the right next step.
    #[tokio::test]
    async fn adapter_returns_error_envelope_on_timeout() {
        let client = err_client_with(
            McpError::RequestTimeout(std::time::Duration::from_secs(30)),
            std::time::Duration::ZERO,
        );
        let tools = adapt_client_as_tools(client, "mcp__web").await.unwrap();
        let tool = tools.into_iter().next().unwrap();
        let out = tool.invoke(json!({})).await.unwrap();
        assert_eq!(out["type"], "error");
        assert_eq!(out["exit_code"], -1);
        let body = out["output"].as_str().unwrap();
        assert!(body.contains("timed out"), "{body}");
        assert!(body.contains("30s"), "{body}");
        assert!(body.contains("Try:"), "{body}");
    }

    /// Acceptance: `duration_ms` reflects the real RPC round-trip, not
    /// a hardcoded 0. The TUI's `[exit:N | Xms]` footer surfaces this
    /// to the user, so a stuck call should look stuck.
    #[tokio::test]
    async fn adapter_records_real_duration_ms() {
        let client = err_client_with(McpError::ServerDown, std::time::Duration::from_millis(20));
        let tools = adapt_client_as_tools(client, "mcp__web").await.unwrap();
        let tool = tools.into_iter().next().unwrap();
        let out = tool.invoke(json!({})).await.unwrap();
        let dur = out["duration_ms"].as_u64().expect("duration_ms u64");
        assert!(
            dur >= 15,
            "duration must include the upstream sleep: got {dur}ms"
        );
    }
}
