#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::print_stdout,
        clippy::print_stderr
    )
)]

//! MCP (Model Context Protocol) client. A server is reached over stdio
//! or HTTP+SSE, supervised by [`McpServerHandle`], and each tool it
//! exposes becomes a [`Tool`] via [`adapt_handle_as_tools`].

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use assistd_tools::Tool;
use async_trait::async_trait;
use base64::Engine;
use serde_json::{Value, json};

pub mod backoff;
pub mod error;
pub mod handle;
pub mod health_route;
pub mod jsonrpc;
mod protocol;
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
    /// Server-native tool name, without any registry prefix.
    pub name: String,
    pub description: String,
    /// JSON Schema for the `arguments` object, forwarded verbatim.
    pub input_schema: Value,
}

/// Result of an MCP tool invocation.
#[derive(Debug, Clone)]
pub enum ToolResult {
    Text(String),
    Image { mime: String, bytes: Vec<u8> },
    Json(Value),
}

/// A connection to a single MCP server.
#[async_trait]
pub trait McpClient: Send + Sync + 'static {
    /// Every tool the server currently exposes. Safe to call concurrently.
    async fn list_tools(&self) -> Result<Vec<ToolSchema>>;

    /// Invoke the server-native tool `name` with `arguments`.
    async fn invoke(&self, name: &str, arguments: Value) -> Result<ToolResult>;
}

/// Exposes one MCP tool as a [`Tool`] under `registry_name`; the
/// server-native name stays in `schema.name`.
pub struct McpToolAdapter {
    client: Arc<dyn McpClient>,
    schema: ToolSchema,
    registry_name: String,
}

impl McpToolAdapter {
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

    /// Always `Ok`: a failed call is rendered into the envelope by
    /// [`error_envelope`] so the model keeps its recovery hint.
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

/// The tool-error envelope. It has the same `output` / `exit_code` /
/// `duration_ms` / `truncated` shape as a successful result, with
/// `exit_code: -1` and an `[error] <tool>: <what>. <Hint>: <recovery>`
/// line as `output`, so the model handles every failure the same way.
/// [`HealthRoutedTool`] emits the same shape without an RPC.
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

/// Render a [`ToolResult`] as the tool-result envelope; an image goes
/// into `attachments[].data` as base64.
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
            let data_b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
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

/// [`Tool`] entries for every tool the server exposes, named
/// `<name_prefix>__<tool>` and gated on the supervisor's health.
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

#[cfg(test)]
async fn adapt_client_as_tools(
    client: Arc<dyn McpClient>,
    name_prefix: &str,
) -> Result<Vec<Box<dyn Tool>>> {
    let schemas = client.list_tools().await?;
    Ok(schemas
        .into_iter()
        .map(|schema| {
            let name = registry_name(name_prefix, &schema.name);
            Box::new(McpToolAdapter::new(client.clone(), schema, name)) as Box<dyn Tool>
        })
        .collect())
}

fn registry_name(prefix: &str, server_native: &str) -> String {
    if prefix.is_empty() {
        server_native.to_string()
    } else {
        format!("{prefix}__{server_native}")
    }
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
