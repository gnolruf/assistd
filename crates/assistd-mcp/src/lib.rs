//! MCP (Model Context Protocol) client. A server is reached over stdio
//! or HTTP+SSE, supervised by [`McpServerHandle`], and each tool it
//! exposes becomes a [`Tool`] via [`adapt_handle_as_tools`].

use std::sync::Arc;
use std::time::Instant;

use assistd_tools::{Tool, ToolError};
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
    async fn list_tools(&self) -> Result<Vec<ToolSchema>, McpError>;

    /// Invoke the server-native tool `name` with `arguments`.
    async fn invoke(&self, name: &str, arguments: Value) -> Result<ToolResult, McpError>;
}

/// Exposes one MCP tool as a [`Tool`] under `registry_name`; the
/// server-native name stays in `schema.name`.
pub struct McpToolAdapter {
    client: Arc<dyn McpClient>,
    schema: ToolSchema,
    registry_name: String,
}

impl McpToolAdapter {
    /// Adapter that invokes `schema.name` on `client`.
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

    /// Always `Ok`: a failed call comes back as an error envelope
    /// (`exit_code: -1`, with a [`mcp_error_line`] as `output`) so the
    /// model keeps its recovery hint.
    async fn invoke(&self, args: Value) -> Result<Value, ToolError> {
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
fn error_envelope(tool_name: &str, e: &McpError, duration_ms: u128) -> Value {
    json!({
        "type": "error",
        "output": mcp_error_line(tool_name, e),
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
) -> Result<Vec<Box<dyn Tool>>, McpError> {
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
) -> Result<Vec<Box<dyn Tool>>, McpError> {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Returns a static tool list and echoes arguments back as text.
    struct FakeMcpClient {
        schemas: Vec<ToolSchema>,
    }

    #[async_trait]
    impl McpClient for FakeMcpClient {
        async fn list_tools(&self) -> Result<Vec<ToolSchema>, McpError> {
            Ok(self.schemas.clone())
        }

        async fn invoke(&self, name: &str, arguments: Value) -> Result<ToolResult, McpError> {
            Ok(ToolResult::Text(format!("called {name} with {arguments}")))
        }
    }

    /// Fails its one `invoke` with a pre-armed error after `sleep`.
    struct ErrFakeClient {
        err: parking_lot::Mutex<Option<McpError>>,
        sleep: std::time::Duration,
    }

    #[async_trait]
    impl McpClient for ErrFakeClient {
        async fn list_tools(&self) -> Result<Vec<ToolSchema>, McpError> {
            Ok(vec![ToolSchema {
                name: "search".into(),
                description: "search".into(),
                input_schema: json!({"type": "object"}),
            }])
        }

        async fn invoke(&self, _name: &str, _arguments: Value) -> Result<ToolResult, McpError> {
            tokio::time::sleep(self.sleep).await;
            Err(self.err.lock().take().expect("err pre-armed"))
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

    async fn failing_tool(err: McpError, sleep: std::time::Duration) -> Box<dyn Tool> {
        let client = Arc::new(ErrFakeClient {
            err: parking_lot::Mutex::new(Some(err)),
            sleep,
        });
        let mut tools = adapt_client_as_tools(client, "mcp__web").await.unwrap();
        tools.pop().unwrap()
    }

    #[tokio::test]
    async fn adapter_forwards_tool_metadata_under_registry_name() {
        for (prefix, expected_name) in [("mcp__web", "mcp__web__search"), ("", "search")] {
            let tools = adapt_client_as_tools(one_tool_client(), prefix)
                .await
                .unwrap();
            let [tool] = tools.as_slice() else {
                panic!("prefix {prefix:?}: expected one tool");
            };
            assert_eq!(tool.name(), expected_name, "prefix {prefix:?}");
            assert_eq!(tool.description(), "search the web");
            assert_eq!(
                tool.parameters_schema(),
                json!({"type": "object", "properties": {}})
            );
        }
    }

    #[tokio::test]
    async fn adapter_invokes_the_server_native_name() {
        let tools = adapt_client_as_tools(one_tool_client(), "mcp__web")
            .await
            .unwrap();
        let out = tools[0].invoke(json!({"q": "rust"})).await.unwrap();
        assert_eq!(out["type"], "text");
        assert_eq!(out["output"], r#"called search with {"q":"rust"}"#);
    }

    #[test]
    fn tool_results_render_as_dispatch_envelopes() {
        let cases = [
            (
                ToolResult::Text("hello".into()),
                json!({
                    "type": "text",
                    "output": "hello",
                    "exit_code": 0,
                    "duration_ms": 42,
                    "truncated": false,
                }),
            ),
            (
                ToolResult::Json(json!({"answer": 42})),
                json!({
                    "type": "json",
                    "output": r#"{"answer":42}"#,
                    "value": {"answer": 42},
                    "exit_code": 0,
                    "duration_ms": 42,
                    "truncated": false,
                }),
            ),
            (
                ToolResult::Image {
                    mime: "image/png".into(),
                    bytes: vec![0xDE, 0xAD, 0xBE, 0xEF],
                },
                json!({
                    "type": "image",
                    "output": "(image: image/png, 4 bytes)",
                    "exit_code": 0,
                    "duration_ms": 42,
                    "truncated": false,
                    "attachments": [
                        {"type": "image", "mime": "image/png", "data": "3q2+7w=="}
                    ],
                }),
            ),
        ];
        for (result, expected) in cases {
            assert_eq!(tool_result_to_json(result, 42), expected);
        }
    }

    #[tokio::test]
    async fn adapter_turns_client_errors_into_error_envelopes() {
        let err = || McpError::RpcError {
            code: -32602,
            message: "missing field 'query'".into(),
            data: None,
        };
        let tool = failing_tool(err(), std::time::Duration::ZERO).await;
        let mut out = tool.invoke(json!({})).await.unwrap();
        assert!(out["duration_ms"].is_u64(), "{out}");
        out.as_object_mut().unwrap().remove("duration_ms");
        assert_eq!(
            out,
            json!({
                "type": "error",
                "output": mcp_error_line("mcp__web__search", &err()),
                "exit_code": -1,
                "truncated": false,
            })
        );
    }

    #[tokio::test]
    async fn adapter_records_real_duration_ms() {
        let tool = failing_tool(McpError::ServerDown, std::time::Duration::from_millis(20)).await;
        let out = tool.invoke(json!({})).await.unwrap();
        let dur = out["duration_ms"].as_u64().expect("duration_ms u64");
        assert!(
            dur >= 15,
            "duration must include the upstream sleep: got {dur}ms"
        );
    }
}
