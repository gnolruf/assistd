//! MCP (Model Context Protocol) client. A server is reached over stdio
//! or HTTP+SSE, supervised by [`McpServerHandle`], and each tool it
//! exposes becomes a [`Tool`] via [`adapt_handle_as_tools`].

use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use assistd_tools::presentation::{PresentSpec, TextTruncator, TruncatedText};
use assistd_tools::{Tool, ToolError};
use async_trait::async_trait;
use base64::Engine;
use serde_json::{Value, json};

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
pub trait McpClient: fmt::Debug + Send + Sync + 'static {
    /// Every tool the server currently exposes. Safe to call concurrently.
    async fn list_tools(&self) -> Result<Vec<ToolSchema>, McpError>;

    /// Invoke the server-native tool `name` with `arguments`.
    async fn invoke(&self, name: &str, arguments: Value) -> Result<ToolResult, McpError>;
}

/// Exposes one MCP tool as a [`Tool`] under `registry_name`; the
/// server-native name stays in `schema.name`.
#[derive(Debug)]
pub struct McpToolAdapter {
    client: Arc<dyn McpClient>,
    schema: ToolSchema,
    registry_name: String,
    truncator: Arc<TextTruncator>,
}

impl McpToolAdapter {
    /// Adapter that invokes `schema.name` on `client`, cutting text and
    /// JSON results with `truncator` before they reach the model.
    pub fn new(
        client: Arc<dyn McpClient>,
        schema: ToolSchema,
        registry_name: String,
        truncator: Arc<TextTruncator>,
    ) -> Self {
        Self {
            client,
            schema,
            registry_name,
            truncator,
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

    /// Always `Ok`: a failed call becomes an error envelope carrying an
    /// [`mcp_error_line`].
    async fn invoke(&self, args: Value) -> Result<Value, ToolError> {
        let start = Instant::now();
        let outcome = self.client.invoke(&self.schema.name, args).await;
        let duration_ms = start.elapsed().as_millis();
        match outcome {
            Ok(result) => Ok(tool_result_to_json(result, duration_ms, &self.truncator)),
            Err(err) => Ok(error_envelope(&self.registry_name, &err, duration_ms)),
        }
    }
}

/// Same shape as a successful result, with `exit_code: -1` and an
/// [`mcp_error_line`] as `output`.
fn error_envelope(tool_name: &str, err: &McpError, duration_ms: u128) -> Value {
    json!({
        "type": "error",
        "output": mcp_error_line(tool_name, err),
        "exit_code": -1,
        "duration_ms": duration_ms,
        "truncated": false,
    })
}

/// Render a [`ToolResult`] as the tool-result envelope; text and JSON
/// bodies are cut by `truncator`, an image goes into
/// `attachments[].data` as base64.
fn tool_result_to_json(result: ToolResult, duration_ms: u128, truncator: &TextTruncator) -> Value {
    match result {
        ToolResult::Text(text) => text_envelope("text", truncator.truncate(text), duration_ms),
        ToolResult::Json(value) => {
            let mut envelope =
                text_envelope("json", truncator.truncate(value.to_string()), duration_ms);
            envelope["value"] = value;
            envelope
        }
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

fn text_envelope(kind: &str, cut: TruncatedText, duration_ms: u128) -> Value {
    let mut envelope = json!({
        "type": kind,
        "output": cut.text,
        "exit_code": 0,
        "duration_ms": duration_ms,
        "truncated": cut.truncated,
    });
    if let Some(path) = cut.overflow_file {
        envelope["overflow_file"] = json!(path.to_string_lossy());
    }
    envelope
}

/// [`Tool`] entries for every tool the server exposes, named
/// `<name_prefix>__<tool>` and gated on the supervisor's health. Results
/// past `output`'s caps are cut, with the overflow spilled as
/// `mcp-<server>-<n>.txt`.
pub async fn adapt_handle_as_tools(
    handle: &McpServerHandle,
    name_prefix: &str,
    output: PresentSpec,
) -> Result<Vec<Box<dyn Tool>>, McpError> {
    let client = handle.client();
    let schemas = client.list_tools().await?;
    let health_rx = handle.watch_health();
    let server_name = handle.name.clone();
    let truncator = Arc::new(TextTruncator::new(output, format!("mcp-{server_name}")));

    let mut tools: Vec<Box<dyn Tool>> = Vec::with_capacity(schemas.len());
    for schema in schemas {
        let registry_name = registry_name(name_prefix, &schema.name);
        let adapter = McpToolAdapter::new(client.clone(), schema, registry_name, truncator.clone());
        let routed = HealthRoutedTool::new(adapter, server_name.clone(), health_rx.clone());
        tools.push(Box::new(routed));
    }
    Ok(tools)
}

#[cfg(test)]
async fn adapt_client_as_tools(
    client: Arc<dyn McpClient>,
    name_prefix: &str,
    output: PresentSpec,
) -> Result<Vec<Box<dyn Tool>>, McpError> {
    let schemas = client.list_tools().await?;
    let truncator = Arc::new(TextTruncator::new(output, "mcp-test"));
    Ok(schemas
        .into_iter()
        .map(|schema| {
            let name = registry_name(name_prefix, &schema.name);
            let adapter = McpToolAdapter::new(client.clone(), schema, name, truncator.clone());
            Box::new(adapter) as Box<dyn Tool>
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
    use std::time::Duration;

    use super::*;

    /// Returns a static tool list and echoes arguments back as text.
    #[derive(Debug)]
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
    #[derive(Debug)]
    struct ErrFakeClient {
        err: parking_lot::Mutex<Option<McpError>>,
        sleep: Duration,
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

    async fn failing_tool(err: McpError, sleep: Duration) -> Box<dyn Tool> {
        let client = Arc::new(ErrFakeClient {
            err: parking_lot::Mutex::new(Some(err)),
            sleep,
        });
        let mut tools = adapt_client_as_tools(client, "mcp__web", PresentSpec::default())
            .await
            .unwrap();
        tools.pop().unwrap()
    }

    fn unlimited() -> TextTruncator {
        TextTruncator::new(PresentSpec::default(), "mcp-test")
    }

    #[tokio::test]
    async fn adapter_forwards_tool_metadata_under_registry_name() {
        for (prefix, expected_name) in [("mcp__web", "mcp__web__search"), ("", "search")] {
            let tools = adapt_client_as_tools(one_tool_client(), prefix, PresentSpec::default())
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
        let tools = adapt_client_as_tools(one_tool_client(), "mcp__web", PresentSpec::default())
            .await
            .unwrap();
        let out = tools[0].invoke(json!({"q": "rust"})).await.unwrap();
        assert_eq!(out["type"], "text");
        assert_eq!(out["output"], r#"called search with {"q":"rust"}"#);
        assert_eq!(out["truncated"], false);
    }

    #[tokio::test]
    async fn adapter_cuts_long_results_and_spills_the_rest() {
        let dir = tempfile::tempdir().unwrap();
        let spec = PresentSpec {
            max_lines: 200,
            max_bytes: 16,
            overflow_dir: dir.path().to_path_buf(),
        };
        let tools = adapt_client_as_tools(one_tool_client(), "mcp__web", spec)
            .await
            .unwrap();
        let out = tools[0].invoke(json!({"q": "rust"})).await.unwrap();
        let spilled = dir.path().join("mcp-test-1.txt");
        assert_eq!(out["truncated"], true);
        assert_eq!(out["overflow_file"], json!(spilled.to_string_lossy()));
        let output = out["output"].as_str().unwrap();
        assert!(
            output.starts_with("called search wi\n--- output truncated"),
            "{output}"
        );
        assert_eq!(
            std::fs::read_to_string(spilled).unwrap(),
            r#"called search with {"q":"rust"}"#
        );
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
            assert_eq!(tool_result_to_json(result, 42, &unlimited()), expected);
        }
    }

    #[tokio::test]
    async fn adapter_turns_client_errors_into_error_envelopes() {
        let err = || McpError::RpcError {
            code: -32602,
            message: "missing field 'query'".into(),
            data: None,
        };
        let tool = failing_tool(err(), Duration::ZERO).await;
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
        let tool = failing_tool(McpError::ServerDown, Duration::from_millis(20)).await;
        let out = tool.invoke(json!({})).await.unwrap();
        let dur = out["duration_ms"].as_u64().expect("duration_ms u64");
        assert!(
            dur >= 15,
            "duration must include the upstream sleep: got {dur}ms"
        );
    }
}
