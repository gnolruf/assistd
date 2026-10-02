//! MCP (Model Context Protocol) client. A server is reached over stdio
//! or HTTP+SSE, supervised by [`McpServerHandle`], and each tool it
//! exposes becomes a [`Tool`] via [`adapt_handle_as_tools`].

use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use assistd_tools::presentation::{PresentSpec, TextTruncator, TruncatedText};
use assistd_tools::{ApprovalGate, ConfirmationRequest, Tool, ToolError};
use async_trait::async_trait;
use base64::Engine;
use serde_json::{Value, json};

pub mod error;
pub mod handle;
pub mod health_route;
pub mod jsonrpc;
mod protocol;
mod response_id;
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
/// server-native name stays in `schema.name`. Each call asks the user
/// first until the tool is approved for good.
#[derive(Debug)]
pub struct McpToolAdapter {
    client: Arc<dyn McpClient>,
    schema: ToolSchema,
    registry_name: String,
    truncator: Arc<TextTruncator>,
    approvals: ApprovalGate,
}

impl McpToolAdapter {
    /// Adapter that invokes `schema.name` on `client`, asking first unless
    /// `approvals` holds the tool, and cutting text and JSON results with
    /// `truncator` before they reach the model.
    pub fn new(
        client: Arc<dyn McpClient>,
        schema: ToolSchema,
        registry_name: String,
        truncator: Arc<TextTruncator>,
        approvals: ApprovalGate,
    ) -> Self {
        Self {
            client,
            schema,
            registry_name,
            truncator,
            approvals,
        }
    }

    /// Whether the user allows this call with `args`; "always" approves
    /// the tool for good.
    async fn confirmed(&self, args: &Value) -> bool {
        let name = self.registry_name.as_str();
        self.approvals
            .confirm(name, || ConfirmationRequest {
                tool: name.to_string(),
                script: format!("{args:#}"),
                matched_pattern: "calls an MCP tool that is not yet approved".to_string(),
                always_allow: vec![name.to_string()],
            })
            .await
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

    /// Always `Ok`: a declined or failed call becomes an error envelope.
    async fn invoke(&self, args: Value) -> Result<Value, ToolError> {
        if !self.confirmed(&args).await {
            return Ok(declined_envelope(&self.registry_name));
        }
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

/// The error envelope of a call the user declined.
fn declined_envelope(tool_name: &str) -> Value {
    json!({
        "type": "error",
        "output": format!(
            "[error] {tool_name}: call cancelled by user. Try: a different approach\n"
        ),
        "exit_code": -1,
        "duration_ms": 0,
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
/// `mcp-<server>-<n>.txt`. Each call asks first unless `approvals` holds
/// the tool.
pub async fn adapt_handle_as_tools(
    handle: &McpServerHandle,
    name_prefix: &str,
    output: PresentSpec,
    approvals: &ApprovalGate,
) -> Result<Vec<Box<dyn Tool>>, McpError> {
    let client = handle.client();
    let schemas = client.list_tools().await?;
    let health_rx = handle.watch_health();
    let server_name = handle.name.clone();
    let truncator = Arc::new(TextTruncator::new(output, format!("mcp-{server_name}")));

    let mut tools: Vec<Box<dyn Tool>> = Vec::with_capacity(schemas.len());
    for schema in schemas {
        let registry_name = registry_name(name_prefix, &schema.name);
        let adapter = McpToolAdapter::new(
            client.clone(),
            schema,
            registry_name,
            truncator.clone(),
            approvals.clone(),
        );
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
    approvals: &ApprovalGate,
) -> Result<Vec<Box<dyn Tool>>, McpError> {
    let schemas = client.list_tools().await?;
    let truncator = Arc::new(TextTruncator::new(output, "mcp-test"));
    Ok(schemas
        .into_iter()
        .map(|schema| {
            let name = registry_name(name_prefix, &schema.name);
            let adapter = McpToolAdapter::new(
                client.clone(),
                schema,
                name,
                truncator.clone(),
                approvals.clone(),
            );
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
mod tests;
