//! MCP (Model Context Protocol) client. Each server runs as a child
//! process managed by [`McpServer`], and each tool it exposes becomes a
//! [`Tool`] via [`adapt_client_as_tools`].

use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use assistd_tools::presentation::{PresentSpec, TextTruncator, TruncatedText};
use assistd_tools::{ApprovalGate, ConfirmationRequest, MCP_TOOL_NAME_PREFIX, Tool, ToolError};
use async_trait::async_trait;
use rmcp::model::{CallToolResult, ContentBlock};
use serde_json::{Value, json};

mod error;
mod server;
mod stdio;

pub use error::{McpError, mcp_error_line};
pub use server::McpServer;
pub use stdio::StdioConfig;

/// A connection to a single MCP server.
#[async_trait]
pub trait McpClient: fmt::Debug + Send + Sync + 'static {
    /// Every tool the server currently exposes. Safe to call concurrently.
    async fn list_tools(&self) -> Result<Vec<rmcp::model::Tool>, McpError>;

    /// Invoke the server-native tool `name` with `arguments`.
    async fn invoke(&self, name: &str, arguments: Value) -> Result<CallToolResult, McpError>;
}

/// Exposes one MCP tool as a [`Tool`] under `registry_name`; the
/// server-native name stays in `tool.name`. Each call asks the user
/// first until the tool is approved for good.
#[derive(Debug)]
struct McpToolAdapter {
    client: Arc<dyn McpClient>,
    tool: rmcp::model::Tool,
    registry_name: String,
    truncator: Arc<TextTruncator>,
    approvals: ApprovalGate,
}

impl McpToolAdapter {
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
        self.tool.description.as_deref().unwrap_or_default()
    }

    fn parameters_schema(&self) -> Value {
        self.tool.schema_as_json_value()
    }

    /// Always `Ok`: a declined or failed call becomes an error envelope.
    async fn invoke(&self, args: Value) -> Result<Value, ToolError> {
        if !self.confirmed(&args).await {
            return Ok(declined_envelope(&self.registry_name));
        }
        let start = Instant::now();
        let outcome = self.client.invoke(&self.tool.name, args).await;
        let duration_ms = start.elapsed().as_millis();
        match outcome {
            Ok(result) => Ok(tool_result_to_json(result, duration_ms, &self.truncator)),
            Err(err) => Ok(error_envelope(&self.registry_name, &err, duration_ms)),
        }
    }
}

/// [`Tool`] entries for every tool `client` exposes, named
/// `mcp__<server_name>__<tool>`. Results past `output`'s caps are cut, with
/// the overflow spilled as `mcp-<server_name>-<n>.txt`. Each call asks
/// first unless `approvals` holds the tool.
pub async fn adapt_client_as_tools(
    client: Arc<dyn McpClient>,
    server_name: &str,
    output: PresentSpec,
    approvals: &ApprovalGate,
) -> Result<Vec<Box<dyn Tool>>, McpError> {
    let tools = client.list_tools().await?;
    let truncator = Arc::new(TextTruncator::new(output, format!("mcp-{server_name}")));
    Ok(tools
        .into_iter()
        .map(|tool| {
            Box::new(McpToolAdapter {
                client: client.clone(),
                registry_name: format!("{MCP_TOOL_NAME_PREFIX}{server_name}__{}", tool.name),
                tool,
                truncator: truncator.clone(),
                approvals: approvals.clone(),
            }) as Box<dyn Tool>
        })
        .collect())
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

/// Render the first content block of `result` as the tool-result
/// envelope. Text and JSON bodies are cut by `truncator`, an `isError`
/// text is prefixed, and an image goes into `attachments[]` as is.
fn tool_result_to_json(
    result: CallToolResult,
    duration_ms: u128,
    truncator: &TextTruncator,
) -> Value {
    match result.content.into_iter().next() {
        None => text_envelope("text", truncator.truncate(String::new()), duration_ms),
        Some(ContentBlock::Text(text)) => {
            let text = match result.is_error {
                Some(true) => format!("[mcp tool error] {}", text.text),
                _ => text.text,
            };
            text_envelope("text", truncator.truncate(text), duration_ms)
        }
        Some(ContentBlock::Image(image)) => json!({
            "type": "image",
            "output": format!("(image: {})", image.mime_type),
            "exit_code": 0,
            "duration_ms": duration_ms,
            "truncated": false,
            "attachments": [
                {"type": "image", "mime": image.mime_type, "data": image.data}
            ],
        }),
        Some(other) => {
            let value = serde_json::to_value(other).unwrap_or_default();
            let mut envelope =
                text_envelope("json", truncator.truncate(value.to_string()), duration_ms);
            envelope["value"] = value;
            envelope
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

#[cfg(test)]
mod tests;
