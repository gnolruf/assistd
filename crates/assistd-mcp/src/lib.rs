//! MCP (Model Context Protocol) client. Each server runs as a child
//! process managed by [`McpServer`], and each tool it exposes becomes a
//! [`Tool`] via [`adapt_client_as_tools`].

use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use assistd_tools::attachment::MAX_IMAGE_BYTES;
use assistd_tools::presentation::{PresentSpec, TextTruncator, TruncatedText};
use assistd_tools::{
    ApprovalGate, ConfirmationRequest, MCP_TOOL_NAME_PREFIX, Tool, ToolError, VisionGate,
};
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
/// first until the tool is approved for good; image results are dropped
/// while `vision` reports no support.
#[derive(Debug)]
struct McpToolAdapter {
    client: Arc<dyn McpClient>,
    tool: rmcp::model::Tool,
    registry_name: String,
    truncator: Arc<TextTruncator>,
    approvals: ApprovalGate,
    vision: Arc<VisionGate>,
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
            Ok(result) => Ok(tool_result_to_json(
                result,
                duration_ms,
                &self.truncator,
                self.vision.supported(),
            )),
            Err(err) => Ok(error_envelope(&self.registry_name, &err, duration_ms)),
        }
    }
}

/// [`Tool`] entries for every tool `client` exposes, named
/// `mcp__<server_name>__<tool>`. Results past `output`'s caps are cut, with
/// the overflow spilled as `mcp-<server_name>-<n>.txt`. Each call asks
/// first unless `approvals` holds the tool. Image results are replaced by a
/// text note whenever `vision` reports the model cannot take images.
pub async fn adapt_client_as_tools(
    client: Arc<dyn McpClient>,
    server_name: &str,
    output: PresentSpec,
    approvals: &ApprovalGate,
    vision: &Arc<VisionGate>,
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
                vision: vision.clone(),
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

/// Render every content block of `result` as one tool-result envelope.
/// Blocks are joined by newlines (non-text ones as JSON) and cut by
/// `truncator`; with `vision`, images also go into `attachments[]` as is,
/// and without it they become a note only. Without content,
/// `structuredContent` is the body. An `isError` result is prefixed as a whole.
fn tool_result_to_json(
    result: CallToolResult,
    duration_ms: u128,
    truncator: &TextTruncator,
    vision: bool,
) -> Value {
    let (sections, attachments): (Vec<String>, Vec<Option<Value>>) = result
        .content
        .into_iter()
        .map(|block| render_block(block, vision))
        .unzip();
    let body = match result.structured_content {
        Some(structured) if sections.is_empty() => structured.to_string(),
        _ => sections.join("\n"),
    };
    let body = match result.is_error {
        Some(true) => format!("[mcp tool error] {body}"),
        _ => body,
    };
    let mut envelope = text_envelope(truncator.truncate(body), duration_ms);
    let attachments: Vec<Value> = attachments.into_iter().flatten().collect();
    if !attachments.is_empty() {
        envelope["attachments"] = Value::Array(attachments);
    }
    envelope
}

/// The text one content block contributes to the output, plus its
/// attachment if it is an image, `vision` is on, and it decodes to at
/// most [`MAX_IMAGE_BYTES`].
fn render_block(block: ContentBlock, vision: bool) -> (String, Option<Value>) {
    match block {
        ContentBlock::Text(text) => (text.text, None),
        ContentBlock::Image(image) if !vision => (
            format!("(image omitted: {}; model has no vision)", image.mime_type),
            None,
        ),
        ContentBlock::Image(image) if decoded_len(&image.data) > MAX_IMAGE_BYTES => (
            format!(
                "(image omitted: {}; larger than {MAX_IMAGE_BYTES} bytes)",
                image.mime_type
            ),
            None,
        ),
        ContentBlock::Image(image) => (
            format!("(image: {})", image.mime_type),
            Some(json!({"type": "image", "mime": image.mime_type, "data": image.data})),
        ),
        other => (
            serde_json::to_value(other).unwrap_or_default().to_string(),
            None,
        ),
    }
}

/// Bytes `base64` decodes to, give or take padding.
fn decoded_len(base64: &str) -> u64 {
    base64.len() as u64 / 4 * 3
}

fn text_envelope(cut: TruncatedText, duration_ms: u128) -> Value {
    let mut envelope = json!({
        "type": "text",
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
