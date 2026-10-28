//! MCP (Model Context Protocol) client: each server runs as a child process managed by
//! [`McpServer`], and each tool it exposes becomes a [`Tool`] via [`adapt_client_as_tools`].

use std::borrow::Cow;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use assistd_tools::attachment::MAX_IMAGE_BYTES;
use assistd_tools::{
    ApprovalGate, Attachment, ConfirmationRequest, ImageMime, Risk, Tool, ToolError, ToolOutput,
    VisionGate,
};
use async_trait::async_trait;
use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use rmcp::model::{CallToolResult, ContentBlock, ImageContent};
use serde_json::Value;

mod admission;
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

/// Exposes one MCP tool as a [`Tool`] under `registry_name`. Each call asks the user first
/// until the tool is approved for good; image results are dropped while `vision` reports no
/// support.
#[derive(Debug)]
struct McpToolAdapter {
    client: Arc<dyn McpClient>,
    tool: rmcp::model::Tool,
    registry_name: String,
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
                risk: Risk::Elevated,
            })
            .await
    }
}

#[async_trait]
impl Tool for McpToolAdapter {
    fn name(&self) -> &str {
        &self.registry_name
    }

    fn description(&self) -> Cow<'_, str> {
        self.tool.description.as_deref().unwrap_or_default().into()
    }

    fn parameters_schema(&self) -> Value {
        self.tool.schema_as_json_value()
    }

    /// Always `Ok`: a declined or failed call becomes a failed [`ToolOutput`].
    async fn invoke(&self, args: Value) -> Result<ToolOutput, ToolError> {
        if !self.confirmed(&args).await {
            return Ok(declined_output(&self.registry_name));
        }
        let start = Instant::now();
        let outcome = self.client.invoke(&self.tool.name, args).await;
        let duration = start.elapsed();
        match outcome {
            Ok(result) => Ok(tool_output(result, duration, self.vision.supported())),
            Err(err) => Ok(ToolOutput::failed(
                mcp_error_line(&self.registry_name, &err),
                duration,
            )),
        }
    }
}

/// [`Tool`] entries for every tool `client` exposes, named `mcp__<server_name>__<tool>`; tools
/// with an invalid or duplicate name or oversized description or schema are skipped. Calls ask
/// first unless `approvals` holds the tool; images become a text note when `vision` says no.
pub async fn adapt_client_as_tools(
    client: Arc<dyn McpClient>,
    server_name: &str,
    approvals: &ApprovalGate,
    vision: &Arc<VisionGate>,
) -> Result<Vec<Box<dyn Tool>>, McpError> {
    let tools = client.list_tools().await?;
    Ok(admission::admit_tools(server_name, tools)
        .into_iter()
        .map(|(registry_name, tool)| {
            Box::new(McpToolAdapter {
                client: client.clone(),
                registry_name,
                tool,
                approvals: approvals.clone(),
                vision: vision.clone(),
            }) as Box<dyn Tool>
        })
        .collect())
}

/// The result of a call the user declined.
fn declined_output(tool_name: &str) -> ToolOutput {
    ToolOutput::failed(
        format!("[error] {tool_name}: call cancelled by user. Try: a different approach\n"),
        Duration::ZERO,
    )
}

/// Render `result`'s content blocks as one output, joined by newlines (non-text as JSON),
/// falling back to `structuredContent`. Images become attachments only with `vision`, else a
/// note; an `isError` result is prefixed as a whole.
fn tool_output(result: CallToolResult, duration: Duration, vision: bool) -> ToolOutput {
    let (sections, attachments): (Vec<String>, Vec<Option<Attachment>>) = result
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
    ToolOutput {
        attachments: attachments.into_iter().flatten().collect(),
        ..ToolOutput::text(body, duration)
    }
}

/// The text one content block contributes to the output, plus its attachment if it is an
/// image and `vision` is on.
fn render_block(block: ContentBlock, vision: bool) -> (String, Option<Attachment>) {
    match block {
        ContentBlock::Text(text) => (text.text, None),
        ContentBlock::Image(image) if !vision => (
            format!("(image omitted: {}; model has no vision)", image.mime_type),
            None,
        ),
        ContentBlock::Image(image) => match image_attachment(&image) {
            Ok(attachment) => (format!("(image: {})", image.mime_type), Some(attachment)),
            Err(why) => (format!("(image omitted: {}; {why})", image.mime_type), None),
        },
        other => (
            serde_json::to_value(other).unwrap_or_default().to_string(),
            None,
        ),
    }
}

/// Decode `image`, or say why it cannot be shown to the model: an unsupported format, invalid
/// base64, or more than [`MAX_IMAGE_BYTES`].
fn image_attachment(image: &ImageContent) -> Result<Attachment, String> {
    let mime = ImageMime::from_mime(&image.mime_type)
        .ok_or_else(|| "not PNG, JPEG or WebP".to_string())?;
    if decoded_len(&image.data) > MAX_IMAGE_BYTES {
        return Err(format!("larger than {MAX_IMAGE_BYTES} bytes"));
    }
    let bytes = B64
        .decode(&image.data)
        .map_err(|e| format!("invalid base64: {e}"))?;
    Ok(Attachment::Image { mime, bytes })
}

/// Bytes `base64` decodes to, give or take padding.
fn decoded_len(base64: &str) -> u64 {
    base64.len() as u64 / 4 * 3
}

#[cfg(test)]
mod tests;
