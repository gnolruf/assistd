//! MCP (Model Context Protocol) client: each server, a child process or a Streamable HTTP URL,
//! is managed by [`McpServer`], and each tool it exposes becomes a [`Tool`] via
//! [`adapt_client_as_tools`].

use std::borrow::Cow;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use assistd_tools::attachment::MAX_IMAGE_BYTES;
use assistd_tools::{
    ApprovalGate, Attachment, ConfirmSource, ConfirmationRequest, ImageMime, JudgeContext,
    JudgedClass, Risk, Tool, ToolError, ToolOutput, Unattended, VisionGate,
};
use assistd_utils::text::human_size;
use async_trait::async_trait;
use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use rmcp::model::{CallToolResult, ContentBlock, EmbeddedResource, Resource, ResourceContents};
use serde_json::Value;

mod admission;
mod error;
mod launch;
mod oauth;
mod server;
mod session;
mod stdio;
mod streamable;

pub use error::{McpError, mcp_error_line};
pub use launch::{LaunchTarget, launch, withdraw_tools};
pub use oauth::{
    LOGIN_TIMEOUT, OAuthConfig, PendingLogin, Revocation, TokenStore, begin_login, sign_out,
};
pub use server::McpServer;
pub use session::Connector;
pub use stdio::StdioConfig;
pub use streamable::{HttpAuth, HttpConfig};

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
                unattended: Unattended::Judged(JudgeContext {
                    class: JudgedClass::McpTool,
                    cwd: None,
                    why: format!(
                        "it calls the MCP tool `{name}`, which is not yet approved, with these \
                         JSON arguments"
                    ),
                }),
                source: ConfirmSource::Main,
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

/// Render `result`'s content blocks as one output, joined by newlines, falling back to
/// `structuredContent`. Images become attachments only with `vision`, else a note; an
/// `isError` result is prefixed as a whole.
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
/// image and `vision` is on. Block types without a rendering are passed on as JSON.
fn render_block(block: ContentBlock, vision: bool) -> (String, Option<Attachment>) {
    match block {
        ContentBlock::Text(text) => (text.text, None),
        ContentBlock::Image(image) => render_image(&image.mime_type, &image.data, vision),
        ContentBlock::ResourceLink(link) => (resource_link_line(&link), None),
        ContentBlock::Resource(embedded) => render_embedded_resource(embedded, vision),
        other => (
            serde_json::to_value(other).unwrap_or_default().to_string(),
            None,
        ),
    }
}

/// A note naming the image, plus the decoded image when `vision` is on and it can be shown.
fn render_image(mime: &str, base64: &str, vision: bool) -> (String, Option<Attachment>) {
    if !vision {
        return (
            format!("(image omitted: {mime}; model has no vision)"),
            None,
        );
    }
    match image_attachment(mime, base64) {
        Ok(attachment) => (format!("(image: {mime})"), Some(attachment)),
        Err(why) => (format!("(image omitted: {mime}; {why})"), None),
    }
}

/// `link` on one line: its name, URI, and MIME type and size when the server gave them.
fn resource_link_line(link: &Resource) -> String {
    let mut line = format!("(resource link: {}, {}", link.name, link.uri);
    for field in [link.mime_type.clone(), link.size.map(human_size)]
        .into_iter()
        .flatten()
    {
        line.push_str(", ");
        line.push_str(&field);
    }
    line.push(')');
    line
}

/// A text resource as a header line naming it followed by its text verbatim; a blob as an
/// image when its MIME type is a supported image format, else a note with its size. Kinds
/// without a rendering are passed on as JSON.
fn render_embedded_resource(
    embedded: EmbeddedResource,
    vision: bool,
) -> (String, Option<Attachment>) {
    match embedded.resource {
        ResourceContents::TextResourceContents {
            uri,
            mime_type,
            text,
            ..
        } => {
            let header = match mime_type {
                Some(mime) => format!("(resource: {uri}, {mime})"),
                None => format!("(resource: {uri})"),
            };
            (format!("{header}\n{text}"), None)
        }
        ResourceContents::BlobResourceContents {
            mime_type, blob, ..
        } => match mime_type {
            Some(mime) if ImageMime::from_mime(&mime).is_some() => {
                render_image(&mime, &blob, vision)
            }
            mime => (
                format!(
                    "(binary resource omitted: {}, {})",
                    mime.as_deref().unwrap_or("unknown type"),
                    human_size(decoded_len(&blob))
                ),
                None,
            ),
        },
        other => (
            serde_json::to_value(other).unwrap_or_default().to_string(),
            None,
        ),
    }
}

/// Decode a base64 image of type `mime`, or say why it cannot be shown to the model: an
/// unsupported format, invalid base64, or more than [`MAX_IMAGE_BYTES`].
fn image_attachment(mime: &str, base64: &str) -> Result<Attachment, String> {
    let mime = ImageMime::from_mime(mime).ok_or_else(|| "not PNG, JPEG or WebP".to_string())?;
    if decoded_len(base64) > MAX_IMAGE_BYTES {
        return Err(format!("larger than {MAX_IMAGE_BYTES} bytes"));
    }
    let bytes = B64
        .decode(base64)
        .map_err(|e| format!("invalid base64: {e}"))?;
    Ok(Attachment::Image { mime, bytes })
}

/// Bytes well-formed padded `base64` decodes to.
fn decoded_len(base64: &str) -> u64 {
    let padding = base64
        .bytes()
        .rev()
        .take_while(|&byte| byte == b'=')
        .count();
    (base64.len() / 4 * 3).saturating_sub(padding) as u64
}

#[cfg(test)]
mod tests;
