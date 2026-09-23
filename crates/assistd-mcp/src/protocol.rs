//! The MCP request and response shapes both transports share.

use std::time::Duration;

use base64::Engine;
use serde_json::{Value, json};
use tokio::sync::oneshot;
use tracing::warn;

use crate::error::McpError;
use crate::jsonrpc::{Reply, RpcError};
use crate::{ToolResult, ToolSchema};

const PROTOCOL_VERSION: &str = "2024-11-05";
const CLIENT_NAME: &str = "assistd";
const CLIENT_VERSION: &str = env!("CARGO_PKG_VERSION");

pub(crate) fn initialize_params() -> Value {
    json!({
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": { "tools": {} },
        "clientInfo": { "name": CLIENT_NAME, "version": CLIENT_VERSION },
    })
}

pub(crate) fn warn_on_version_mismatch(label: &str, initialize_result: &Value) {
    if let Some(server_pv) = initialize_result
        .get("protocolVersion")
        .and_then(Value::as_str)
        && server_pv != PROTOCOL_VERSION
    {
        warn!(
            target: "assistd::mcp",
            server = %label,
            client_version = PROTOCOL_VERSION,
            server_version = server_pv,
            "MCP protocol version mismatch (continuing optimistically)",
        );
    }
}

pub(crate) fn tool_call_params(name: &str, arguments: Value) -> Value {
    json!({ "name": name, "arguments": arguments })
}

pub(crate) fn parse_tools_list(result: &Value) -> Result<Vec<ToolSchema>, McpError> {
    let tools = result
        .get("tools")
        .and_then(Value::as_array)
        .ok_or_else(|| McpError::Protocol("tools/list missing `tools` array".into()))?;
    tools
        .iter()
        .map(|entry| {
            let name = entry
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| McpError::Protocol("tool entry missing `name`".into()))?
                .to_string();
            let description = entry
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let input_schema = entry
                .get("inputSchema")
                .cloned()
                .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
            Ok(ToolSchema {
                name,
                description,
                input_schema,
            })
        })
        .collect()
}

/// The first content entry of a `tools/call` result, with an
/// `isError` text result prefixed so the model can tell it apart.
pub(crate) fn parse_tool_call(result: Value) -> Result<ToolResult, McpError> {
    let is_error = result
        .get("isError")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let first = result
        .get("content")
        .and_then(Value::as_array)
        .and_then(|entries| entries.first().cloned());
    let parsed = match first {
        None => ToolResult::Text(String::new()),
        Some(entry) => parse_content_entry(entry)?,
    };
    Ok(match parsed {
        ToolResult::Text(t) if is_error => ToolResult::Text(format!("[mcp tool error] {t}")),
        other => other,
    })
}

fn parse_content_entry(entry: Value) -> Result<ToolResult, McpError> {
    match entry.get("type").and_then(Value::as_str).unwrap_or("") {
        "text" => {
            let text = entry
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            Ok(ToolResult::Text(text))
        }
        "image" => {
            let mime = entry
                .get("mimeType")
                .and_then(Value::as_str)
                .ok_or_else(|| McpError::Protocol("image content missing `mimeType`".into()))?
                .to_string();
            let data_b64 = entry
                .get("data")
                .and_then(Value::as_str)
                .ok_or_else(|| McpError::Protocol("image content missing `data`".into()))?;
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(data_b64)
                .map_err(|e| McpError::Protocol(format!("image base64 decode failed: {e}")))?;
            Ok(ToolResult::Image { mime, bytes })
        }
        _ => Ok(ToolResult::Json(entry)),
    }
}

/// Await a correlated reply within `timeout`, mapping a dropped sender
/// to `TransportClosed` and a server-side error to `RpcError`.
pub(crate) async fn await_reply(
    rx: &mut oneshot::Receiver<Reply>,
    timeout: Duration,
) -> Result<Value, McpError> {
    match tokio::time::timeout(timeout, rx).await {
        Ok(Ok(Ok(value))) => Ok(value),
        Ok(Ok(Err(RpcError {
            code,
            message,
            data,
        }))) => Err(McpError::RpcError {
            code,
            message,
            data,
        }),
        Ok(Err(_)) => Err(McpError::TransportClosed),
        Err(_) => Err(McpError::RequestTimeout(timeout)),
    }
}

pub(crate) fn closed_err() -> RpcError {
    RpcError {
        code: -32603,
        message: "MCP transport closed".into(),
        data: None,
    }
}
