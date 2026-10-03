use std::time::Duration;

use rmcp::service::{ClientInitializeError, ServiceError};
use thiserror::Error;

/// Errors from an MCP server or its supervisor. [`mcp_error_line`]
/// renders each variant for the model.
#[derive(Debug, Error)]
pub enum McpError {
    #[error("failed to spawn MCP server `{path}`: {source}")]
    Spawn {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("MCP initialize failed: {0}")]
    Initialize(#[source] Box<ClientInitializeError>),

    #[error("MCP transport closed")]
    TransportClosed,

    #[error("MCP request timed out after {0:?}")]
    RequestTimeout(Duration),

    #[error("MCP server reported an error (code {code}): {message}")]
    RpcError { code: i32, message: String },

    /// The server's answer was not one the client could use.
    #[error("MCP protocol error: {0}")]
    Protocol(String),

    #[error("MCP tool arguments must be a JSON object")]
    ArgumentsNotObject,

    #[error("MCP server is currently unavailable")]
    ServerDown,
}

impl From<ServiceError> for McpError {
    fn from(error: ServiceError) -> Self {
        match error {
            ServiceError::McpError(data) => Self::RpcError {
                code: data.code.0,
                message: data.message.into_owned(),
            },
            ServiceError::TransportClosed | ServiceError::TransportSend(_) => Self::TransportClosed,
            ServiceError::Timeout { timeout } => Self::RequestTimeout(timeout),
            other => Self::Protocol(other.to_string()),
        }
    }
}

/// Render `err` as the `[error] <tool_name>: <what>. <Hint>: <recovery>\n`
/// line native tool failures use.
pub fn mcp_error_line(tool_name: &str, err: &McpError) -> String {
    match err {
        McpError::Spawn { path, source } => format!(
            "[error] {tool_name}: failed to spawn MCP server `{path}`: {source}. \
             Check: the server command/args in config.toml\n"
        ),
        McpError::Initialize(source) => format!(
            "[error] {tool_name}: MCP initialize failed: {source}. \
             Check: daemon logs for the server's startup output\n"
        ),
        McpError::TransportClosed => format!(
            "[error] {tool_name}: MCP transport closed. \
             Try: another tool while the server restarts\n"
        ),
        McpError::RequestTimeout(timeout) => format!(
            "[error] {tool_name}: MCP request timed out after {timeout:?}. \
             Try: the call again or a smaller request\n"
        ),
        McpError::RpcError { code, message } => format!(
            "[error] {tool_name}: MCP server returned error code {code}: {message}. \
             Check: the arguments and try again\n"
        ),
        McpError::Protocol(detail) => format!(
            "[error] {tool_name}: MCP protocol error: {detail}. \
             Check: daemon logs for malformed responses\n"
        ),
        McpError::ArgumentsNotObject => format!(
            "[error] {tool_name}: MCP tool arguments must be a JSON object. \
             Use: an object matching the tool's parameters schema\n"
        ),
        McpError::ServerDown => format!(
            "[error] {tool_name}: MCP server is currently unavailable. \
             Try: another tool while the server restarts\n"
        ),
    }
}

#[cfg(test)]
mod tests {
    use rmcp::model::{ErrorCode, ErrorData};

    use super::*;

    fn contains_hint(s: &str) -> bool {
        s.contains("Use:") || s.contains("Try:") || s.contains("Check:") || s.contains("Available:")
    }

    #[test]
    fn every_variant_emits_convention_compliant_line() {
        let cases: Vec<(&str, McpError)> = vec![
            (
                "spawn",
                McpError::Spawn {
                    path: "/usr/bin/fake".into(),
                    source: std::io::Error::new(std::io::ErrorKind::NotFound, "missing"),
                },
            ),
            (
                "initialize",
                McpError::Initialize(Box::new(ClientInitializeError::ConnectionClosed(
                    "eof".into(),
                ))),
            ),
            ("transport_closed", McpError::TransportClosed),
            ("timeout", McpError::RequestTimeout(Duration::from_secs(30))),
            (
                "rpc",
                McpError::RpcError {
                    code: -32602,
                    message: "Invalid params".into(),
                },
            ),
            ("protocol", McpError::Protocol("bad frame".into())),
            ("arguments", McpError::ArgumentsNotObject),
            ("server_down", McpError::ServerDown),
        ];
        for (label, err) in cases {
            let line = mcp_error_line("mcp__web__search", &err);
            assert!(
                line.starts_with("[error] mcp__web__search: "),
                "{label}: missing `[error] <tool>: ` prefix, got {line:?}"
            );
            assert!(
                contains_hint(&line),
                "{label}: missing recovery hint (Use/Try/Check/Available), got {line:?}"
            );
            assert!(line.ends_with('\n'), "{label}: missing trailing newline");
        }
    }

    #[test]
    fn lines_carry_the_variant_details_and_matching_hint() {
        let cases = [
            (
                McpError::RpcError {
                    code: -32602,
                    message: "Invalid params: missing `query`".into(),
                },
                "[error] mcp__web__search: MCP server returned error code -32602: \
                 Invalid params: missing `query`. Check: the arguments and try again\n",
            ),
            (
                McpError::RequestTimeout(Duration::from_secs(30)),
                "[error] mcp__web__search: MCP request timed out after 30s. \
                 Try: the call again or a smaller request\n",
            ),
            (
                McpError::ServerDown,
                "[error] mcp__web__search: MCP server is currently unavailable. \
                 Try: another tool while the server restarts\n",
            ),
        ];
        for (err, expected) in cases {
            assert_eq!(mcp_error_line("mcp__web__search", &err), expected);
        }
    }

    #[test]
    fn service_errors_map_to_the_matching_variant() {
        let rpc = McpError::from(ServiceError::McpError(ErrorData::new(
            ErrorCode(-32601),
            "method not found",
            None,
        )));
        assert!(
            matches!(&rpc, McpError::RpcError { code: -32601, message } if message == "method not found"),
            "{rpc:?}"
        );
        let closed = McpError::from(ServiceError::TransportClosed);
        assert!(matches!(closed, McpError::TransportClosed), "{closed:?}");
        let timeout = McpError::from(ServiceError::Timeout {
            timeout: Duration::from_secs(3),
        });
        assert!(
            matches!(timeout, McpError::RequestTimeout(after) if after == Duration::from_secs(3)),
            "{timeout:?}"
        );
    }
}
