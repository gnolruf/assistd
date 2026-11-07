//! MCP errors and their rendering as model-facing tool error lines.

use std::io::ErrorKind;
use std::time::Duration;

use rmcp::service::{ClientInitializeError, ServiceError};
use thiserror::Error;

use crate::streamable::transport_failure;

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

    /// The HTTP request never got an answer: DNS, connect, TLS, or a refused redirect.
    #[error("could not reach the MCP server: {0}")]
    Connection(String),

    #[error("MCP server answered HTTP {0}")]
    HttpStatus(u16),

    /// The server refused the credentials sent, or none are stored; `scopes` are the ones a
    /// 403 `insufficient_scope` asked for.
    #[error("MCP server needs authorization{}", scope_list(scopes))]
    NeedsAuthorization { scopes: Vec<String> },

    /// No `client_id` is configured and the authorization server offers no dynamic client
    /// registration.
    #[error("MCP server's authorization server does not offer dynamic client registration")]
    RegistrationUnsupported,

    /// An OAuth step failed for a reason worth retrying. Never carries a token, code or secret.
    #[error("MCP OAuth failed: {0}")]
    OAuth(String),

    /// The server no longer knows the session; the next call opens a new one.
    #[error("MCP server ended the session")]
    SessionExpired,
}

impl McpError {
    /// Whether starting the server again unchanged would fail the same way: its command is
    /// missing or cannot be executed, or it refuses the configured credentials.
    pub(crate) fn is_permanent(&self) -> bool {
        match self {
            Self::Spawn { source, .. } => {
                matches!(
                    source.kind(),
                    ErrorKind::NotFound | ErrorKind::PermissionDenied
                )
            }
            Self::NeedsAuthorization { .. } | Self::RegistrationUnsupported => true,
            _ => false,
        }
    }
}

impl From<ServiceError> for McpError {
    fn from(error: ServiceError) -> Self {
        match error {
            ServiceError::McpError(data) => Self::RpcError {
                code: data.code.0,
                message: data.message.into_owned(),
            },
            ServiceError::TransportSend(error) => {
                transport_failure(&error).unwrap_or(Self::TransportClosed)
            }
            ServiceError::TransportClosed => Self::TransportClosed,
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
        McpError::Connection(detail) => format!(
            "[error] {tool_name}: could not reach the MCP server: {detail}. \
             Try: another tool; the server may be offline\n"
        ),
        McpError::HttpStatus(status) => format!(
            "[error] {tool_name}: MCP server answered HTTP {status}. \
             Try: the call again later\n"
        ),
        McpError::NeedsAuthorization { scopes } => format!(
            "[error] {tool_name}: MCP server needs authorization{}. \
             Check: the server's `headers` or `auth` in config.toml\n",
            scope_list(scopes)
        ),
        McpError::RegistrationUnsupported => format!(
            "[error] {tool_name}: MCP server's authorization server does not offer dynamic \
             client registration. Check: set the server's `client_id` in config.toml\n"
        ),
        McpError::OAuth(detail) => {
            format!("[error] {tool_name}: MCP OAuth failed: {detail}. Try: the call again later\n")
        }
        McpError::SessionExpired => format!(
            "[error] {tool_name}: MCP server ended the session. \
             Try: the call again\n"
        ),
    }
}

/// ` (scopes: a b)`, or nothing when `scopes` is empty.
fn scope_list(scopes: &[String]) -> String {
    match scopes {
        [] => String::new(),
        scopes => format!(" (scopes: {})", scopes.join(" ")),
    }
}

#[cfg(test)]
mod tests {
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
                    source: std::io::Error::new(ErrorKind::NotFound, "missing"),
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
            ("connection", McpError::Connection("refused".into())),
            ("http_status", McpError::HttpStatus(502)),
            (
                "authorization",
                McpError::NeedsAuthorization {
                    scopes: vec!["drive.readonly".into()],
                },
            ),
            ("registration", McpError::RegistrationUnsupported),
            ("oauth", McpError::OAuth("token store: unreadable".into())),
            ("session_expired", McpError::SessionExpired),
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
}
