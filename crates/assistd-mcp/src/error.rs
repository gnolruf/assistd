use std::time::Duration;

use thiserror::Error;

/// Errors from the MCP transports and the per-server supervisor.
/// [`mcp_error_line`] renders each variant for the model.
#[derive(Debug, Error)]
pub enum McpError {
    #[error("failed to spawn MCP server `{path}`: {source}")]
    Spawn {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("MCP transport closed")]
    TransportClosed,

    #[error("MCP request timed out after {0:?}")]
    RequestTimeout(Duration),

    #[error("MCP server reported an error (code {code}): {message}")]
    RpcError {
        code: i64,
        message: String,
        data: Option<serde_json::Value>,
    },

    /// The server diverged from the MCP spec.
    #[error("MCP protocol error: {0}")]
    Protocol(String),

    /// The local server config is unusable (bad URL, bad header).
    #[error("MCP config error: {context}: {source}")]
    Config {
        context: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync + 'static>,
    },

    #[error("MCP server is currently unavailable")]
    ServerDown,

    #[error("too many in-flight MCP requests (cap reached)")]
    TooManyInFlight,

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    /// A POST of `method` was answered with a non-success status.
    #[error("POST {method} failed: HTTP {status}")]
    HttpStatus {
        method: &'static str,
        status: reqwest::StatusCode,
    },
}

impl McpError {
    /// A [`McpError::Config`] wrapping `source` with `context`.
    pub fn config(
        context: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Config {
            context: context.into(),
            source: Box::new(source),
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
        McpError::TransportClosed => format!(
            "[error] {tool_name}: MCP transport closed. \
             Try: another tool while the server reconnects\n"
        ),
        McpError::RequestTimeout(timeout) => format!(
            "[error] {tool_name}: MCP request timed out after {timeout:?}. \
             Try: the call again or a smaller request\n"
        ),
        McpError::RpcError {
            code,
            message,
            data: _,
        } => format!(
            "[error] {tool_name}: MCP server returned error code {code}: {message}. \
             Check: the arguments and try again\n"
        ),
        McpError::Protocol(detail) => format!(
            "[error] {tool_name}: MCP protocol error: {detail}. \
             Check: daemon logs for malformed responses\n"
        ),
        McpError::Config { context, source } => format!(
            "[error] {tool_name}: MCP config error: {context}: {source}. \
             Check: ~/.config/assistd/config.toml `[[mcp.servers]]` block\n"
        ),
        McpError::ServerDown => format!(
            "[error] {tool_name}: MCP server is currently unavailable. \
             Try: another tool while the server reconnects\n"
        ),
        McpError::TooManyInFlight => format!(
            "[error] {tool_name}: too many in-flight MCP requests. \
             Try: the call again after pending requests drain\n"
        ),
        McpError::Json(source) => format!(
            "[error] {tool_name}: MCP JSON error: {source}. \
             Check: daemon logs for transport details\n"
        ),
        McpError::Http(source) => format!(
            "[error] {tool_name}: MCP HTTP error: {source}. \
             Check: daemon logs for transport details\n"
        ),
        McpError::HttpStatus { method, status } => format!(
            "[error] {tool_name}: MCP POST {method} failed: HTTP {status}. \
             Check: daemon logs for transport details\n"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contains_hint(s: &str) -> bool {
        s.contains("Use:") || s.contains("Try:") || s.contains("Check:") || s.contains("Available:")
    }

    /// `reqwest` has no public error constructor; a request to a
    /// malformed URL fails before touching the network.
    async fn fake_http_error() -> reqwest::Error {
        reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("client builds")
            .get("not-a-valid-url")
            .send()
            .await
            .expect_err("must error on malformed URL")
    }

    #[tokio::test]
    async fn every_variant_emits_convention_compliant_line() {
        let cases: Vec<(&str, McpError)> = vec![
            (
                "spawn",
                McpError::Spawn {
                    path: "/usr/bin/fake".into(),
                    source: std::io::Error::new(std::io::ErrorKind::NotFound, "missing"),
                },
            ),
            ("transport_closed", McpError::TransportClosed),
            ("timeout", McpError::RequestTimeout(Duration::from_secs(30))),
            (
                "rpc",
                McpError::RpcError {
                    code: -32602,
                    message: "Invalid params".into(),
                    data: None,
                },
            ),
            ("protocol", McpError::Protocol("bad frame".into())),
            (
                "config",
                McpError::config(
                    "invalid header `X-Bad`",
                    std::io::Error::new(std::io::ErrorKind::InvalidInput, "not ascii"),
                ),
            ),
            ("server_down", McpError::ServerDown),
            ("too_many", McpError::TooManyInFlight),
            (
                "json",
                McpError::Json(serde_json::from_str::<serde_json::Value>("{").unwrap_err()),
            ),
            ("http", McpError::Http(fake_http_error().await)),
            (
                "http_status",
                McpError::HttpStatus {
                    method: "tools/call",
                    status: reqwest::StatusCode::BAD_GATEWAY,
                },
            ),
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
                McpError::config(
                    "invalid header `X-Bad`",
                    std::io::Error::new(std::io::ErrorKind::InvalidInput, "bad bytes"),
                ),
                "[error] mcp__web__search: MCP config error: invalid header `X-Bad`: bad bytes. \
                 Check: ~/.config/assistd/config.toml `[[mcp.servers]]` block\n",
            ),
            (
                McpError::RpcError {
                    code: -32602,
                    message: "Invalid params: missing `query`".into(),
                    data: None,
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
                 Try: another tool while the server reconnects\n",
            ),
        ];
        for (err, expected) in cases {
            assert_eq!(mcp_error_line("mcp__web__search", &err), expected);
        }
    }
}
