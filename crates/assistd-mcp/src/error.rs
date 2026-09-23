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
    RequestTimeout(std::time::Duration),

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

/// Render `e` as the `[error] <tool_name>: <what>. <Hint>: <recovery>\n`
/// line native tool failures use.
pub fn mcp_error_line(tool_name: &str, e: &McpError) -> String {
    match e {
        McpError::Spawn { path, source } => format!(
            "[error] {tool_name}: failed to spawn MCP server `{path}`: {source}. \
             Check: the server command/args in config.toml\n"
        ),
        McpError::TransportClosed => format!(
            "[error] {tool_name}: MCP transport closed. \
             Try: another tool while the server reconnects\n"
        ),
        McpError::RequestTimeout(d) => format!(
            "[error] {tool_name}: MCP request timed out after {d:?}. \
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
        McpError::Protocol(m) => format!(
            "[error] {tool_name}: MCP protocol error: {m}. \
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
        McpError::Json(e) => format!(
            "[error] {tool_name}: MCP JSON error: {e}. \
             Check: daemon logs for transport details\n"
        ),
        McpError::Http(m) => format!(
            "[error] {tool_name}: MCP HTTP error: {m}. \
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

    /// Build a real `reqwest::Error` for the test; `reqwest` doesn't
    /// expose a public constructor, so we provoke one with a malformed
    /// URL request. Async because the only path that yields a
    /// `reqwest::Error` flows through `Client::execute`.
    async fn fake_http_error() -> reqwest::Error {
        // No proxy + a URL that fails connect-time parsing (well-formed
        // URL but unreachable scheme) is the most reliable way to get
        // an Error back without hitting the network.
        reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("client builds")
            .get("not-a-valid-url")
            .send()
            .await
            .expect_err("must error on malformed URL")
    }

    /// Every `McpError` variant must produce a convention-compliant line:
    /// starts with `[error] <tool>: `, includes one of the four hint words,
    /// and ends with a newline. Mirrors the gating test in
    /// `assistd-tools/src/command.rs::every_registered_command_emits_…`.
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
            (
                "timeout",
                McpError::RequestTimeout(std::time::Duration::from_secs(30)),
            ),
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
            ("http", McpError::Http(fake_http_error().await)),
            (
                "http_status",
                McpError::HttpStatus {
                    method: "tools/call",
                    status: reqwest::StatusCode::BAD_GATEWAY,
                },
            ),
        ];
        for (label, e) in cases {
            let line = mcp_error_line("mcp__web__search", &e);
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

    /// Source chain on `Http` must walk back to the underlying
    /// `reqwest::Error`. Regression for the previous `to_string()`
    /// shape that lost it.
    #[tokio::test]
    async fn http_variant_preserves_source_chain() {
        let original = fake_http_error().await;
        let original_text = original.to_string();
        let wrapped = McpError::Http(original);
        let source = std::error::Error::source(&wrapped).expect("source chain present");
        assert_eq!(
            source.to_string(),
            original_text,
            "source must be the original reqwest::Error"
        );
    }

    /// Config variant carries actionable context and source; the
    /// `mcp_error_line` rendering must include both.
    #[test]
    fn config_variant_renders_with_check_hint_and_context() {
        let inner = std::io::Error::new(std::io::ErrorKind::InvalidInput, "bad bytes");
        let e = McpError::config("invalid header `X-Bad`", inner);
        let line = mcp_error_line("mcp__web__search", &e);
        assert!(line.contains("invalid header `X-Bad`"), "{line}");
        assert!(line.contains("bad bytes"), "{line}");
        assert!(
            line.contains("Check: ~/.config/assistd/config.toml"),
            "{line}"
        );
    }

    #[test]
    fn rpc_error_line_carries_code_and_message() {
        let e = McpError::RpcError {
            code: -32602,
            message: "Invalid params: missing `query`".into(),
            data: None,
        };
        let line = mcp_error_line("mcp__web__search", &e);
        assert!(line.contains("-32602"), "{line}");
        assert!(line.contains("Invalid params"), "{line}");
        assert!(line.contains("Check:"), "{line}");
    }

    #[test]
    fn timeout_line_carries_duration() {
        let e = McpError::RequestTimeout(std::time::Duration::from_secs(30));
        let line = mcp_error_line("mcp__web__search", &e);
        assert!(line.contains("30s"), "duration must be visible: {line}");
        assert!(line.contains("Try:"), "{line}");
    }

    #[test]
    fn server_down_line_suggests_retry() {
        let line = mcp_error_line("mcp__web__search", &McpError::ServerDown);
        assert!(line.contains("Try:"), "{line}");
        assert!(line.contains("reconnect"), "{line}");
    }
}
