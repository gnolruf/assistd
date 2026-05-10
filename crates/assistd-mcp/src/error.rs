use thiserror::Error;

/// Errors surfaced by the MCP transport layer and the per-server supervisor.
///
/// `RpcError` carries the JSON-RPC server-side error verbatim (code,
/// message, and optional data) so callers can distinguish "the server
/// processed our request and returned a structured error" from "the
/// transport itself broke." `TransportClosed` and `RequestTimeout` are
/// the two terminal failure modes the supervisor watches for.
///
/// # Translating to the agent-facing error convention
///
/// Native tool errors follow `[error] <cmd>: <what>. <Hint>: <recovery>` so
/// the LLM gets a one-step recovery path on every failure (the convention
/// is documented in `assistd-tools/src/command.rs`). MCP tool failures must
/// honor the same shape so the model treats them like any other tool error.
/// [`mcp_error_line`] does that translation per variant; call it from any
/// site that surfaces an `McpError` to the model.
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

    /// Genuine wire-protocol violations (malformed JSON-RPC frames,
    /// missing required fields in MCP responses, base64 decode of an
    /// image attachment failed, …). Distinct from [`Self::Config`]:
    /// `Protocol` means the server diverged from the spec, `Config`
    /// means our local config is wrong.
    #[error("MCP protocol error: {0}")]
    Protocol(String),

    /// User-config issue surfaced at connect time: bad URL, bad
    /// header name/value, etc. Carries a context string and the
    /// original parse error so `error.source()` walks the chain. Routed
    /// to a "Check: <config>" recovery hint by [`mcp_error_line`] so
    /// the operator (and the model) sees the actionable next step.
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

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    /// Wraps `reqwest::Error` directly so the source chain survives
    /// for callers that downcast or walk `error.source()`. The Display
    /// impl of `reqwest::Error` already gives a useful one-liner.
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
}

impl McpError {
    /// Build a [`Self::Config`] from a context string and a parse
    /// error. Use for header/URL parse failures at connect time so the
    /// operator sees a "Check: config.toml" recovery hint instead of a
    /// generic "protocol error" line.
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

/// Translate an [`McpError`] into the daemon's `[error] <cmd>: <what>.
/// <Hint>: <recovery>\n` line shape so the model sees an MCP failure with
/// the same recovery affordances as a native tool failure.
///
/// `tool_name` is the registry-facing identifier (typically
/// `mcp__<server>__<tool>`); the LLM is already addressing the tool by
/// that name, so echoing it back keeps the message anchored.
///
/// The `<Hint>` keyword is one of `Use:` / `Try:` / `Check:` / `Available:`
/// per the convention in `assistd-tools/src/command.rs:11-32`. The recovery
/// clause is a concrete next step the model can take (retry, switch tools,
/// or check daemon-side state) chosen to match each variant's failure mode.
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
        McpError::Io(e) => format!(
            "[error] {tool_name}: MCP I/O error: {e}. \
             Check: daemon logs for transport details\n"
        ),
        McpError::Json(e) => format!(
            "[error] {tool_name}: MCP JSON error: {e}. \
             Check: daemon logs for transport details\n"
        ),
        McpError::Http(m) => format!(
            "[error] {tool_name}: MCP HTTP error: {m}. \
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
            ("io", McpError::Io(std::io::Error::other("pipe broke"))),
            ("http", McpError::Http(fake_http_error().await)),
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
