use thiserror::Error;

/// Errors surfaced by the MCP transport layer and the per-server supervisor.
///
/// `RpcError` carries the JSON-RPC server-side error verbatim — code,
/// message, and optional data — so callers can distinguish "the server
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
/// [`mcp_error_line`] does that translation per variant — call it from any
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

    #[error("MCP protocol error: {0}")]
    Protocol(String),

    #[error("MCP server is currently unavailable")]
    ServerDown,

    #[error("too many in-flight MCP requests (cap reached)")]
    TooManyInFlight,

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[error("HTTP error: {0}")]
    Http(String),
}

impl McpError {
    pub fn http(e: impl std::fmt::Display) -> Self {
        Self::Http(e.to_string())
    }
}

/// Translate an [`McpError`] into the daemon's `[error] <cmd>: <what>.
/// <Hint>: <recovery>\n` line shape so the model sees an MCP failure with
/// the same recovery affordances as a native tool failure.
///
/// `tool_name` is the registry-facing identifier (typically
/// `mcp__<server>__<tool>`) — the LLM is already addressing the tool by
/// that name, so echoing it back keeps the message anchored.
///
/// The `<Hint>` keyword is one of `Use:` / `Try:` / `Check:` / `Available:`
/// per the convention in `assistd-tools/src/command.rs:11-32`. The recovery
/// clause is a concrete next step the model can take — retry, switch tools,
/// or check daemon-side state — chosen to match each variant's failure mode.
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

    /// Every `McpError` variant must produce a convention-compliant line:
    /// starts with `[error] <tool>: `, includes one of the four hint words,
    /// and ends with a newline. Mirrors the gating test in
    /// `assistd-tools/src/command.rs::every_registered_command_emits_…`.
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
            ("server_down", McpError::ServerDown),
            ("too_many", McpError::TooManyInFlight),
            ("io", McpError::Io(std::io::Error::other("pipe broke"))),
            ("http", McpError::Http("503".into())),
        ];
        for (label, e) in cases {
            let line = mcp_error_line("mcp__web__search", &e);
            assert!(
                line.starts_with("[error] mcp__web__search: "),
                "{label}: missing `[error] <tool>: ` prefix — got {line:?}"
            );
            assert!(
                contains_hint(&line),
                "{label}: missing recovery hint (Use/Try/Check/Available) — got {line:?}"
            );
            assert!(line.ends_with('\n'), "{label}: missing trailing newline");
        }
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
