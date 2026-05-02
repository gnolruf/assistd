use thiserror::Error;

/// Errors surfaced by the MCP transport layer and the per-server supervisor.
///
/// `RpcError` carries the JSON-RPC server-side error verbatim — code,
/// message, and optional data — so callers can distinguish "the server
/// processed our request and returned a structured error" from "the
/// transport itself broke." `TransportClosed` and `RequestTimeout` are
/// the two terminal failure modes the supervisor watches for.
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
