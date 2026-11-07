//! Wire types for signing MCP servers in and out and reporting their state.

use serde::{Deserialize, Serialize};

/// How the daemon reaches an MCP server.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum McpTransportKind {
    /// A child process spoken to over stdin and stdout.
    Stdio,
    /// A remote server spoken to over Streamable HTTP.
    Http,
}

/// Where one MCP server stands.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum McpServerState {
    Starting,
    /// Its tools are offered to the model.
    Ready,
    /// It has no usable OAuth token; `scopes` are the ones it asked for, if any.
    NeedsAuthorization {
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        scopes: Vec<String>,
    },
    /// Failed to start; holds why.
    Unavailable {
        reason: String,
    },
}

/// What signing a server out did to its token at the authorization server; the stored token
/// is deleted in every case.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum TokenRevocation {
    Revoked,
    /// The authorization server offers no revocation.
    NotOffered,
    /// No token was stored.
    NothingStored,
    Failed {
        reason: String,
    },
}
