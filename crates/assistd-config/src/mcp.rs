//! MCP (Model Context Protocol) client configuration.

use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::num::NonZeroU64;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::defaults::{DEFAULT_MCP_ENABLED, DEFAULT_MCP_REQUEST_TIMEOUT_SECS};

/// Model Context Protocol client settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct McpConfig {
    /// When `false`, no server in `servers` is connected.
    pub enabled: bool,
    pub servers: Vec<McpServerConfig>,
}

impl Default for McpConfig {
    fn default() -> Self {
        Self {
            enabled: DEFAULT_MCP_ENABLED,
            servers: Vec::new(),
        }
    }
}

/// One `[[mcp.servers]]` entry: a server run as a child process and
/// spoken to with newline-delimited JSON-RPC over its stdin/stdout.
/// `Debug` lists env var names, never their values.
#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct McpServerConfig {
    /// Label in tool names (`mcp__<name>__<tool>`) and logs. Must be
    /// unique, use only ASCII letters, digits, `_` or `-`, and contain
    /// no `__` and no trailing `_`, so no two servers' tools share a name.
    pub name: String,
    /// Command to spawn; a bare name is resolved via `$PATH`.
    pub command: PathBuf,
    #[serde(default)]
    pub args: Vec<String>,
    /// Environment variables for the child, which inherits from the daemon
    /// only `HOME`, `LANG`, `LANGUAGE`, `LC_*`, `LOGNAME`, `PATH`, `SHELL`,
    /// `TERM`, `TMPDIR`, `TZ` and `USER`. Set credentials here, and keep the
    /// config file mode `0600`.
    #[serde(default)]
    pub env: HashMap<String, String>,
    /// Timeout in seconds for the initialize handshake and each request.
    #[serde(default = "default_request_timeout_secs")]
    pub request_timeout_secs: NonZeroU64,
}

impl fmt::Debug for McpServerConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("McpServerConfig")
            .field("name", &self.name)
            .field("command", &self.command)
            .field("args", &self.args)
            .field("env_names", &self.env.keys().collect::<BTreeSet<_>>())
            .field("request_timeout_secs", &self.request_timeout_secs)
            .finish()
    }
}

fn default_request_timeout_secs() -> NonZeroU64 {
    DEFAULT_MCP_REQUEST_TIMEOUT_SECS
}

#[cfg(test)]
mod tests;
