//! MCP (Model Context Protocol) client configuration.

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use url::Url;

use crate::defaults::{DEFAULT_MCP_ENABLED, DEFAULT_MCP_REQUEST_TIMEOUT_SECS};

/// Model Context Protocol client settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
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

/// One `[[mcp.servers]]` entry, discriminated by `transport`. A key of the
/// other transport (`url` on stdio, `env` on SSE) is a parse error.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "transport", rename_all = "lowercase", deny_unknown_fields)]
pub enum McpServerConfig {
    /// Newline-delimited JSON-RPC over a child process's stdin/stdout.
    Stdio {
        /// Label in tool names (`mcp__<name>__<tool>`) and logs. Must be
        /// unique and use only ASCII letters, digits, `_` or `-`.
        name: String,
        /// Command to spawn; a bare name is resolved via `$PATH`.
        command: PathBuf,
        #[serde(default)]
        args: Vec<String>,
        /// Extra environment variables for the child.
        #[serde(default)]
        env: HashMap<String, String>,
        /// Per-request JSON-RPC timeout in seconds.
        #[serde(default = "default_request_timeout_secs")]
        request_timeout_secs: NonZeroU64,
    },
    /// HTTP Server-Sent Events endpoint.
    Sse {
        /// As for [`Self::Stdio`].
        name: String,
        /// Endpoint URL, validated at load.
        url: Url,
        /// Extra HTTP headers for every request.
        #[serde(default)]
        headers: HashMap<String, String>,
        /// Per-request JSON-RPC timeout in seconds.
        #[serde(default = "default_request_timeout_secs")]
        request_timeout_secs: NonZeroU64,
    },
}

impl McpServerConfig {
    /// The server's label, whatever its transport.
    pub fn name(&self) -> &str {
        match self {
            Self::Stdio { name, .. } | Self::Sse { name, .. } => name,
        }
    }

    /// The server's per-request JSON-RPC timeout, whatever its transport.
    pub fn request_timeout_secs(&self) -> NonZeroU64 {
        match self {
            Self::Stdio {
                request_timeout_secs,
                ..
            }
            | Self::Sse {
                request_timeout_secs,
                ..
            } => *request_timeout_secs,
        }
    }

    /// Lowercase transport name, for logs and error messages.
    pub fn transport(&self) -> &'static str {
        match self {
            Self::Stdio { .. } => "stdio",
            Self::Sse { .. } => "sse",
        }
    }
}

fn default_request_timeout_secs() -> NonZeroU64 {
    DEFAULT_MCP_REQUEST_TIMEOUT_SECS
}

#[cfg(test)]
mod tests;
