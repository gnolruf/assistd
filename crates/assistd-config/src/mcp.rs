//! MCP (Model Context Protocol) client configuration.

use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::num::NonZeroU64;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use url::Url;

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

/// One `[[mcp.servers]]` entry, discriminated by `transport`. A key of the
/// other transport (`url` on stdio, `env` on SSE) is ignored like any unknown key.
/// `Debug` lists env var and header names, never their values.
#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "transport", rename_all = "lowercase")]
pub enum McpServerConfig {
    /// Newline-delimited JSON-RPC over a child process's stdin/stdout.
    Stdio {
        /// Label in tool names (`mcp__<name>__<tool>`) and logs. Must be
        /// unique, use only ASCII letters, digits, `_` or `-`, and contain
        /// no `__` and no trailing `_`, so no two servers' tools share a name.
        name: String,
        /// Command to spawn; a bare name is resolved via `$PATH`.
        command: PathBuf,
        #[serde(default)]
        args: Vec<String>,
        /// Environment variables for the child, which inherits from the daemon
        /// only `HOME`, `LANG`, `LANGUAGE`, `LC_*`, `LOGNAME`, `PATH`, `SHELL`,
        /// `TERM`, `TMPDIR`, `TZ` and `USER`. Set credentials here.
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

impl fmt::Debug for McpServerConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stdio {
                name,
                command,
                args,
                env,
                request_timeout_secs,
            } => f
                .debug_struct("Stdio")
                .field("name", name)
                .field("command", command)
                .field("args", args)
                .field("env_names", &env.keys().collect::<BTreeSet<_>>())
                .field("request_timeout_secs", request_timeout_secs)
                .finish(),
            Self::Sse {
                name,
                url,
                headers,
                request_timeout_secs,
            } => f
                .debug_struct("Sse")
                .field("name", name)
                .field("url", url)
                .field("header_names", &headers.keys().collect::<BTreeSet<_>>())
                .field("request_timeout_secs", request_timeout_secs)
                .finish(),
        }
    }
}

fn default_request_timeout_secs() -> NonZeroU64 {
    DEFAULT_MCP_REQUEST_TIMEOUT_SECS
}

#[cfg(test)]
mod tests;
