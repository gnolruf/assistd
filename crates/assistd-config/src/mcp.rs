//! MCP (Model Context Protocol) client configuration.
//!
//! `[mcp]` is a top-level config section. When `enabled = true` the
//! daemon connects to each `[[mcp.servers]]` entry at startup, discovers
//! its tool catalog via `tools/list`, and registers each tool in the
//! global `ToolRegistry` under the namespace `mcp__<server>__<tool>`.
//!
//! Two transports are supported via the `transport` field:
//!   * `"stdio"` — daemon spawns a child process and speaks
//!     newline-delimited JSON-RPC over its stdin/stdout. Requires
//!     `command` (and optional `args`/`env`).
//!   * `"sse"` — daemon connects to a remote HTTP+SSE endpoint.
//!     Requires `url` (and optional `headers`).
//!
//! The transport-discriminated fields (`command`/`url`) are siblings
//! rather than tagged enum variants because TOML's untagged-enum support
//! is finicky; `Config::validate` enforces shape across the (transport,
//! field) combinations.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::defaults::{
    DEFAULT_MCP_ENABLED, DEFAULT_MCP_REQUEST_TIMEOUT_SECS, DEFAULT_MCP_SSE_PING_INTERVAL_SECS,
    DEFAULT_MCP_SSE_READ_TIMEOUT_SECS,
};

/// `[mcp]` section of `config.toml`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct McpConfig {
    /// Master switch. When `false` the daemon doesn't connect to any
    /// MCP server, regardless of `servers`.
    #[serde(default = "default_mcp_enabled")]
    pub enabled: bool,
    /// One entry per server the daemon should connect to at startup.
    #[serde(default)]
    pub servers: Vec<McpServerConfig>,
}

impl Default for McpConfig {
    fn default() -> Self {
        Self {
            enabled: default_mcp_enabled(),
            servers: Vec::new(),
        }
    }
}

/// One `[[mcp.servers]]` entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct McpServerConfig {
    /// Stable label used in tool-name prefixes (`mcp__<name>__<tool>`)
    /// and in tracing logs. Must be unique within `[mcp.servers]` and
    /// non-empty. Convention: lowercase, no spaces.
    pub name: String,
    pub transport: McpTransport,

    // Stdio fields (required when `transport = "stdio"`).
    #[serde(default)]
    pub command: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,

    // SSE fields (required when `transport = "sse"`).
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub headers: HashMap<String, String>,

    // Common.
    #[serde(default = "default_request_timeout_secs")]
    pub request_timeout_secs: u64,
    #[serde(default = "default_sse_read_timeout_secs")]
    pub sse_read_timeout_secs: u64,
    #[serde(default = "default_sse_ping_interval_secs")]
    pub sse_ping_interval_secs: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum McpTransport {
    Stdio,
    Sse,
}

fn default_mcp_enabled() -> bool {
    DEFAULT_MCP_ENABLED
}
fn default_request_timeout_secs() -> u64 {
    DEFAULT_MCP_REQUEST_TIMEOUT_SECS
}
fn default_sse_read_timeout_secs() -> u64 {
    DEFAULT_MCP_SSE_READ_TIMEOUT_SECS
}
fn default_sse_ping_interval_secs() -> u64 {
    DEFAULT_MCP_SSE_PING_INTERVAL_SECS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_round_trips_through_toml() {
        let cfg = McpConfig::default();
        let s = toml::to_string(&cfg).unwrap();
        let back: McpConfig = toml::from_str(&s).unwrap();
        assert_eq!(cfg, back);
    }

    #[test]
    fn omitted_section_uses_defaults() {
        #[derive(Deserialize)]
        struct Wrap {
            #[serde(default)]
            mcp: McpConfig,
        }
        let parsed: Wrap = toml::from_str("").unwrap();
        assert_eq!(parsed.mcp, McpConfig::default());
    }

    #[test]
    fn default_is_disabled() {
        let cfg = McpConfig::default();
        assert!(!cfg.enabled);
        assert!(cfg.servers.is_empty());
    }

    #[test]
    fn parses_stdio_server_minimally() {
        let toml = r#"
            enabled = true

            [[servers]]
            name = "filesystem"
            transport = "stdio"
            command = "npx"
            args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        "#;
        let cfg: McpConfig = toml::from_str(toml).unwrap();
        assert_eq!(cfg.servers.len(), 1);
        let s = &cfg.servers[0];
        assert_eq!(s.name, "filesystem");
        assert_eq!(s.transport, McpTransport::Stdio);
        assert_eq!(s.command.as_deref(), Some("npx"));
        assert_eq!(s.args.len(), 3);
        assert!(s.url.is_none());
        // Defaults applied when section omits them.
        assert_eq!(s.request_timeout_secs, DEFAULT_MCP_REQUEST_TIMEOUT_SECS);
    }

    #[test]
    fn parses_sse_server_minimally() {
        let toml = r#"
            enabled = true

            [[servers]]
            name = "remote"
            transport = "sse"
            url = "https://mcp.example.com/sse"

            [servers.headers]
            Authorization = "Bearer xyz"
        "#;
        let cfg: McpConfig = toml::from_str(toml).unwrap();
        assert_eq!(cfg.servers.len(), 1);
        let s = &cfg.servers[0];
        assert_eq!(s.transport, McpTransport::Sse);
        assert_eq!(s.url.as_deref(), Some("https://mcp.example.com/sse"));
        assert_eq!(s.headers.get("Authorization").map(String::as_str), Some("Bearer xyz"));
        assert!(s.command.is_none());
    }

    #[test]
    fn transport_serialises_lowercase() {
        let s = McpServerConfig {
            name: "x".into(),
            transport: McpTransport::Stdio,
            command: Some("/bin/x".into()),
            args: Vec::new(),
            env: HashMap::new(),
            url: None,
            headers: HashMap::new(),
            request_timeout_secs: DEFAULT_MCP_REQUEST_TIMEOUT_SECS,
            sse_read_timeout_secs: DEFAULT_MCP_SSE_READ_TIMEOUT_SECS,
            sse_ping_interval_secs: DEFAULT_MCP_SSE_PING_INTERVAL_SECS,
        };
        let toml = toml::to_string(&s).unwrap();
        assert!(toml.contains("transport = \"stdio\""));
    }
}
