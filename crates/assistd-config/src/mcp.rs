//! MCP (Model Context Protocol) client configuration.

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use url::Url;

use crate::defaults::{DEFAULT_MCP_ENABLED, DEFAULT_MCP_REQUEST_TIMEOUT_SECS};

/// `[mcp]` section of `config.toml`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct McpConfig {
    /// Master switch. When `false` the daemon doesn't connect to any
    /// MCP server, regardless of `servers`.
    pub enabled: bool,
    /// One entry per server the daemon should connect to at startup.
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

/// One `[[mcp.servers]]` entry, discriminated by `transport`.
///
/// The transport-specific keys live inside their variant, so `url` on a
/// stdio server or `env` on an SSE one is a parse error rather than a
/// silently ignored key.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "transport", rename_all = "lowercase", deny_unknown_fields)]
pub enum McpServerConfig {
    /// Newline-delimited JSON-RPC over a child process's stdin/stdout.
    Stdio {
        /// Stable label used in tool-name prefixes (`mcp__<name>__<tool>`)
        /// and in tracing logs. Must be unique within `[mcp.servers]`.
        /// Convention: lowercase, no spaces.
        name: String,
        /// Command to spawn. A bare name is resolved via `$PATH`.
        command: PathBuf,
        /// Arguments passed to the command.
        #[serde(default)]
        args: Vec<String>,
        /// Environment variables injected into the child process.
        #[serde(default)]
        env: HashMap<String, String>,
        /// Per-request JSON-RPC timeout in seconds.
        #[serde(default = "default_request_timeout_secs")]
        request_timeout_secs: NonZeroU64,
    },
    /// HTTP Server-Sent Events endpoint.
    Sse {
        /// See the `stdio` variant's `name`.
        name: String,
        /// Endpoint URL. Parsed at load, so a malformed address fails
        /// with the config rather than at connect time.
        url: Url,
        /// Extra HTTP headers sent with each request.
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
        let McpServerConfig::Stdio {
            name,
            command,
            args,
            request_timeout_secs,
            ..
        } = &cfg.servers[0]
        else {
            panic!("expected a stdio server, got {:?}", cfg.servers[0]);
        };
        assert_eq!(name, "filesystem");
        assert_eq!(command, &PathBuf::from("npx"));
        assert_eq!(args.len(), 3);
        assert_eq!(*request_timeout_secs, DEFAULT_MCP_REQUEST_TIMEOUT_SECS);
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
        let McpServerConfig::Sse { url, headers, .. } = &cfg.servers[0] else {
            panic!("expected an sse server, got {:?}", cfg.servers[0]);
        };
        assert_eq!(url.as_str(), "https://mcp.example.com/sse");
        assert_eq!(
            headers.get("Authorization").map(String::as_str),
            Some("Bearer xyz")
        );
    }

    #[test]
    fn transport_serialises_lowercase() {
        let s = McpServerConfig::Stdio {
            name: "x".into(),
            command: "/bin/x".into(),
            args: Vec::new(),
            env: HashMap::new(),
            request_timeout_secs: DEFAULT_MCP_REQUEST_TIMEOUT_SECS,
        };
        let toml = toml::to_string(&s).unwrap();
        assert!(toml.contains("transport = \"stdio\""), "{toml}");
    }

    #[test]
    fn transport_specific_keys_do_not_cross_variants() {
        let stdio_with_url = r#"
            [[servers]]
            name = "x"
            transport = "stdio"
            command = "npx"
            url = "https://example.com/sse"
        "#;
        let err = toml::from_str::<McpConfig>(stdio_with_url)
            .expect_err("`url` on a stdio server must not parse");
        assert!(err.to_string().contains("url"), "{err}");

        let sse_with_command = r#"
            [[servers]]
            name = "x"
            transport = "sse"
            url = "https://example.com/sse"
            command = "npx"
        "#;
        let err = toml::from_str::<McpConfig>(sse_with_command)
            .expect_err("`command` on an sse server must not parse");
        assert!(err.to_string().contains("command"), "{err}");
    }

    #[test]
    fn malformed_url_is_rejected_at_load() {
        let toml = r#"
            [[servers]]
            name = "x"
            transport = "sse"
            url = "not a url"
        "#;
        assert!(toml::from_str::<McpConfig>(toml).is_err());
    }

    #[test]
    fn zero_request_timeout_is_rejected_at_load() {
        let toml = r#"
            [[servers]]
            name = "x"
            transport = "stdio"
            command = "npx"
            request_timeout_secs = 0
        "#;
        assert!(toml::from_str::<McpConfig>(toml).is_err());
    }
}
