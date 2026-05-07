//! MCP server subsystem wiring for the daemon.
//!
//! Translates each configured MCP server into a [`TransportConfig`],
//! starts the corresponding [`McpServerHandle`], and adapts discovered
//! tools onto the daemon's [`assistd_tools::Tool`] surface. Failures of
//! individual servers are logged and skipped so a single broken server
//! cannot block daemon startup.

use std::time::Duration;

use assistd_core::{Config, McpServerConfig, McpTransport};
use assistd_mcp::{
    McpServerHandle, SseConfig, StdioConfig, TransportConfig, adapt_handle_as_tools,
};
use tokio::sync::watch;
use tracing::info;

pub struct McpSubsystem {
    pub handles: Vec<McpServerHandle>,
    pub tools: Vec<Box<dyn assistd_tools::Tool>>,
}

impl McpSubsystem {
    pub async fn shutdown(self) {
        for handle in self.handles {
            handle.shutdown().await;
        }
    }
}

pub async fn init(config: &Config, shutdown_tx: &watch::Sender<bool>) -> McpSubsystem {
    if !config.mcp.enabled {
        info!("mcp: disabled in config (mcp.enabled = false)");
        return McpSubsystem {
            handles: Vec::new(),
            tools: Vec::new(),
        };
    }

    let mut handles: Vec<McpServerHandle> = Vec::new();
    let mut tools: Vec<Box<dyn assistd_tools::Tool>> = Vec::new();
    for s_cfg in &config.mcp.servers {
        let transport_cfg = build_transport_config(s_cfg);
        let label = s_cfg.name.clone();
        match McpServerHandle::start(label.clone(), transport_cfg, shutdown_tx.subscribe()).await {
            Ok(handle) => {
                let prefix = format!("mcp__{}", handle.name);
                match adapt_handle_as_tools(&handle, &prefix).await {
                    Ok(t) => {
                        info!(
                            "mcp: {} ready ({} tools, transport={:?})",
                            handle.name,
                            t.len(),
                            s_cfg.transport
                        );
                        tools.extend(t);
                        handles.push(handle);
                    }
                    Err(e) => {
                        tracing::warn!(
                            "mcp: {} discovery failed ({e:#}); shutting down server",
                            handle.name
                        );
                        handle.shutdown().await;
                    }
                }
            }
            Err(e) => {
                tracing::warn!("mcp: {label} failed to start ({e:#}); skipping");
            }
        }
    }
    McpSubsystem { handles, tools }
}

fn build_transport_config(s: &McpServerConfig) -> TransportConfig {
    match s.transport {
        McpTransport::Stdio => {
            let mut cfg = StdioConfig::new(s.name.clone(), s.command.clone().unwrap_or_default());
            cfg.args = s.args.clone();
            cfg.env = s.env.clone();
            cfg.request_timeout = Duration::from_secs(s.request_timeout_secs);
            TransportConfig::Stdio(cfg)
        }
        McpTransport::Sse => {
            let mut cfg = SseConfig::new(s.name.clone(), s.url.clone().unwrap_or_default());
            cfg.headers = s.headers.clone();
            cfg.request_timeout = Duration::from_secs(s.request_timeout_secs);
            cfg.read_timeout = Duration::from_secs(s.sse_read_timeout_secs);
            cfg.ping_interval = Duration::from_secs(s.sse_ping_interval_secs);
            TransportConfig::Sse(cfg)
        }
    }
}
