//! MCP subsystem wiring for the daemon.

use std::time::Duration;

use assistd_core::{Config, McpServerConfig, McpStartupFailure};
use assistd_mcp::{
    McpServerHandle, SseConfig, StdioConfig, TransportConfig, adapt_handle_as_tools,
};
use tokio::sync::watch;
use tracing::info;

pub struct McpSubsystem {
    pub handles: Vec<McpServerHandle>,
    pub tools: Vec<Box<dyn assistd_tools::Tool>>,
    pub startup_failures: Vec<McpStartupFailure>,
}

impl McpSubsystem {
    pub async fn shutdown(self) {
        for handle in self.handles {
            handle.shutdown().await;
        }
    }
}

/// Start every configured MCP server. A server that fails to start or to
/// list its tools is recorded in `startup_failures` and skipped.
pub async fn init(config: &Config, shutdown_tx: &watch::Sender<bool>) -> McpSubsystem {
    if !config.mcp.enabled {
        info!("mcp: disabled in config (mcp.enabled = false)");
        return McpSubsystem {
            handles: Vec::new(),
            tools: Vec::new(),
            startup_failures: Vec::new(),
        };
    }

    let mut handles: Vec<McpServerHandle> = Vec::new();
    let mut tools: Vec<Box<dyn assistd_tools::Tool>> = Vec::new();
    let mut startup_failures: Vec<McpStartupFailure> = Vec::new();
    for s_cfg in &config.mcp.servers {
        let transport_cfg = build_transport_config(s_cfg);
        let label = s_cfg.name().to_string();
        match McpServerHandle::start(label.clone(), transport_cfg, shutdown_tx.subscribe()).await {
            Ok(handle) => {
                let prefix = format!("{}{}", assistd_tools::MCP_TOOL_NAME_PREFIX, handle.name);
                match adapt_handle_as_tools(&handle, &prefix).await {
                    Ok(t) => {
                        info!(
                            "mcp: {} ready ({} tools, transport={})",
                            handle.name,
                            t.len(),
                            s_cfg.transport()
                        );
                        tools.extend(t);
                        handles.push(handle);
                    }
                    Err(e) => {
                        let reason = format!("discovery failed: {e:#}");
                        tracing::warn!("mcp: {} {reason}; shutting down server", handle.name);
                        startup_failures.push(McpStartupFailure {
                            server_name: handle.name.clone(),
                            reason,
                        });
                        handle.shutdown().await;
                    }
                }
            }
            Err(e) => {
                let reason = format!("failed to start: {e:#}");
                tracing::warn!("mcp: {label} {reason}; skipping");
                startup_failures.push(McpStartupFailure {
                    server_name: label,
                    reason,
                });
            }
        }
    }
    McpSubsystem {
        handles,
        tools,
        startup_failures,
    }
}

fn build_transport_config(s: &McpServerConfig) -> TransportConfig {
    let request_timeout = Duration::from_secs(s.request_timeout_secs().get());
    match s {
        McpServerConfig::Stdio {
            name,
            command,
            args,
            env,
            ..
        } => {
            let mut cfg = StdioConfig::new(name.clone(), command.to_string_lossy().into_owned());
            cfg.args = args.clone();
            cfg.env = env.clone();
            cfg.request_timeout = request_timeout;
            TransportConfig::Stdio(cfg)
        }
        McpServerConfig::Sse {
            name, url, headers, ..
        } => {
            let mut cfg = SseConfig::new(name.clone(), url.to_string());
            cfg.headers = headers.clone();
            cfg.request_timeout = request_timeout;
            TransportConfig::Sse(cfg)
        }
    }
}
