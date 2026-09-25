//! MCP subsystem wiring for the daemon.

use std::time::Duration;

use assistd_core::{Config, McpServerConfig, McpStartupFailure};
use assistd_mcp::{
    McpServerHandle, SseConfig, StdioConfig, TransportConfig, adapt_handle_as_tools,
};
use assistd_tools::{MCP_TOOL_NAME_PREFIX, Tool};
use tokio::sync::watch;
use tracing::info;

pub struct McpSubsystem {
    pub handles: Vec<McpServerHandle>,
    pub tools: Vec<Box<dyn Tool>>,
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
    let mut subsystem = McpSubsystem {
        handles: Vec::new(),
        tools: Vec::new(),
        startup_failures: Vec::new(),
    };
    if !config.mcp.enabled {
        info!("mcp: disabled in config (mcp.enabled = false)");
        return subsystem;
    }

    for server in &config.mcp.servers {
        match start_server(server, shutdown_tx.subscribe()).await {
            Ok((handle, tools)) => {
                subsystem.tools.extend(tools);
                subsystem.handles.push(handle);
            }
            Err(failure) => subsystem.startup_failures.push(failure),
        }
    }
    subsystem
}

/// Start one server and discover its tools, shutting the server down again
/// when discovery fails.
async fn start_server(
    server: &McpServerConfig,
    shutdown: watch::Receiver<bool>,
) -> Result<(McpServerHandle, Vec<Box<dyn Tool>>), McpStartupFailure> {
    let transport = build_transport_config(server);
    let label = server.name().to_string();
    let handle = match McpServerHandle::start(label.clone(), transport, shutdown).await {
        Ok(handle) => handle,
        Err(e) => {
            let reason = format!("failed to start: {e:#}");
            tracing::warn!("mcp: {label} {reason}; skipping");
            return Err(McpStartupFailure {
                server_name: label,
                reason,
            });
        }
    };

    let prefix = format!("{MCP_TOOL_NAME_PREFIX}{}", handle.name);
    match adapt_handle_as_tools(&handle, &prefix).await {
        Ok(tools) => {
            info!(
                "mcp: {} ready ({} tools, transport={})",
                handle.name,
                tools.len(),
                server.transport()
            );
            Ok((handle, tools))
        }
        Err(e) => {
            let reason = format!("discovery failed: {e:#}");
            tracing::warn!("mcp: {} {reason}; shutting down server", handle.name);
            let failure = McpStartupFailure {
                server_name: handle.name.clone(),
                reason,
            };
            handle.shutdown().await;
            Err(failure)
        }
    }
}

fn build_transport_config(server: &McpServerConfig) -> TransportConfig {
    let request_timeout = Duration::from_secs(server.request_timeout_secs().get());
    match server {
        McpServerConfig::Stdio {
            name,
            command,
            args,
            env,
            ..
        } => {
            let mut stdio = StdioConfig::new(name.clone(), command.to_string_lossy().into_owned());
            stdio.args = args.clone();
            stdio.env = env.clone();
            stdio.request_timeout = request_timeout;
            TransportConfig::Stdio(stdio)
        }
        McpServerConfig::Sse {
            name, url, headers, ..
        } => {
            let mut sse = SseConfig::new(name.clone(), url.to_string());
            sse.headers = headers.clone();
            sse.request_timeout = request_timeout;
            TransportConfig::Sse(sse)
        }
    }
}
