//! MCP subsystem wiring for the daemon.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use assistd_core::{Config, McpServerConfig, McpStartupFailure};
use assistd_mcp::{McpServer, StdioConfig, adapt_client_as_tools};
use assistd_tools::presentation::PresentSpec;
use assistd_tools::{ApprovalGate, Tool};
use tracing::info;

#[derive(Default)]
pub(super) struct McpSubsystem {
    pub servers: Vec<Arc<McpServer>>,
    pub tools: Vec<Box<dyn Tool>>,
    pub startup_failures: Vec<McpStartupFailure>,
}

impl McpSubsystem {
    pub(super) async fn shutdown(self) {
        for server in self.servers {
            server.shutdown().await;
        }
    }
}

/// Start every configured MCP server, whose tools ask before each call
/// unless `approvals` holds them. A server that fails to start or to
/// list its tools is recorded in `startup_failures` and skipped.
pub(super) async fn init(config: &Config, approvals: &ApprovalGate) -> McpSubsystem {
    let mut subsystem = McpSubsystem::default();
    if !config.mcp.enabled {
        info!("mcp: disabled in config (mcp.enabled = false)");
        return subsystem;
    }

    let overflow_dir = PathBuf::from(&config.tools.output.overflow_dir);
    let output = PresentSpec::from_config(&config.tools.output, overflow_dir);
    for server in &config.mcp.servers {
        match start_server(server, output.clone(), approvals).await {
            Ok((server, tools)) => {
                subsystem.tools.extend(tools);
                subsystem.servers.push(server);
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
    output: PresentSpec,
    approvals: &ApprovalGate,
) -> Result<(Arc<McpServer>, Vec<Box<dyn Tool>>), McpStartupFailure> {
    let name = server.name.clone();
    let started = match McpServer::start(name.clone(), stdio_config(server)).await {
        Ok(started) => Arc::new(started),
        Err(e) => {
            let reason = format!("failed to start: {e:#}");
            tracing::warn!("mcp: {name} {reason}; skipping");
            return Err(McpStartupFailure {
                server_name: name,
                reason,
            });
        }
    };

    match adapt_client_as_tools(started.clone(), &name, output, approvals).await {
        Ok(tools) => {
            info!("mcp: {name} ready ({} tools)", tools.len());
            Ok((started, tools))
        }
        Err(e) => {
            let reason = format!("discovery failed: {e:#}");
            tracing::warn!("mcp: {name} {reason}; shutting down server");
            started.shutdown().await;
            Err(McpStartupFailure {
                server_name: name,
                reason,
            })
        }
    }
}

fn stdio_config(server: &McpServerConfig) -> StdioConfig {
    let mut stdio = StdioConfig::new(
        server.name.clone(),
        server.command.to_string_lossy().into_owned(),
    );
    stdio.args.clone_from(&server.args);
    stdio.env.clone_from(&server.env);
    stdio.request_timeout = Duration::from_secs(server.request_timeout_secs.get());
    stdio
}
