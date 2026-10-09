//! MCP subsystem wiring for the daemon: servers start in the background
//! and each one's tools are offered once it has connected.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use assistd_core::{AppState, Config, McpServerConfig, McpServerStatus};
use assistd_ipc::{ComponentReadiness, StartupComponent};
use assistd_mcp::{McpServer, StdioConfig, adapt_client_as_tools};
use assistd_tools::presentation::PresentSpec;
use assistd_tools::{ApprovalGate, Tool, VisionGate};
use assistd_utils::readiness::Readiness;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tracing::info;

/// What starting the configured MCP servers in the background needs.
pub(super) struct McpStartup {
    servers: Vec<McpServerConfig>,
    output: PresentSpec,
    approvals: ApprovalGate,
    vision: Arc<VisionGate>,
}

/// The background start, joined at shutdown for the servers it started.
pub(super) struct McpService {
    startup: Option<JoinHandle<Vec<Arc<McpServer>>>>,
}

impl McpStartup {
    /// Start each server in turn, offering its tools and recording its
    /// readiness as it settles. Stops starting servers once `shutdown` flips.
    pub(super) fn spawn(self, state: Arc<AppState>, shutdown: watch::Receiver<bool>) -> McpService {
        McpService {
            startup: Some(tokio::spawn(start_all(self, state, shutdown))),
        }
    }
}

impl McpService {
    pub(super) fn disabled() -> Self {
        Self { startup: None }
    }

    /// Wait for the background start to stop, then stop every server it
    /// started. The `tools` shutdown stage must already have flipped.
    pub(super) async fn shutdown(self) {
        let Some(startup) = self.startup else {
            return;
        };
        let servers = startup.await.unwrap_or_else(|e| {
            tracing::error!("mcp startup task failed: {e}");
            Vec::new()
        });
        for server in servers {
            server.shutdown().await;
        }
    }
}

/// A status per configured server and what starting them needs, whose
/// tools ask before each call unless `approvals` holds them and whose
/// image results follow `vision`. Nothing to start when MCP is disabled.
pub(super) fn prepare(
    config: &Config,
    approvals: ApprovalGate,
    vision: Arc<VisionGate>,
) -> (Vec<McpServerStatus>, Option<McpStartup>) {
    if !config.mcp.enabled {
        info!("mcp: disabled in config (mcp.enabled = false)");
        return (Vec::new(), None);
    }
    let statuses = config
        .mcp
        .servers
        .iter()
        .map(|server| McpServerStatus::starting(server.name.clone()))
        .collect();
    let overflow_dir = PathBuf::from(&config.tools.output.overflow_dir);
    let startup = McpStartup {
        servers: config.mcp.servers.clone(),
        output: PresentSpec::from_config(&config.tools.output, overflow_dir),
        approvals,
        vision,
    };
    (statuses, Some(startup))
}

async fn start_all(
    startup: McpStartup,
    state: Arc<AppState>,
    mut shutdown: watch::Receiver<bool>,
) -> Vec<Arc<McpServer>> {
    let mut started = Vec::new();
    for config in &startup.servers {
        let outcome = tokio::select! {
            biased;
            _ = shutdown.wait_for(|stopping| *stopping) => break,
            outcome = start_server(config, &startup) => outcome,
        };
        let readiness = match outcome {
            Ok((server, tools)) => {
                state.subsystems.tools.extend(tools);
                started.push(server);
                Readiness::Ready(())
            }
            Err(reason) => Readiness::Unavailable(reason.into()),
        };
        record_readiness(&state, &config.name, readiness);
    }
    started
}

fn record_readiness(state: &AppState, name: &str, readiness: Readiness<()>) {
    let Some(status) = state
        .subsystems
        .mcp_servers
        .iter()
        .find(|s| s.name() == name)
    else {
        return;
    };
    status.set(readiness);
    state.publish_readiness(
        StartupComponent::Mcp {
            server: name.to_string(),
        },
        ComponentReadiness::from(status.readiness()),
    );
}

/// Start one server and discover its tools, shutting the server down again
/// when discovery fails. `Err` is why the server is unavailable.
async fn start_server(
    server: &McpServerConfig,
    startup: &McpStartup,
) -> Result<(Arc<McpServer>, Vec<Box<dyn Tool>>), String> {
    let name = server.name.clone();
    let started = match McpServer::start(name.clone(), stdio_config(server)).await {
        Ok(started) => Arc::new(started),
        Err(e) => {
            let reason = format!("failed to start: {e:#}");
            tracing::warn!("mcp: {name} {reason}; skipping");
            return Err(reason);
        }
    };

    match adapt_client_as_tools(
        started.clone(),
        &name,
        startup.output.clone(),
        &startup.approvals,
        &startup.vision,
    )
    .await
    {
        Ok(tools) => {
            info!("mcp: {name} ready ({} tools)", tools.len());
            Ok((started, tools))
        }
        Err(e) => {
            let reason = format!("discovery failed: {e:#}");
            tracing::warn!("mcp: {name} {reason}; shutting down server");
            started.shutdown().await;
            Err(reason)
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

#[cfg(test)]
mod tests;
