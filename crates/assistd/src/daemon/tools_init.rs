//! Tool wiring for the daemon: the sandbox probe, then the tool registry
//! and the MCP servers to start, neither of which exists without a sandbox.

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use assistd_config::{AgentCatalog, AgentSpec, Config};
use assistd_core::planner::PlanningTools;
use assistd_core::{BuildToolsDeps, McpServerStatus, RelayDeps};
use assistd_memory::SessionId;
use assistd_tools::{
    ConfirmationGate, IpcConfirmationGate, MemoryOps, PinnedApprovalGate, SandboxInfo,
    ToolRegistry, ToolSandbox, ToolsDisabled, VisionGate, WatchedPaths,
};
use assistd_utils::child_server::ChildServerStatus;
use assistd_utils::readiness::ReadinessCell;
use assistd_wm::WindowManager;
use tokio::sync::watch;
use tracing::{info, warn};

use super::embed_init::EmbeddingHandles;
use super::mcp_init::{self, McpFleet};
use super::memory_init::MemorySubsystem;

/// The built-in tools the model is offered from startup, and the MCP
/// servers whose tools join them once each has connected.
pub(super) struct ToolsSubsystem {
    pub registry: Arc<ToolRegistry>,
    pub planning: PlanningTools,
    pub mcp_servers: Vec<McpServerStatus>,
    pub mcp_fleet: Option<Arc<McpFleet>>,
    /// The sandbox the tools run in; without one the registry is empty.
    pub sandbox: Result<Arc<SandboxInfo>, ToolsDisabled>,
}

impl ToolsSubsystem {
    /// The watched paths the sandbox reads; with no sandbox, a set nothing reads.
    pub(super) fn watched_paths(&self) -> Arc<WatchedPaths> {
        self.sandbox
            .as_ref()
            .map_or_else(|_| Arc::default(), |sandbox| Arc::clone(&sandbox.watched))
    }
}

/// The subsystems the tools reach.
pub(super) struct ToolDeps<'a> {
    pub vision_gate: Arc<VisionGate>,
    pub memory: &'a MemorySubsystem,
    pub embed: &'a EmbeddingHandles,
    pub current_session: watch::Receiver<SessionId>,
    pub window_manager: Arc<dyn WindowManager>,
    pub web_search: Arc<ReadinessCell<ChildServerStatus>>,
    pub relay: RelayDeps,
}

/// Probe the sandbox, then build the registry and prepare the MCP
/// servers, or with no sandbox leave both empty.
pub(super) fn init(
    config: &Config,
    config_path: &Path,
    deps: ToolDeps<'_>,
) -> Result<ToolsSubsystem> {
    let sandbox = match assistd_core::probe_tool_sandbox(config, config_path)? {
        ToolSandbox::Bwrap(sandbox) => sandbox,
        ToolSandbox::Disabled(reason) => {
            return Ok(ToolsSubsystem {
                registry: Arc::new(ToolRegistry::new()),
                planning: PlanningTools::default(),
                mcp_servers: Vec::new(),
                mcp_fleet: None,
                sandbox: Err(reason),
            });
        }
    };
    let gate: Arc<dyn ConfirmationGate> = Arc::new(IpcConfirmationGate);
    let mcp_approvals = PinnedApprovalGate::new(
        gate.clone(),
        Arc::new(assistd_core::mcp_tool_approvals(config_path)?),
    );
    let (mcp_servers, mcp_fleet) = mcp_init::prepare(
        config,
        config_path,
        &sandbox,
        mcp_approvals.clone(),
        deps.vision_gate.clone(),
    );
    let overflow_dir = config.tools.output.resolved_overflow_dir()?;
    let built = assistd_core::build_tools(BuildToolsDeps {
        config,
        config_path,
        overflow_dir: overflow_dir.clone(),
        sandbox: Arc::clone(&sandbox),
        confirmation_gate: gate,
        vision_gate: deps.vision_gate,
        memory_ops: Arc::new(MemoryOps::new(
            deps.memory.memory_store.clone(),
            deps.memory.transcript_store.clone(),
        )),
        embedder: deps.embed.embedder.clone(),
        semantic: deps.embed.semantic_store.clone(),
        current_session: deps.current_session,
        window_manager: deps.window_manager,
        web_search: deps.web_search,
        relay: deps.relay,
        agents: load_agents(config, config_path),
        mcp_approvals,
    })?;
    info!(
        "tools: registered {} (overflow dir {})",
        built.registry.len(),
        overflow_dir.display()
    );
    Ok(ToolsSubsystem {
        registry: built.registry,
        planning: built.planning,
        mcp_servers,
        mcp_fleet,
        sandbox: Ok(sandbox),
    })
}

/// The sub-agents `relay` offers: the built-ins and the agent files beside the config at
/// `config_path`, each refused file logged.
fn load_agents(config: &Config, config_path: &Path) -> Vec<AgentSpec> {
    let dir = AgentCatalog::dir_beside(config_path);
    let catalog = AgentCatalog::load(&dir, config);
    for refused in &catalog.refused {
        warn!("agents: {refused}");
    }
    info!(
        "agents: {} loaded, {} refused ({})",
        catalog.agents.len(),
        catalog.refused.len(),
        dir.display()
    );
    catalog.agents
}
