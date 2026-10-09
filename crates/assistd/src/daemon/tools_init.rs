//! Tool wiring for the daemon: the sandbox probe, then the tool registry
//! and the MCP servers to start, neither of which exists without a sandbox.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use assistd_core::{BuildToolsDeps, Config, McpServerStatus, ToolRegistry, WindowManager};
use assistd_memory::SessionId;
use assistd_tools::{
    ApprovalGate, ConfirmationGate, IpcConfirmationGate, MemoryOps, ToolSandbox, ToolsDisabled,
    VisionGate,
};
use tokio::sync::watch;
use tracing::info;

use super::embed_init::EmbeddingHandles;
use super::mcp_init::{self, McpStartup};
use super::memory_init::MemorySubsystem;

/// The built-in tools the model is offered from startup, and the MCP
/// servers whose tools join them once each has connected.
pub(super) struct ToolsSubsystem {
    pub registry: Arc<ToolRegistry>,
    pub mcp_servers: Vec<McpServerStatus>,
    pub mcp_startup: Option<McpStartup>,
    /// Set when no sandbox is available, so the registry is empty.
    pub disabled: Option<ToolsDisabled>,
}

/// The subsystems the tools reach.
pub(super) struct ToolDeps<'a> {
    pub vision_gate: Arc<VisionGate>,
    pub memory: &'a MemorySubsystem,
    pub embed: &'a EmbeddingHandles,
    pub current_session: watch::Receiver<Arc<SessionId>>,
    pub window_manager: Arc<dyn WindowManager>,
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
                mcp_servers: Vec::new(),
                mcp_startup: None,
                disabled: Some(reason),
            });
        }
    };
    let gate: Arc<dyn ConfirmationGate> = Arc::new(IpcConfirmationGate);
    let mcp_approvals = ApprovalGate::new(
        gate.clone(),
        Arc::new(assistd_core::mcp_tool_approvals(config_path)?),
    );
    let (mcp_servers, mcp_startup) =
        mcp_init::prepare(config, mcp_approvals, deps.vision_gate.clone());
    let overflow_dir = PathBuf::from(&config.tools.output.overflow_dir);
    let registry = assistd_core::build_tools(BuildToolsDeps {
        config,
        config_path,
        overflow_dir: overflow_dir.clone(),
        sandbox,
        confirmation_gate: gate,
        vision_gate: deps.vision_gate,
        memory_ops: Arc::new(MemoryOps::new(
            deps.memory.memory_store.clone(),
            deps.memory.conversation_store.clone(),
        )),
        embedder: deps.embed.embedder.clone(),
        semantic: deps.embed.semantic_store.clone(),
        embed_tx: deps.embed.embed_tx.clone(),
        current_session: deps.current_session,
        window_manager: deps.window_manager,
    })?;
    info!(
        "tools: registered {} (overflow dir {})",
        registry.len(),
        overflow_dir.display()
    );
    Ok(ToolsSubsystem {
        registry,
        mcp_servers,
        mcp_startup,
        disabled: None,
    })
}
