//! Tool wiring for the daemon: the sandbox probe, then the MCP servers and
//! tool registry, neither of which starts without a sandbox.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use assistd_core::{BuildToolsDeps, Config, ToolRegistry, WindowManager};
use assistd_memory::SessionId;
use assistd_tools::{
    ApprovalGate, ConfirmationGate, IpcConfirmationGate, MemoryOps, ToolSandbox, ToolsDisabled,
    VisionGate,
};
use tokio::sync::watch;
use tracing::info;

use super::embed_init::EmbeddingSubsystem;
use super::mcp_init::{self, McpSubsystem};
use super::memory_init::MemorySubsystem;

/// The tools the model is offered and the MCP servers behind them.
pub(super) struct ToolsSubsystem {
    pub registry: Arc<ToolRegistry>,
    pub mcp: McpSubsystem,
    /// Set when no sandbox is available, so the registry is empty.
    pub disabled: Option<ToolsDisabled>,
}

/// The subsystems the tools reach.
pub(super) struct ToolDeps<'a> {
    pub vision_gate: Arc<VisionGate>,
    pub memory: &'a MemorySubsystem,
    pub embed: &'a EmbeddingSubsystem,
    pub current_session: watch::Receiver<Arc<SessionId>>,
    pub window_manager: Arc<dyn WindowManager>,
}

/// Probe the sandbox, then start the MCP servers and build the registry,
/// or with no sandbox leave both empty.
pub(super) async fn init(
    config: &Config,
    config_path: &Path,
    deps: ToolDeps<'_>,
) -> Result<ToolsSubsystem> {
    let sandbox = match assistd_core::probe_tool_sandbox(config, config_path)? {
        ToolSandbox::Bwrap(sandbox) => sandbox,
        ToolSandbox::Disabled(reason) => {
            return Ok(ToolsSubsystem {
                registry: Arc::new(ToolRegistry::new()),
                mcp: McpSubsystem::default(),
                disabled: Some(reason),
            });
        }
    };
    let gate: Arc<dyn ConfirmationGate> = Arc::new(IpcConfirmationGate);
    let mcp_approvals = ApprovalGate::new(
        gate.clone(),
        Arc::new(assistd_core::mcp_tool_approvals(config_path)?),
    );
    let mut mcp = mcp_init::init(config, &mcp_approvals).await;
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
        embedding_model: deps.embed.model_name.clone(),
        current_session: deps.current_session,
        window_manager: deps.window_manager,
        mcp_tools: std::mem::take(&mut mcp.tools),
    })?;
    info!(
        "tools: registered {} (overflow dir {})",
        registry.len(),
        overflow_dir.display()
    );
    Ok(ToolsSubsystem {
        registry,
        mcp,
        disabled: None,
    })
}
