pub mod agent;
pub mod config;
pub mod presence;
pub mod socket;
pub mod state;

pub use agent::run_agent_turn;
pub use assistd_ipc as ipc;
pub use assistd_ipc::PresenceState;
pub use assistd_tools::{CommandRegistry, ToolRegistry};
pub use config::{AgentConfig, Config, DaemonConfig, PresenceConfig, SleepConfig};
pub use presence::{PresenceManager, RequestGuard};
pub use state::AppState;

use anyhow::{Context, Result};
use assistd_tools::{
    PresentSpec, RunTool,
    commands::{
        BashCommand, CatCommand, EchoCommand, GrepCommand, LsCommand, SeeCommand, WcCommand,
        WebCommand, WriteCommand,
    },
};
use std::path::PathBuf;
use std::sync::Arc;

/// Build the shared tool registry used by both the daemon and the chat
/// TUI. Clears and recreates `overflow_dir` so per-process spill files
/// land in a known-empty location at every startup.
///
/// `overflow_dir` is taken as an owned parameter so callers (daemon vs
/// TUI) can use different paths — sharing one directory would race
/// when both processes try to reset it.
pub fn build_tools(config: &Config, overflow_dir: PathBuf) -> Result<Arc<ToolRegistry>> {
    if overflow_dir.exists() {
        std::fs::remove_dir_all(&overflow_dir).with_context(|| {
            format!(
                "failed to clear tools.output.overflow_dir {}",
                overflow_dir.display()
            )
        })?;
    }
    std::fs::create_dir_all(&overflow_dir).with_context(|| {
        format!(
            "failed to create tools.output.overflow_dir {}",
            overflow_dir.display()
        )
    })?;

    let mut commands = CommandRegistry::new();
    commands.register(CatCommand);
    commands.register(LsCommand);
    commands.register(GrepCommand);
    commands.register(WcCommand);
    commands.register(EchoCommand);
    commands.register(WriteCommand);
    commands.register(SeeCommand);
    commands.register(WebCommand::new());
    commands.register(BashCommand::default());

    let present_spec = PresentSpec {
        max_lines: config.tools.output.max_lines as usize,
        max_bytes: (config.tools.output.max_kb as usize) * 1024,
        overflow_dir,
    };
    let mut tools = ToolRegistry::new();
    tools.register(RunTool::new(Arc::new(commands), present_spec));
    Ok(Arc::new(tools))
}

/// Returns the version string of the assistd-core crate.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_not_empty() {
        assert!(!version().is_empty());
    }
}
