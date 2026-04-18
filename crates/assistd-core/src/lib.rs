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
    ConfirmationGate, PresentSpec, RunTool, SandboxRequest,
    commands::{
        BashCommand, BashPolicyCfg, CatCommand, EchoCommand, GrepCommand, LsCommand, SeeCommand,
        WcCommand, WebCommand, WriteCommand, WritePolicyCfg,
    },
    probe_sandbox,
};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tracing::warn;

/// Build the shared tool registry used by both the daemon and the chat
/// TUI. Clears and recreates `overflow_dir` so per-process spill files
/// land in a known-empty location at every startup.
///
/// `overflow_dir` is taken as an owned parameter so callers (daemon vs
/// TUI) can use different paths — sharing one directory would race
/// when both processes try to reset it.
///
/// `gate` is consulted by destructive bash commands. The daemon passes a
/// `DenyAllGate` (headless path: no interactive confirmation available);
/// the chat TUI passes a `TuiGate` that forwards prompts to the user via
/// a modal overlay.
pub fn build_tools(
    config: &Config,
    overflow_dir: PathBuf,
    gate: Arc<dyn ConfirmationGate>,
) -> Result<Arc<ToolRegistry>> {
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

    // Resolve sandbox availability once at startup. `Auto` silently
    // degrades to unsandboxed with a warn; `Bwrap` bails startup if bwrap
    // is missing.
    let sandbox_request = match config.tools.bash.sandbox {
        config::BashSandboxMode::Auto => SandboxRequest::Auto,
        config::BashSandboxMode::Bwrap => SandboxRequest::Bwrap,
        config::BashSandboxMode::None => SandboxRequest::None,
    };
    let sandbox = probe_sandbox(sandbox_request, config.tools.bash.bwrap_extra_args.clone())?;

    // Build bash policy config: shlex-tokenize destructive patterns once
    // so the hot path in `matches_destructive` doesn't re-parse them.
    let destructive_patterns: Vec<Vec<String>> = config
        .tools
        .bash
        .destructive_patterns
        .iter()
        .filter_map(|p| shlex::split(p))
        .filter(|toks| !toks.is_empty())
        .collect();
    let bash_cfg = Arc::new(BashPolicyCfg {
        timeout: Duration::from_secs(config.tools.bash.timeout_secs),
        denylist: config.tools.bash.denylist.clone(),
        destructive_patterns,
    });

    // Build write policy config: expand `~`, canonicalize each entry.
    // Non-existent entries are dropped with a warning so a fresh install
    // missing (say) `~/Documents` doesn't break the command entirely.
    let mut writable_paths: Vec<PathBuf> = Vec::new();
    for raw in &config.tools.write.writable_paths {
        let expanded = expand_config_tilde(raw);
        match std::fs::canonicalize(&expanded) {
            Ok(p) => writable_paths.push(p),
            Err(e) => {
                warn!(
                    target: "assistd::policy",
                    path = %expanded.display(),
                    error = %e,
                    "tools.write.writable_paths entry does not exist — dropping from allowlist"
                );
            }
        }
    }
    if writable_paths.is_empty() {
        anyhow::bail!(
            "tools.write.writable_paths contains no resolvable directories; \
             fix ~/.config/assistd/config.toml"
        );
    }
    let write_cfg = Arc::new(WritePolicyCfg::new(writable_paths));

    let mut commands = CommandRegistry::new();
    commands.register(CatCommand);
    commands.register(LsCommand);
    commands.register(GrepCommand);
    commands.register(WcCommand);
    commands.register(EchoCommand);
    commands.register(WriteCommand::new(write_cfg));
    commands.register(SeeCommand);
    commands.register(WebCommand::new());
    commands.register(BashCommand::new(bash_cfg, sandbox, gate));

    let present_spec = PresentSpec {
        max_lines: config.tools.output.max_lines as usize,
        max_bytes: (config.tools.output.max_kb as usize) * 1024,
        overflow_dir,
    };
    let mut tools = ToolRegistry::new();
    tools.register(RunTool::new(Arc::new(commands), present_spec));
    Ok(Arc::new(tools))
}

/// Expand a leading `~` / `~/` in a config-supplied path string using
/// `$HOME`. Falls back to the literal path when `$HOME` is unset so the
/// subsequent canonicalize call produces the diagnostic error.
fn expand_config_tilde(raw: &str) -> PathBuf {
    if let Some(rest) = raw.strip_prefix("~/") {
        match std::env::var("HOME") {
            Ok(home) => PathBuf::from(home).join(rest),
            Err(_) => PathBuf::from(raw),
        }
    } else if raw == "~" {
        match std::env::var("HOME") {
            Ok(home) => PathBuf::from(home),
            Err(_) => PathBuf::from(raw),
        }
    } else {
        PathBuf::from(raw)
    }
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
