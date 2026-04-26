//! Daemon orchestration crate — the glue that wires every subsystem
//! into a running `assistd` process.
//!
//! # Not a stable public library
//!
//! `assistd-core` is the aggregate facade consumed by the `assistd`
//! binary (and a small amount of test code). It is **not** a versioned
//! public API and external code should not depend on it. The re-exports
//! below exist so the binary can `use assistd_core::*` instead of
//! reaching into every subsystem crate.
//!
//! When adding a new subsystem crate (e.g. `assistd-memory`,
//! `assistd-mcp`), define its typed API in that crate and re-export
//! only the daemon-facing surface here, in its own `pub use` block
//! grouped with a comment naming the upstream crate. Anything exported
//! here that is not used outside `assistd-core` itself should be
//! demoted to `pub(crate)`.
//!
//! # Modules
//!
//! - [`agent`] — per-turn LLM/tool loop driver.
//! - [`presence`] — Active/Drowsy/Sleeping state machine and
//!   `LlamaService` lifecycle.
//! - [`socket`] — Unix-socket IPC server (line-delimited JSON).
//! - [`state`] — `AppState` request dispatcher; one handler per
//!   `assistd_ipc::Request` variant.

pub mod agent;
pub mod presence;
pub mod socket;
pub mod state;

pub use agent::run_agent_turn;

// Re-exports from `assistd-config`. Config types are the boundary
// between user-supplied TOML and subsystem constructors; the daemon
// passes them through unchanged.
pub use assistd_config as config;
pub use assistd_config::{
    AgentConfig, BashSandboxMode, ChatConfig, CompositorConfig, CompositorType, Config,
    ConfigError, ContinuousListenConfig, DaemonConfig, LlamaServerConfig, ModelConfig,
    PresenceConfig, RemoteConfig, SleepConfig, SynthesisConfig, ToolsBashConfig, ToolsConfig,
    ToolsOutputConfig, ToolsWriteConfig, VoiceConfig,
};

// Re-exports from `assistd-ipc`. Wire-protocol types crossing the
// socket boundary. The daemon binary uses these to drive request
// handlers and emit events.
pub use assistd_ipc as ipc;
pub use assistd_ipc::{PresenceState, VoiceCaptureState};

// Re-exports from `assistd-tools`. The tool/command subsystem lives
// in its own crate; these are the types the daemon constructs and
// hands to `AppState`.
pub use assistd_tools::{CommandRegistry, ToolRegistry};

// Re-exports from `assistd-voice`. Voice input/output traits plus
// the `No*` placeholders used when the corresponding feature is
// disabled at build time or in config.
pub use assistd_voice::{
    ContinuousListener, NoContinuousListener, NoVoiceInput, NoVoiceOutput, SpeakDecision,
    VoiceInput, VoiceOutput, VoiceOutputController,
};

// In-crate exports. `AppState` is the daemon's shared state container;
// `PresenceManager` owns the presence machine and llama-server lifecycle;
// `RequestGuard` is the RAII handle held for the duration of a request.
pub use presence::{PresenceManager, RequestGuard};
pub use state::AppState;

use anyhow::{Context, Result};
use assistd_tools::{
    ConfirmationGate, RunTool, SandboxRequest,
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
        BashSandboxMode::Auto => SandboxRequest::Auto,
        BashSandboxMode::Bwrap => SandboxRequest::Bwrap,
        BashSandboxMode::None => SandboxRequest::None,
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

    let mut tools = ToolRegistry::new();
    tools.register(RunTool::new(
        Arc::new(commands),
        &config.tools.output,
        overflow_dir,
    ));
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
