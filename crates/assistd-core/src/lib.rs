#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::print_stdout,
        clippy::print_stderr
    )
)]

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
pub mod recovery;
pub mod socket;
pub mod state;

pub use agent::Agent;
pub use recovery::{Component, RecoverySeverity, install_panic_hook, spawn_supervised};

pub use assistd_config as config;
pub use assistd_config::{
    AgentConfig, BashSandboxMode, ChatConfig, CompositorConfig, CompositorType, Config,
    ConfigError, ContinuousListenConfig, DaemonConfig, LlamaServerConfig, McpConfig,
    McpServerConfig, McpTransport, ModelConfig, PresenceConfig, RemoteConfig, ScreenshotBackend,
    SleepConfig, SynthesisConfig, ToolsBashConfig, ToolsConfig, ToolsOutputConfig,
    ToolsScreenshotConfig, ToolsWriteConfig, VoiceConfig,
};

pub use assistd_ipc as ipc;
pub use assistd_ipc::{PresenceState, VoiceCaptureState};

pub use assistd_tools::{CommandRegistry, ToolRegistry};

pub use assistd_voice::{
    ContinuousListener, NoContinuousListener, NoVoiceInput, NoVoiceOutput, SpeakDecision,
    VoiceInput, VoiceOutput, VoiceOutputController,
};

pub use assistd_wm::{NoWindowManager, WindowManager};

pub use presence::{PresenceManager, RequestGuard};
pub use state::{AppState, ConversationContext};

use anyhow::{Context, Result};
use assistd_embed::{EmbedJob, Embedder};
use assistd_memory::SemanticStore;
use assistd_tools::{
    ConfirmationGate, MemoryOps, RecallTool, RememberTool, ReminisceTool, RunTool, SandboxRequest,
    commands::{
        BashCommand, BashPolicyCfg, CatCommand, EchoCommand, GrepCommand, LsCommand,
        ScreenshotBackendKind, ScreenshotCommand, ScreenshotPolicyCfg, SeeCommand, WcCommand,
        WebCommand, WmCommand, WriteCommand, WritePolicyCfg,
    },
    probe_sandbox,
};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::warn;

/// Subsystem handles the daemon injects into [`build_tools`]. Bundled
/// into one struct so the call site is a flat field initialiser rather
/// than a dozen positional arguments.
///
/// Field notes:
///
/// - `overflow_dir` is owned so the caller chooses the path — kept as a
///   distinct field for future call sites that need a different spill
///   directory.
/// - `confirmation_gate` is consulted by destructive bash commands. The
///   daemon passes an `IpcConfirmationGate` that forwards prompts to the
///   active IPC client (the TUI is one such client) and falls through to
///   deny when no router is in scope.
/// - `vision_gate` is a shared, runtime-mutable flag (see
///   [`assistd_tools::VisionGate`]) initialised from a `/props` probe of
///   the running llama-server. When the gate reports `supported = false`,
///   the image-producing commands (`see`, `screenshot`) still register
///   but their `run()` short-circuits with the
///   `[error] …: vision not available …` line and their `summary()` flips
///   so the LLM sees the unavailability in its tool schema. The gate is
///   re-evaluated on every command invocation, so a daemon-side
///   revalidation that flips it after a model swap takes effect without
///   rebuilding the registry.
/// - `memory_ops` is the combined CRUD façade backing the LLM-callable
///   `remember` and `recall` tools. The daemon passes a SQLite-backed
///   handle.
/// - `window_manager` backs the LLM-callable `wm` command. The daemon
///   passes either a connected `I3Backend` (when `[compositor].type =
///   "i3"` and the i3 socket is reachable) or [`NoWindowManager`] when
///   the configured backend is unavailable; the latter makes every `wm`
///   subcommand short-circuit with a uniform "compositor not connected"
///   error rather than failing per-subcommand.
pub struct BuildToolsDeps<'a> {
    pub config: &'a Config,
    pub overflow_dir: PathBuf,
    pub confirmation_gate: Arc<dyn ConfirmationGate>,
    pub vision_gate: Arc<assistd_tools::VisionGate>,
    pub memory_ops: Arc<MemoryOps>,
    pub embedder: Arc<dyn Embedder>,
    pub semantic: Arc<dyn SemanticStore>,
    pub embed_tx: mpsc::Sender<EmbedJob>,
    pub embedding_model: String,
    pub window_manager: Arc<dyn WindowManager>,
    pub mcp_tools: Vec<Box<dyn assistd_tools::Tool>>,
}

/// Build the tool registry consumed by the daemon. Clears and recreates
/// [`BuildToolsDeps::overflow_dir`] so per-process spill files land in a
/// known-empty location at every startup. See [`BuildToolsDeps`] for the
/// role of each injected handle.
pub fn build_tools(deps: BuildToolsDeps<'_>) -> Result<Arc<ToolRegistry>> {
    let BuildToolsDeps {
        config,
        overflow_dir,
        confirmation_gate,
        vision_gate,
        memory_ops,
        embedder,
        semantic,
        embed_tx,
        embedding_model,
        window_manager,
        mcp_tools,
    } = deps;

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

    let sandbox_request = match config.tools.bash.sandbox {
        BashSandboxMode::Auto => SandboxRequest::Auto,
        BashSandboxMode::Bwrap => SandboxRequest::Bwrap,
        BashSandboxMode::None => SandboxRequest::None,
    };
    let sandbox = probe_sandbox(sandbox_request, config.tools.bash.bwrap_extra_args.clone())?;

    // Shlex-tokenize destructive patterns once so the hot path in
    // `matches_destructive` doesn't re-parse them on every invocation.
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

    let screenshot_cfg = Arc::new(ScreenshotPolicyCfg {
        backend: match config.tools.screenshot.backend {
            ScreenshotBackend::Auto => None,
            ScreenshotBackend::X11 => Some(ScreenshotBackendKind::X11),
            ScreenshotBackend::Wayland => Some(ScreenshotBackendKind::Wayland),
        },
        timeout: Duration::from_secs(config.tools.screenshot.timeout_secs),
    });

    let mut commands = CommandRegistry::new();
    commands.register(CatCommand);
    commands.register(LsCommand);
    commands.register(GrepCommand);
    commands.register(WcCommand);
    commands.register(EchoCommand);
    commands.register(WriteCommand::new(write_cfg));
    commands.register(SeeCommand::new(vision_gate.clone()));
    commands.register(ScreenshotCommand::new(screenshot_cfg, vision_gate));
    commands.register(WebCommand::new());
    commands.register(BashCommand::new(bash_cfg, sandbox, confirmation_gate));
    commands.register(WmCommand::new(window_manager));

    let mut tools = ToolRegistry::new();
    tools.register(RunTool::new(
        Arc::new(commands),
        &config.tools.output,
        overflow_dir,
    ));
    tools.register(RememberTool::new(memory_ops, embed_tx));
    tools.register(RecallTool::new(
        embedder.clone(),
        semantic.clone(),
        embedding_model.clone(),
    ));
    tools.register(ReminisceTool::new(embedder, semantic, embedding_model));

    // MCP tools are registered last so a malicious server cannot shadow
    // native tool names — the registry's linear lookup returns the first match.
    for t in mcp_tools {
        tools.register_boxed(t);
    }
    Ok(Arc::new(tools))
}

/// Cache the running llama-server's model id alongside the
/// [`VisionGate`] state, so a re-probe can detect a model swap and
/// flip vision availability without rebuilding the tool registry.
///
/// The daemon constructs one [`VisionRevalidator`] at startup, hands it
/// to [`AppState`], and the per-query handler calls [`revalidate`] at
/// the top of every turn. The probe is cheap (one local HTTP `GET
/// /props` with a 2-second timeout), and we only mutate the gate when
/// the cached model id actually changes — so a steady-state daemon
/// pays a single round-trip per turn and never thrashes the gate.
///
/// [`revalidate`]: VisionRevalidator::revalidate
pub struct VisionRevalidator {
    gate: Arc<assistd_tools::VisionGate>,
    cached_model: tokio::sync::Mutex<Option<String>>,
    host: String,
    port: u16,
}

impl VisionRevalidator {
    /// Construct a new revalidator and wrap it in an `Arc`.
    ///
    /// `initial_model_id` is the model id known at startup; `None` means the
    /// first probe result will be accepted unconditionally as the baseline.
    pub fn new(
        gate: Arc<assistd_tools::VisionGate>,
        initial_model_id: Option<String>,
        host: String,
        port: u16,
    ) -> Arc<Self> {
        Arc::new(Self {
            gate,
            cached_model: tokio::sync::Mutex::new(initial_model_id),
            host,
            port,
        })
    }

    /// Re-probe `/props` and, if the model id changed, update the
    /// gate. Tolerates probe failures silently — a transient HTTP
    /// blip should not flip vision off mid-session.
    pub async fn revalidate(&self) {
        let probe = assistd_llm::probe_capabilities(&self.host, self.port).await;
        self.apply_probe(probe).await;
    }

    /// Apply a probe result to the cache and gate. Split out from
    /// [`Self::revalidate`] so unit tests can drive the swap logic
    /// without standing up an HTTP server.
    pub async fn apply_probe(&self, probe: assistd_llm::VisionState) {
        // No model id means the probe failed. Don't mutate the gate
        // on a transient error; keep the last known good state.
        if probe.model_id.is_none() {
            return;
        }
        let mut cache = self.cached_model.lock().await;
        if *cache != probe.model_id {
            tracing::info!(
                old = ?*cache,
                new = ?probe.model_id,
                vision_supported = probe.vision_supported,
                "llama-server model changed; updating vision gate"
            );
            *cache = probe.model_id;
            self.gate.set(probe.vision_supported);
        }
    }

    /// Returns a reference to the shared [`assistd_tools::VisionGate`] this
    /// revalidator manages.
    pub fn gate(&self) -> &Arc<assistd_tools::VisionGate> {
        &self.gate
    }
}

#[cfg(test)]
mod vision_revalidator_tests {
    use super::*;
    use assistd_llm::VisionState;

    fn make_revalidator(gate_initial: bool, cached_model: Option<&str>) -> Arc<VisionRevalidator> {
        VisionRevalidator::new(
            assistd_tools::VisionGate::new(gate_initial),
            cached_model.map(str::to_string),
            "127.0.0.1".to_string(),
            // Port is unused in the apply_probe path — set to 0 to
            // make a real revalidate() obviously fail loudly if anyone
            // accidentally points the test at it.
            0,
        )
    }

    #[tokio::test]
    async fn no_model_id_keeps_gate_unchanged() {
        // Simulates a transient probe failure: probe_capabilities
        // returns the default (None model id, false vision). The gate
        // must stay where it was; flipping it on a network blip would
        // disable vision mid-session.
        let rev = make_revalidator(true, Some("model-A"));
        rev.apply_probe(VisionState::default()).await;
        assert!(
            rev.gate().supported(),
            "gate must not flip on probe failure"
        );
    }

    #[tokio::test]
    async fn unchanged_model_id_keeps_gate_unchanged() {
        let rev = make_revalidator(true, Some("model-A"));
        rev.apply_probe(VisionState {
            model_id: Some("model-A".into()),
            // Even if the new probe says false, no model swap means we
            // don't disturb the gate — same model can't lose vision.
            vision_supported: false,
        })
        .await;
        assert!(rev.gate().supported());
    }

    #[tokio::test]
    async fn model_swap_flips_gate_off() {
        let rev = make_revalidator(true, Some("vision-model"));
        rev.apply_probe(VisionState {
            model_id: Some("text-only-model".into()),
            vision_supported: false,
        })
        .await;
        assert!(!rev.gate().supported(), "gate must flip when model changes");
    }

    #[tokio::test]
    async fn model_swap_flips_gate_on() {
        let rev = make_revalidator(false, Some("text-only-model"));
        rev.apply_probe(VisionState {
            model_id: Some("vision-model".into()),
            vision_supported: true,
        })
        .await;
        assert!(rev.gate().supported());
    }

    #[tokio::test]
    async fn fresh_revalidator_with_no_cached_model_still_picks_up_first_probe() {
        // Daemon starts before the probe completes — the initial cache
        // is None. When the first probe lands, we accept it as the
        // baseline (None != Some) and update the gate to match.
        let rev = make_revalidator(false, None);
        rev.apply_probe(VisionState {
            model_id: Some("vision-model".into()),
            vision_supported: true,
        })
        .await;
        assert!(rev.gate().supported());
    }
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
