//! Daemon orchestration: the agent loop, presence state machine, IPC
//! socket server, and the `AppState` request dispatcher. Not a stable
//! public API; it re-exports the subsystem crates so dependents need
//! only this one.

pub mod agent;
pub mod presence;
pub mod recovery;
pub mod socket;
pub mod state;

pub use agent::Agent;
pub use recovery::{
    Component, StatusSeverity, drain_join_set, install_panic_hook, spawn_supervised,
};

pub use assistd_config as config;
pub use assistd_config::{
    BashSandboxMode, ChatConfig, CompositorConfig, CompositorType, Config, ConfigError,
    ContinuousListenConfig, DaemonConfig, LlamaServerConfig, McpConfig, McpServerConfig,
    ModelConfig, PresenceConfig, ScreenshotBackend, SleepConfig, SynthesisConfig, ToolsBashConfig,
    ToolsConfig, ToolsOutputConfig, ToolsScreenshotConfig, ToolsWriteConfig, VoiceConfig,
};

pub use assistd_ipc as ipc;
pub use assistd_ipc::{PresenceState, VoiceCaptureState};

pub use assistd_tools::{CommandRegistry, ToolRegistry};

pub use assistd_voice::{
    ContinuousListener, NoContinuousListener, NoVoiceInput, NoVoiceOutput, SpeakDecision,
    VoiceInput, VoiceOutput, VoiceOutputController,
};

pub use assistd_wm::{NoWindowManager, WindowManager};

pub use presence::{PresenceError, PresenceManager, RequestGuard};
pub use state::{
    AppState, ConversationContext, DispatchError, McpStartupFailure, MemoryStack, RuntimeState,
    Subsystems, history_entries,
};

use assistd_embed::{EmbedJob, Embedder};
use assistd_memory::SemanticStore;
use assistd_tools::{
    APPROVALS_FILE, Allowlist, AllowlistError, ConfirmationGate, DestructivePattern, MemoryOps,
    Protected, RecallTool, RememberTool, ReminisceTool, RunTool, SandboxError, SandboxRequest,
    commands::{
        BashCommand, BashPolicyCfg, CatCommand, EchoCommand, GrepCommand, HeadCommand, LsCommand,
        ScreenshotBackendKind, ScreenshotCommand, ScreenshotPolicyCfg, SeeCommand, SortCommand,
        TailCommand, UniqCommand, WcCommand, WebCommand, WmCommand, WriteCommand, WritePolicyCfg,
    },
    probe_sandbox,
};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::{mpsc, watch};
use tracing::warn;

/// Why [`build_tools`] could not assemble the tool registry.
#[derive(Debug, Error)]
pub enum BuildToolsError {
    #[error("failed to clear tools.output.overflow_dir {}: {source}", path.display())]
    ClearOverflowDir {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to create tools.output.overflow_dir {}: {source}", path.display())]
    CreateOverflowDir {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error(transparent)]
    Sandbox(#[from] SandboxError),

    #[error("config file {} has no directory", path.display())]
    ConfigDir { path: PathBuf },

    #[error(transparent)]
    Allowlist(#[from] AllowlistError),

    #[error(
        "tools.write.writable_paths contains no resolvable directories; \
         fix ~/.config/assistd/config.toml"
    )]
    NoWritablePaths,
}

/// Subsystem handles [`build_tools`] wires into the tool registry.
pub struct BuildToolsDeps<'a> {
    pub config: &'a Config,
    /// The file `config` was loaded from. Its directory holds the
    /// allowlist approvals and is protected from commands.
    pub config_path: &'a Path,
    pub overflow_dir: PathBuf,
    pub confirmation_gate: Arc<dyn ConfirmationGate>,
    pub vision_gate: Arc<assistd_tools::VisionGate>,
    pub memory_ops: Arc<MemoryOps>,
    pub embedder: Arc<dyn Embedder>,
    pub semantic: Arc<dyn SemanticStore>,
    pub embed_tx: mpsc::Sender<EmbedJob>,
    pub embedding_model: String,
    /// Live view of the active session, so `reminisce` can leave the
    /// conversation already in context out of its results.
    pub current_session: watch::Receiver<Arc<assistd_memory::SessionId>>,
    pub window_manager: Arc<dyn WindowManager>,
    pub mcp_tools: Vec<Box<dyn assistd_tools::Tool>>,
}

/// Assemble the tool registry: the built-in commands behind `run`, the
/// memory tools, and `mcp_tools`. Clears and recreates
/// [`BuildToolsDeps::overflow_dir`] so per-process spill files land in a
/// known-empty location at every startup.
pub fn build_tools(deps: BuildToolsDeps<'_>) -> Result<Arc<ToolRegistry>, BuildToolsError> {
    let BuildToolsDeps {
        config,
        config_path,
        overflow_dir,
        confirmation_gate,
        vision_gate,
        memory_ops,
        embedder,
        semantic,
        embed_tx,
        embedding_model,
        current_session,
        window_manager,
        mcp_tools,
    } = deps;

    if overflow_dir.exists() {
        std::fs::remove_dir_all(&overflow_dir).map_err(|source| {
            BuildToolsError::ClearOverflowDir {
                path: overflow_dir.clone(),
                source,
            }
        })?;
    }
    std::fs::create_dir_all(&overflow_dir).map_err(|source| {
        BuildToolsError::CreateOverflowDir {
            path: overflow_dir.clone(),
            source,
        }
    })?;

    let sandbox_request = match config.tools.bash.sandbox {
        BashSandboxMode::Auto => SandboxRequest::Auto,
        BashSandboxMode::Bwrap => SandboxRequest::Bwrap,
        BashSandboxMode::None => SandboxRequest::None,
    };
    let config_dir = std::fs::canonicalize(config_path)
        .ok()
        .and_then(|path| path.parent().map(Path::to_path_buf))
        .ok_or_else(|| BuildToolsError::ConfigDir {
            path: config_path.to_path_buf(),
        })?;
    let protected_dirs = vec![config_dir.clone()];
    let sandbox = probe_sandbox(
        sandbox_request,
        config.tools.bash.bwrap_extra_args.clone(),
        Protected {
            dirs: protected_dirs.clone(),
            sockets: vec![assistd_ipc::socket_path()],
        },
    )?;
    let allowlist = Allowlist::load(
        config.tools.bash.allowed_programs.clone(),
        sandbox.search_path(),
        config_dir.join(APPROVALS_FILE),
    )?;

    let destructive_patterns: Vec<DestructivePattern> = config
        .tools
        .bash
        .destructive_patterns
        .iter()
        .filter_map(|raw| {
            let pattern = shlex::split(raw).and_then(DestructivePattern::new);
            if pattern.is_none() {
                warn!(
                    target: "assistd::policy",
                    pattern = %raw,
                    "tools.bash.destructive_patterns entry is not a command; dropping it"
                );
            }
            pattern
        })
        .collect();
    let bash_cfg = Arc::new(BashPolicyCfg {
        timeout: Duration::from_secs(config.tools.bash.timeout_secs.get()),
        denylist: config.tools.bash.denylist.clone(),
        destructive_patterns,
        allowlist: Arc::new(allowlist),
        protected: protected_dirs.clone(),
    });

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
                    "tools.write.writable_paths entry does not exist; dropping from allowlist"
                );
            }
        }
    }
    let write_cfg = Arc::new(
        WritePolicyCfg::new(writable_paths)
            .ok_or(BuildToolsError::NoWritablePaths)?
            .protecting(protected_dirs),
    );

    let screenshot_cfg = Arc::new(ScreenshotPolicyCfg {
        backend: match config.tools.screenshot.backend {
            ScreenshotBackend::Auto => None,
            ScreenshotBackend::X11 => Some(ScreenshotBackendKind::X11),
            ScreenshotBackend::Wayland => Some(ScreenshotBackendKind::Wayland),
        },
        timeout: ScreenshotPolicyCfg::default().timeout,
    });

    let mut commands = CommandRegistry::new();
    commands.register(CatCommand);
    commands.register(LsCommand);
    commands.register(GrepCommand);
    commands.register(WcCommand);
    commands.register(HeadCommand);
    commands.register(TailCommand);
    commands.register(SortCommand);
    commands.register(UniqCommand);
    commands.register(EchoCommand);
    commands.register(WriteCommand::new(write_cfg));
    commands.register(SeeCommand::new(vision_gate.clone()));
    commands.register(ScreenshotCommand::new(screenshot_cfg, vision_gate));
    commands.register(WebCommand::new());
    commands.register(BashCommand::new(
        bash_cfg.clone(),
        sandbox.clone(),
        confirmation_gate.clone(),
    ));
    commands.register(WmCommand::new(
        window_manager,
        bash_cfg,
        sandbox,
        confirmation_gate,
    ));

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
    tools.register(ReminisceTool::new(
        embedder,
        semantic,
        embedding_model,
        current_session,
    ));

    for t in mcp_tools {
        tools.register_boxed(t);
    }
    Ok(Arc::new(tools))
}

/// Keeps an [`assistd_tools::VisionGate`] in step with the model
/// llama-server has loaded, without rebuilding the tool registry.
///
/// Vision support can only change when weights are loaded, and the
/// daemon always loads the one configured model, so the gate is
/// re-probed only when a load may have happened since the last
/// successful probe: after any presence transition (every return to
/// `Active` reloads the weights) or a supervisor restart of the
/// llama-server child (the router then reloads the model on demand).
/// A failed probe leaves the gate as it was and is retried on the next
/// [`Self::revalidate_if_stale`], so a transient HTTP blip never flips
/// vision off mid-session.
pub struct VisionRevalidator {
    gate: Arc<assistd_tools::VisionGate>,
    control: assistd_llm::LlamaServerControl,
    model_name: String,
    seen: tokio::sync::Mutex<SeenLoad>,
}

/// The load the gate was last probed against.
struct SeenLoad {
    presence: watch::Receiver<PresenceState>,
    llama_pid: Option<u32>,
    probed: bool,
}

impl SeenLoad {
    /// Whether a probe is due: the last one failed, or the presence
    /// state or llama-server child changed since it ran. Marks both as
    /// seen either way.
    fn take_stale(&mut self, llama_pid: Option<u32>) -> bool {
        let reloaded = self.presence.has_changed().unwrap_or(false) || self.llama_pid != llama_pid;
        self.presence.mark_unchanged();
        self.llama_pid = llama_pid;
        reloaded || !self.probed
    }
}

impl VisionRevalidator {
    /// Probe `model_name` through `control` to seed the gate, then track
    /// `presence` for later loads.
    pub async fn new(
        control: assistd_llm::LlamaServerControl,
        model_name: String,
        presence: &PresenceManager,
    ) -> Arc<Self> {
        let mut seen = SeenLoad {
            presence: presence.subscribe(),
            llama_pid: presence.llama_pid().await,
            probed: false,
        };
        let initial = assistd_llm::probe_capabilities_routed(&control, &model_name).await;
        let gate = assistd_tools::VisionGate::new(initial.vision_supported);
        seen.probed = initial.model_id.is_some();
        Arc::new(Self {
            gate,
            control,
            model_name,
            seen: tokio::sync::Mutex::new(seen),
        })
    }

    /// The gate this revalidator keeps current.
    pub fn gate(&self) -> Arc<assistd_tools::VisionGate> {
        Arc::clone(&self.gate)
    }

    /// Probe llama-server now, without touching the gate.
    pub async fn probe(&self) -> assistd_llm::VisionState {
        assistd_llm::probe_capabilities_routed(&self.control, &self.model_name).await
    }

    /// Re-probe and update the gate if a load may have happened since
    /// the last successful probe; otherwise return without any I/O.
    /// Call while holding `presence` `Active`, so the probe sees the
    /// model that will serve the turn.
    pub async fn revalidate_if_stale(&self, presence: &PresenceManager) {
        let mut seen = self.seen.lock().await;
        if seen.take_stale(presence.llama_pid().await) {
            seen.probed = apply_probe(&self.gate, self.probe().await);
        }
    }
}

/// Set `gate` from a probe that reached the model, returning whether it
/// did.
fn apply_probe(gate: &assistd_tools::VisionGate, probe: assistd_llm::VisionState) -> bool {
    if probe.model_id.is_none() {
        return false;
    }
    if gate.supported() != probe.vision_supported {
        tracing::info!(
            target: "assistd::vision",
            model = ?probe.model_id,
            vision_supported = probe.vision_supported,
            "loaded model's vision support changed; updating vision gate"
        );
    }
    gate.set(probe.vision_supported);
    true
}

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

/// This crate's version string.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;
    use assistd_llm::VisionState;

    #[test]
    fn apply_probe_trusts_every_probe_that_reached_the_model() {
        let probe = |id: Option<&str>, vision| VisionState {
            model_id: id.map(str::to_string),
            vision_supported: vision,
        };
        let cases = [
            ("failed probe", true, probe(None, false), true, false),
            (
                "reload lost vision",
                true,
                probe(Some("m"), false),
                false,
                true,
            ),
            (
                "reload gained vision",
                false,
                probe(Some("m"), true),
                true,
                true,
            ),
            ("unchanged", true, probe(Some("m"), true), true, true),
        ];
        for (label, gate_initial, probe, expected_gate, expected_reached) in cases {
            let gate = assistd_tools::VisionGate::new(gate_initial);
            assert_eq!(apply_probe(&gate, probe), expected_reached, "{label}");
            assert_eq!(gate.supported(), expected_gate, "{label}");
        }
    }

    #[test]
    fn a_probe_is_due_only_after_a_failure_or_a_possible_reload() {
        let (presence_tx, presence) = watch::channel(PresenceState::Active);
        let mut seen = SeenLoad {
            presence,
            llama_pid: Some(1),
            probed: true,
        };
        assert!(!seen.take_stale(Some(1)), "nothing changed");

        presence_tx.send(PresenceState::Sleeping).unwrap();
        presence_tx.send(PresenceState::Active).unwrap();
        assert!(seen.take_stale(Some(1)), "presence round trip");
        assert!(!seen.take_stale(Some(1)), "round trip already seen");

        assert!(seen.take_stale(Some(2)), "child restarted");
        assert!(!seen.take_stale(Some(2)), "restart already seen");

        seen.probed = false;
        assert!(seen.take_stale(Some(2)), "last probe failed");
    }
}
