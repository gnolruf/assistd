//! Daemon orchestration: the agent loop, presence state machine, IPC
//! socket server, and the `AppState` request dispatcher. Re-exports the
//! subsystem crates so dependents need only this one.

use std::fs::{DirBuilder, DirEntry, Permissions};
use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use thiserror::Error;
use tokio::sync::{Mutex, mpsc, watch};
use tracing::{info, warn};

use assistd_embed::{EmbedJob, Embedder};
use assistd_llm::{LlamaServerControl, VisionState, probe_capabilities_routed};
use assistd_memory::{SemanticStore, SessionId};
use assistd_tools::{
    APPROVALS_FILE, APPROVED_HOSTS_FILE, APPROVED_MCP_TOOLS_FILE, Allowlist, AllowlistError,
    ApprovalGate, Approvals, ConfirmationGate, DestructivePattern, MemoryOps, Protected,
    RecallTool, RememberTool, ReminisceTool, RunTool, SandboxError, SandboxInfo, SandboxRequest,
    Tool, ToolSandbox, VisionGate,
    commands::{
        BashCommand, BashPolicyCfg, CatCommand, EchoCommand, GrepCommand, HeadCommand, LsCommand,
        ScreenshotBackendKind, ScreenshotCommand, ScreenshotPolicyCfg, SeeCommand, SortCommand,
        TailCommand, UniqCommand, WcCommand, WebCommand, WmCommand, WriteCommand, WritePolicyCfg,
    },
    probe_sandbox,
};
use assistd_utils::path::expand_tilde_from_env;

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
    ContinuousListenConfig, DaemonConfig, McpConfig, McpServerConfig, ModelConfig, PresenceConfig,
    ScreenshotBackend, SleepConfig, SynthesisConfig, ToolsBashConfig, ToolsConfig,
    ToolsOutputConfig, ToolsScreenshotConfig, ToolsWriteConfig, VoiceConfig,
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

/// Mode for the spill directory and any parents created for it.
const OVERFLOW_DIR_MODE: u32 = 0o700;

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
#[derive(Debug)]
pub struct BuildToolsDeps<'a> {
    pub config: &'a Config,
    /// The file `config` was loaded from; its directory holds the
    /// allowlist approvals and is protected from commands.
    pub config_path: &'a Path,
    pub overflow_dir: PathBuf,
    /// The sandbox from [`probe_tool_sandbox`] that subprocesses run in.
    pub sandbox: Arc<SandboxInfo>,
    pub confirmation_gate: Arc<dyn ConfirmationGate>,
    pub vision_gate: Arc<VisionGate>,
    pub memory_ops: Arc<MemoryOps>,
    pub embedder: Arc<dyn Embedder>,
    pub semantic: Arc<dyn SemanticStore>,
    pub embed_tx: mpsc::Sender<EmbedJob>,
    pub embedding_model: String,
    /// The active session, which `reminisce` excludes from its results.
    pub current_session: watch::Receiver<Arc<SessionId>>,
    pub window_manager: Arc<dyn WindowManager>,
    pub mcp_tools: Vec<Box<dyn Tool>>,
}

/// Keeps a [`VisionGate`] in step with the model llama-server has loaded.
///
/// The gate is re-probed only after a presence transition or a
/// llama-server restart, since only those reload weights. A failed probe
/// leaves the gate unchanged and is retried on the next revalidation.
#[derive(Debug)]
pub struct VisionRevalidator {
    gate: Arc<VisionGate>,
    control: LlamaServerControl,
    model_name: String,
    seen: Mutex<SeenLoad>,
}

impl VisionRevalidator {
    /// Probe `model_name` through `control` to seed the gate, then track
    /// `presence` for later loads.
    pub async fn new(
        control: LlamaServerControl,
        model_name: String,
        presence: &PresenceManager,
    ) -> Arc<Self> {
        let mut seen = SeenLoad {
            presence: presence.subscribe(),
            llama_pid: presence.llama_pid().await,
            probed: false,
        };
        let initial = probe_capabilities_routed(&control, &model_name).await;
        let gate = VisionGate::new(initial.vision_supported);
        seen.probed = initial.model_id.is_some();
        Arc::new(Self {
            gate,
            control,
            model_name,
            seen: Mutex::new(seen),
        })
    }

    /// The gate this revalidator keeps current.
    pub fn gate(&self) -> Arc<VisionGate> {
        Arc::clone(&self.gate)
    }

    /// Probe llama-server now, without touching the gate.
    pub async fn probe(&self) -> VisionState {
        probe_capabilities_routed(&self.control, &self.model_name).await
    }

    /// Re-probe and update the gate if a load may have happened since the
    /// last successful probe. Call while holding presence `Active`.
    pub async fn revalidate_if_stale(&self, presence: &PresenceManager) {
        let mut seen = self.seen.lock().await;
        if seen.take_stale(presence.llama_pid().await) {
            seen.probed = apply_probe(&self.gate, &self.probe().await);
        }
    }
}

/// The load the gate was last probed against.
#[derive(Debug)]
struct SeenLoad {
    presence: watch::Receiver<PresenceState>,
    llama_pid: Option<u32>,
    probed: bool,
}

impl SeenLoad {
    /// Whether a probe is due: the last one failed, or the presence state
    /// or llama-server child changed since. Marks both as seen either way.
    fn take_stale(&mut self, llama_pid: Option<u32>) -> bool {
        let reloaded = self.presence.has_changed().unwrap_or(false) || self.llama_pid != llama_pid;
        self.presence.mark_unchanged();
        self.llama_pid = llama_pid;
        reloaded || !self.probed
    }
}

/// The sandbox the model's tools run in under `config`, keeping the
/// directory of `config_path` read-only, or why the model gets no tools.
///
/// # Errors
/// [`BuildToolsError`] when the config directory cannot be resolved or a
/// required `bwrap` is missing.
pub fn probe_tool_sandbox(
    config: &Config,
    config_path: &Path,
) -> Result<ToolSandbox, BuildToolsError> {
    let config_dir = canonical_config_dir(config_path)?;
    Ok(probe_sandbox(
        sandbox_request(config.tools.bash.sandbox),
        config.tools.bash.bwrap_extra_args.clone(),
        Protected {
            dirs: vec![config_dir],
            sockets: vec![assistd_ipc::socket_path()],
        },
    )?)
}

/// The MCP tools approved with "always allow", kept beside the config file.
///
/// # Errors
/// [`BuildToolsError`] when the config directory cannot be resolved or the
/// approvals file cannot be read.
pub fn mcp_tool_approvals(config_path: &Path) -> Result<Approvals, BuildToolsError> {
    let config_dir = canonical_config_dir(config_path)?;
    Ok(Approvals::load(config_dir.join(APPROVED_MCP_TOOLS_FILE))?)
}

/// Assemble the tool registry: the built-in commands behind `run`, the
/// memory tools, and `mcp_tools`. Removes earlier spill files from
/// [`BuildToolsDeps::overflow_dir`].
pub fn build_tools(deps: BuildToolsDeps<'_>) -> Result<Arc<ToolRegistry>, BuildToolsError> {
    let BuildToolsDeps {
        config,
        config_path,
        overflow_dir,
        sandbox,
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

    clear_overflow_dir(&overflow_dir)?;
    let config_dir = canonical_config_dir(config_path)?;
    let protected_dirs = vec![config_dir.clone()];
    let allowlist = Allowlist::load(
        config.tools.bash.allowed_programs.clone(),
        sandbox.search_path(),
        config_dir.join(APPROVALS_FILE),
    )?;
    let bash_cfg = Arc::new(BashPolicyCfg {
        timeout: Duration::from_secs(config.tools.bash.timeout_secs.get()),
        denylist: config.tools.bash.denylist.clone(),
        destructive_patterns: parse_destructive_patterns(&config.tools.bash.destructive_patterns),
        allowlist: Arc::new(allowlist),
        protected: protected_dirs.clone(),
    });
    let write_cfg = Arc::new(
        WritePolicyCfg::new(resolve_writable_paths(&config.tools.write.writable_paths))
            .ok_or(BuildToolsError::NoWritablePaths)?
            .protecting(protected_dirs),
    );

    let commands = builtin_commands(BuiltinCommandDeps {
        bash_cfg,
        write_cfg,
        approved_hosts: Arc::new(Approvals::load(config_dir.join(APPROVED_HOSTS_FILE))?),
        screenshot_cfg: Arc::new(screenshot_policy(config.tools.screenshot.backend)),
        sandbox,
        confirmation_gate,
        vision_gate,
        window_manager,
    });

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
    for tool in mcp_tools {
        tools.register_boxed(tool);
    }
    Ok(Arc::new(tools))
}

struct BuiltinCommandDeps {
    bash_cfg: Arc<BashPolicyCfg>,
    write_cfg: Arc<WritePolicyCfg>,
    approved_hosts: Arc<Approvals>,
    screenshot_cfg: Arc<ScreenshotPolicyCfg>,
    sandbox: Arc<SandboxInfo>,
    confirmation_gate: Arc<dyn ConfirmationGate>,
    vision_gate: Arc<VisionGate>,
    window_manager: Arc<dyn WindowManager>,
}

fn builtin_commands(deps: BuiltinCommandDeps) -> CommandRegistry {
    let BuiltinCommandDeps {
        bash_cfg,
        write_cfg,
        approved_hosts,
        screenshot_cfg,
        sandbox,
        confirmation_gate,
        vision_gate,
        window_manager,
    } = deps;

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
    commands.register(WriteCommand::new(write_cfg, confirmation_gate.clone()));
    commands.register(SeeCommand::new(vision_gate.clone()));
    commands.register(ScreenshotCommand::new(screenshot_cfg, vision_gate));
    commands.register(WebCommand::new(ApprovalGate::new(
        confirmation_gate.clone(),
        approved_hosts,
    )));
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
    commands
}

fn clear_overflow_dir(overflow_dir: &Path) -> Result<(), BuildToolsError> {
    let create_error = |source| BuildToolsError::CreateOverflowDir {
        path: overflow_dir.to_path_buf(),
        source,
    };
    let clear_error = |source| BuildToolsError::ClearOverflowDir {
        path: overflow_dir.to_path_buf(),
        source,
    };
    DirBuilder::new()
        .recursive(true)
        .mode(OVERFLOW_DIR_MODE)
        .create(overflow_dir)
        .map_err(create_error)?;
    std::fs::set_permissions(overflow_dir, Permissions::from_mode(OVERFLOW_DIR_MODE))
        .map_err(create_error)?;
    for entry in std::fs::read_dir(overflow_dir).map_err(clear_error)? {
        let entry = entry.map_err(clear_error)?;
        if is_spill_entry(&entry) {
            std::fs::remove_file(entry.path()).map_err(clear_error)?;
        }
    }
    Ok(())
}

fn is_spill_entry(entry: &DirEntry) -> bool {
    let named_like_spill = entry.file_name().to_str().is_some_and(|name| {
        name.strip_suffix(".txt")
            .and_then(|stem| stem.rsplit_once('-'))
            .is_some_and(|(prefix, n)| {
                !prefix.is_empty() && !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit())
            })
    });
    named_like_spill && entry.file_type().is_ok_and(|kind| !kind.is_dir())
}

fn canonical_config_dir(config_path: &Path) -> Result<PathBuf, BuildToolsError> {
    std::fs::canonicalize(config_path)
        .ok()
        .and_then(|path| path.parent().map(Path::to_path_buf))
        .ok_or_else(|| BuildToolsError::ConfigDir {
            path: config_path.to_path_buf(),
        })
}

fn sandbox_request(mode: BashSandboxMode) -> SandboxRequest {
    match mode {
        BashSandboxMode::Auto => SandboxRequest::Auto,
        BashSandboxMode::Bwrap => SandboxRequest::Bwrap,
        BashSandboxMode::None => SandboxRequest::None,
    }
}

fn parse_destructive_patterns(raw_patterns: &[String]) -> Vec<DestructivePattern> {
    raw_patterns
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
        .collect()
}

/// Canonicalize each configured writable path, dropping (with a warning)
/// those that do not exist.
fn resolve_writable_paths(raw_paths: &[String]) -> Vec<PathBuf> {
    raw_paths
        .iter()
        .filter_map(|raw| {
            let expanded = expand_tilde_from_env(raw);
            std::fs::canonicalize(&expanded)
                .inspect_err(|e| {
                    warn!(
                        target: "assistd::policy",
                        path = %expanded.display(),
                        error = %e,
                        "tools.write.writable_paths entry does not exist; dropping from allowlist"
                    );
                })
                .ok()
        })
        .collect()
}

fn screenshot_policy(backend: ScreenshotBackend) -> ScreenshotPolicyCfg {
    ScreenshotPolicyCfg {
        backend: match backend {
            ScreenshotBackend::Auto => None,
            ScreenshotBackend::X11 => Some(ScreenshotBackendKind::X11),
            ScreenshotBackend::Wayland => Some(ScreenshotBackendKind::Wayland),
        },
        timeout: ScreenshotPolicyCfg::default().timeout,
    }
}

/// Set `gate` from a probe that reached the model, returning whether it
/// did.
fn apply_probe(gate: &VisionGate, probe: &VisionState) -> bool {
    if probe.model_id.is_none() {
        return false;
    }
    if gate.supported() != probe.vision_supported {
        info!(
            target: "assistd::vision",
            model = ?probe.model_id,
            vision_supported = probe.vision_supported,
            "loaded model's vision support changed; updating vision gate"
        );
    }
    gate.set(probe.vision_supported);
    true
}

/// This crate's version string.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

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
            let gate = VisionGate::new(gate_initial);
            assert_eq!(apply_probe(&gate, &probe), expected_reached, "{label}");
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

    #[test]
    fn clear_overflow_dir_removes_only_spill_files_and_restricts_the_dir() {
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path().join("assistd/output");
        std::fs::create_dir_all(dir.join("keep-1.txt")).unwrap();
        for spill in ["cmd-1.txt", "mcp-files-12.txt"] {
            std::fs::write(dir.join(spill), b"stale").unwrap();
        }
        for kept in ["notes.txt", "cmd-.txt", "-3.txt", "cmd-1.md", "cmd-x.txt"] {
            std::fs::write(dir.join(kept), b"user data").unwrap();
        }
        std::os::unix::fs::symlink("/etc/passwd", dir.join("cmd-2.txt")).unwrap();
        std::fs::set_permissions(&dir, Permissions::from_mode(0o755)).unwrap();

        clear_overflow_dir(&dir).unwrap();

        let mut left: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        left.sort();
        assert_eq!(
            left,
            [
                "-3.txt",
                "cmd-.txt",
                "cmd-1.md",
                "cmd-x.txt",
                "keep-1.txt",
                "notes.txt"
            ]
        );
        let mode = std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, OVERFLOW_DIR_MODE);
    }

    #[test]
    fn clear_overflow_dir_creates_a_missing_dir_owner_only() {
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path().join("assistd/output");

        clear_overflow_dir(&dir).unwrap();

        let mode = std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, OVERFLOW_DIR_MODE);
    }
}
