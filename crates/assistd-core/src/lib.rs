#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::print_stdout,
        clippy::print_stderr
    )
)]

//! Daemon orchestration: the agent loop, presence state machine, IPC
//! socket server, and the `AppState` request dispatcher. Not a stable
//! public API; the re-exports exist so the `assistd` binary can reach
//! every subsystem through one crate.

pub mod agent;
pub mod presence;
pub mod recovery;
pub mod socket;
pub mod state;

pub use agent::Agent;
pub use recovery::{Component, RecoverySeverity, install_panic_hook, spawn_supervised};

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

pub use presence::{PresenceManager, RequestGuard};
pub use state::{
    AppState, ConversationContext, McpStartupFailure, MemoryStack, RuntimeState, Subsystems,
};

use anyhow::{Context, Result};
use assistd_embed::{EmbedJob, Embedder};
use assistd_memory::SemanticStore;
use assistd_tools::{
    ConfirmationGate, MemoryOps, RecallTool, RememberTool, ReminisceTool, RunTool, SandboxRequest,
    commands::{
        BashCommand, BashPolicyCfg, CatCommand, EchoCommand, GrepCommand, HeadCommand, LsCommand,
        ScreenshotBackendKind, ScreenshotCommand, ScreenshotPolicyCfg, SeeCommand, SortCommand,
        TailCommand, UniqCommand, WcCommand, WebCommand, WmCommand, WriteCommand, WritePolicyCfg,
    },
    probe_sandbox,
};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, watch};
use tracing::warn;

/// Subsystem handles [`build_tools`] wires into the tool registry.
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
    /// Live view of the active session, so `reminisce` can leave the
    /// conversation already in context out of its results.
    pub current_session: watch::Receiver<Arc<assistd_memory::SessionId>>,
    pub window_manager: Arc<dyn WindowManager>,
    pub mcp_tools: Vec<Box<dyn assistd_tools::Tool>>,
}

/// Build the tool registry consumed by the daemon. Clears and recreates
/// [`BuildToolsDeps::overflow_dir`] so per-process spill files land in a
/// known-empty location at every startup.
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
        current_session,
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

    let destructive_patterns: Vec<Vec<String>> = config
        .tools
        .bash
        .destructive_patterns
        .iter()
        .filter_map(|p| shlex::split(p))
        .filter(|toks| !toks.is_empty())
        .collect();
    let bash_cfg = Arc::new(BashPolicyCfg {
        timeout: Duration::from_secs(config.tools.bash.timeout_secs.get()),
        denylist: config.tools.bash.denylist.clone(),
        destructive_patterns,
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
    let write_cfg = Arc::new(WritePolicyCfg::new(writable_paths).ok_or_else(|| {
        anyhow::anyhow!(
            "tools.write.writable_paths contains no resolvable directories; \
             fix ~/.config/assistd/config.toml"
        )
    })?);

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

/// Caches the running llama-server's model id alongside the
/// [`assistd_tools::VisionGate`] so a re-probe can detect a model swap
/// and flip vision availability without rebuilding the tool registry.
/// The gate only changes when the cached model id changes.
pub struct VisionRevalidator {
    gate: Arc<assistd_tools::VisionGate>,
    cached_model: tokio::sync::Mutex<Option<String>>,
    host: String,
    port: u16,
    model_name: String,
}

impl VisionRevalidator {
    /// `initial_model_id` is the model id known at startup; `None` accepts
    /// the first probe result unconditionally as the baseline.
    pub fn new(
        gate: Arc<assistd_tools::VisionGate>,
        initial_model_id: Option<String>,
        host: String,
        port: u16,
        model_name: String,
    ) -> Arc<Self> {
        Arc::new(Self {
            gate,
            cached_model: tokio::sync::Mutex::new(initial_model_id),
            host,
            port,
            model_name,
        })
    }

    /// Re-probe `/props` and, if the model id changed, update the
    /// gate. Tolerates probe failures silently; a transient HTTP
    /// blip should not flip vision off mid-session.
    pub async fn revalidate(&self) {
        let Ok(control) = assistd_llm::LlamaServerControl::new(&self.host, self.port) else {
            tracing::warn!(
                target: "assistd::vision",
                "VisionRevalidator could not build control client; skipping probe"
            );
            return;
        };
        let probe = assistd_llm::probe_capabilities_routed(
            &self.host,
            self.port,
            &self.model_name,
            &control,
        )
        .await;
        self.apply_probe(probe).await;
    }

    /// Apply a probe result to the cache and gate.
    pub async fn apply_probe(&self, probe: assistd_llm::VisionState) {
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

    pub fn gate(&self) -> &Arc<assistd_tools::VisionGate> {
        &self.gate
    }
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

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;
    use assistd_llm::VisionState;

    #[test]
    fn version_is_not_empty() {
        assert!(!version().is_empty());
    }

    fn make_revalidator(gate_initial: bool, cached_model: Option<&str>) -> Arc<VisionRevalidator> {
        VisionRevalidator::new(
            assistd_tools::VisionGate::new(gate_initial),
            cached_model.map(str::to_string),
            "127.0.0.1".to_string(),
            0,
            "test/model:Q4".to_string(),
        )
    }

    #[tokio::test]
    async fn no_model_id_keeps_gate_unchanged() {
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
        let rev = make_revalidator(false, None);
        rev.apply_probe(VisionState {
            model_id: Some("vision-model".into()),
            vision_supported: true,
        })
        .await;
        assert!(rev.gate().supported());
    }
}
