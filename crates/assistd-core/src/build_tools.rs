//! `build_tools`: assembles the tool registries from config and the subsystem handles.

use std::fs::{DirBuilder, DirEntry, Permissions};
use std::net::SocketAddr;
use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use thiserror::Error;
use tokio::sync::{Notify, watch};
use tracing::warn;

use assistd_config::{BashSandboxMode, Config, NoDefaultDirError, WritablePath};
use assistd_embed::EmbedderHandle;
use assistd_memory::{SemanticStore, SessionId};
use assistd_search::{SearchError, SearxngClient};
use assistd_tools::{
    APPROVALS_FILE, APPROVED_APPS_FILE, APPROVED_HOSTS_FILE, APPROVED_MCP_TOOLS_FILE, ActionCheck,
    Allowlist, AllowlistError, ApprovalGate, Approvals, CommandRegistry, ConfirmationGate,
    MemoryOps, OutputCap, PresentSpec, Protected, RememberTool, ReminisceTool, RunTool,
    SandboxError, SandboxInfo, SandboxRequest, SharedDirs, StandingApprovals, ToolRegistry,
    ToolSandbox, VisionGate,
    commands::{
        BashCommand, BashPolicyCfg, ScreenshotCommand, ScreenshotPolicyCfg, SeeCommand, WebCommand,
        WmCommand, WriteCommand, WritePolicyCfg, register_readers,
    },
    presentation::is_spill_file_name,
    probe_sandbox,
};
use assistd_utils::child_server::ChildServerStatus;
use assistd_utils::path::expand_tilde_from_env;
use assistd_utils::readiness::ReadinessCell;
use assistd_wm::WindowManager;

use crate::planner::{PlanModeTool, PlanningTools};
use crate::relay::{RelayDeps, RelayPolicy, RelayTool};
use crate::scratch;

/// Mode for the spill and scratch directories and any parents created for
/// them.
pub(crate) const OVERFLOW_DIR_MODE: u32 = 0o700;

/// Why [`build_tools`] could not assemble the tool registry.
#[derive(Debug, Error)]
pub enum BuildToolsError {
    #[error(transparent)]
    NoDefaultDir(#[from] NoDefaultDirError),

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

    #[error("failed to create tools.scratch.dir {}: {source}", path.display())]
    CreateScratchDir {
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

    #[error(transparent)]
    SearchClient(#[from] SearchError),
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
    pub embedder: Arc<EmbedderHandle>,
    pub semantic: Arc<dyn SemanticStore>,
    /// Wakes the embedding worker after `remember` saves a value.
    pub embed_wake: Arc<Notify>,
    /// The active session, whose messages still in context `reminisce` excludes.
    pub current_session: watch::Receiver<SessionId>,
    pub window_manager: Arc<dyn WindowManager>,
    /// Readiness of the SearXNG instance `web search` goes through.
    pub web_search: Arc<ReadinessCell<ChildServerStatus>>,
    pub relay: RelayDeps,
    /// The MCP tools approved with "always allow", which a plan's actions are checked against.
    pub mcp_approvals: ApprovalGate,
}

/// The registries [`build_tools`] assembles.
#[derive(Debug)]
pub struct BuiltTools {
    /// What ordinary turns are offered.
    pub registry: Arc<ToolRegistry>,
    pub planning: PlanningTools,
}

/// The tool sandbox for `config` (config dir read-only, scratch and spill dirs shared), or
/// why the model gets no tools. Creates the scratch dir; errors if it or the config dir
/// cannot be set up, or a required `bwrap` is missing.
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
        SharedDirs {
            scratch: Some(scratch::prepare(&config.tools.scratch)?),
            read_only: vec![config.tools.output.resolved_overflow_dir()?],
        },
    )?)
}

/// The MCP tools approved with "always allow", kept beside the config file. Errors when
/// the config directory cannot be resolved or the approvals file cannot be read.
pub fn mcp_tool_approvals(config_path: &Path) -> Result<Approvals, BuildToolsError> {
    let config_dir = canonical_config_dir(config_path)?;
    Ok(Approvals::load(config_dir.join(APPROVED_MCP_TOOLS_FILE))?)
}

/// Assemble the tool registry: the built-in commands behind `run`, `relay` and, with
/// `memory.enabled`, the memory tools; and beside it the planning registry. Removes earlier
/// spill files from [`BuildToolsDeps::overflow_dir`].
pub fn build_tools(deps: BuildToolsDeps<'_>) -> Result<BuiltTools, BuildToolsError> {
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
        embed_wake,
        current_session,
        window_manager,
        web_search,
        relay,
        mcp_approvals,
    } = deps;

    clear_overflow_dir(&overflow_dir)?;
    let config_dir = canonical_config_dir(config_path)?;
    let scratch_dir = sandbox.shared.scratch.clone();
    let (bash_cfg, write_cfg) = file_policies(config, &config_dir, &sandbox)?;

    let (search, approved_hosts) = web_access(config, &config_dir, &confirmation_gate, web_search)?;
    let (approved_apps, checks) = action_checks(
        &config_dir,
        &sandbox,
        &bash_cfg,
        &approved_hosts,
        mcp_approvals,
    )?;
    let output_cap = Arc::new(OutputCap::new(PresentSpec::from_config(
        &config.tools.output,
        overflow_dir,
    )));
    let reminisce = config.memory.enabled.then(|| {
        ReminisceTool::new(
            embedder.clone(),
            semantic.clone(),
            Arc::clone(&memory_ops.transcripts),
            current_session.clone(),
        )
    });
    let planning = planning_tools(PlanningToolDeps {
        commands: planning_commands(PlanningCommandDeps {
            bash_cfg: Arc::clone(&bash_cfg),
            approved_hosts: approved_hosts.clone(),
            sandbox: Arc::clone(&sandbox),
            confirmation_gate: Arc::clone(&confirmation_gate),
            search: search.clone(),
        }),
        output_cap: Arc::clone(&output_cap),
        scratch_dir: scratch_dir.as_deref(),
        reminisce,
        checks,
    });
    let relay = RelayTool::new(
        relay,
        RelayPolicy {
            sandbox: Arc::clone(&sandbox),
            bash: Arc::clone(&bash_cfg),
            gate: Arc::clone(&confirmation_gate),
            hosts: approved_hosts.clone(),
            search: search.clone(),
            output: config.tools.output.clone(),
        },
    );
    let commands = builtin_commands(BuiltinCommandDeps {
        bash_cfg,
        write_cfg,
        approved_hosts,
        approved_apps,
        screenshot_cfg: Arc::new(ScreenshotPolicyCfg {
            compositor: config.compositor.resolve(),
            ..ScreenshotPolicyCfg::default()
        }),
        sandbox,
        confirmation_gate,
        vision_gate,
        window_manager,
        search,
    });

    let run = RunTool::new(Arc::new(commands), Arc::clone(&output_cap));
    let mut tools = ToolRegistry::with_output_cap(output_cap);
    tools.register(match &scratch_dir {
        Some(scratch_dir) => run.with_scratch_dir(scratch_dir),
        None => run,
    });
    tools.register(relay);
    if config.memory.enabled {
        let memory = (memory_ops, embed_wake);
        register_memory_tools(&mut tools, memory, embedder, semantic, current_session);
    }
    Ok(BuiltTools {
        registry: Arc::new(tools),
        planning,
    })
}

/// Register `remember` over `memory`'s store and embedding wake signal, and `reminisce`.
fn register_memory_tools(
    tools: &mut ToolRegistry,
    (memory_ops, embed_wake): (Arc<MemoryOps>, Arc<Notify>),
    embedder: Arc<EmbedderHandle>,
    semantic: Arc<dyn SemanticStore>,
    current_session: watch::Receiver<SessionId>,
) {
    let transcripts = Arc::clone(&memory_ops.transcripts);
    tools.register(RememberTool::new(
        memory_ops,
        embed_wake,
        embedder.clone(),
        semantic.clone(),
    ));
    tools.register(ReminisceTool::new(
        embedder,
        semantic,
        transcripts,
        current_session,
    ));
}

/// The approved apps, and the checks a plan's actions face: the sandbox's programs, and the
/// standing approvals and edit scope that let an action run unasked.
fn action_checks(
    config_dir: &Path,
    sandbox: &SandboxInfo,
    bash_cfg: &BashPolicyCfg,
    hosts: &ApprovalGate,
    mcp_tools: ApprovalGate,
) -> Result<(Arc<Approvals>, ActionCheck), BuildToolsError> {
    let apps = Arc::new(Approvals::load(config_dir.join(APPROVED_APPS_FILE))?);
    let standing = StandingApprovals {
        programs: Arc::clone(&bash_cfg.allowlist),
        hosts: hosts.clone(),
        apps: Arc::clone(&apps),
        mcp_tools,
        edit_scope: bash_cfg.edit_scope.clone(),
    };
    Ok((
        apps,
        ActionCheck::new(sandbox.search_path(), Some(standing)),
    ))
}

/// The `bash` and `write` policies: `config_dir` protected, and the
/// sandbox's scratch directory writable.
fn file_policies(
    config: &Config,
    config_dir: &Path,
    sandbox: &SandboxInfo,
) -> Result<(Arc<BashPolicyCfg>, Arc<WritePolicyCfg>), BuildToolsError> {
    let protected_dirs = vec![config_dir.to_path_buf()];
    let allowlist = Allowlist::load(sandbox.search_path(), config_dir.join(APPROVALS_FILE))?;
    let scratch_dir = sandbox.shared.scratch.clone();
    let writable_paths: Vec<PathBuf> =
        resolve_writable_paths(config.tools.write.writable_paths.as_slice())
            .into_iter()
            .chain(scratch_dir.clone())
            .collect();
    let bash_cfg = Arc::new(BashPolicyCfg {
        timeout: Duration::from_secs(config.tools.bash.timeout_secs.get()),
        allowlist: Arc::new(allowlist),
        protected: protected_dirs.clone(),
        scratch: scratch_dir,
        edit_scope: writable_paths.clone(),
    });
    let write_cfg = Arc::new(
        WritePolicyCfg::new(writable_paths)
            .ok_or(BuildToolsError::NoWritablePaths)?
            .protecting(protected_dirs),
    );
    Ok((bash_cfg, write_cfg))
}

/// The SearXNG client `web search` uses, if search is on, and the gate for web hosts.
fn web_access(
    config: &Config,
    config_dir: &Path,
    gate: &Arc<dyn ConfirmationGate>,
    web_search: Arc<ReadinessCell<ChildServerStatus>>,
) -> Result<(Option<SearxngClient>, ApprovalGate), BuildToolsError> {
    let search = config
        .search
        .enabled
        .then(|| {
            let addr = SocketAddr::new(config.search.host.get(), config.search.port.get());
            SearxngClient::new(addr, web_search)
        })
        .transpose()?;
    let approved_hosts = ApprovalGate::new(
        Arc::clone(gate),
        Arc::new(Approvals::load(config_dir.join(APPROVED_HOSTS_FILE))?),
    );
    Ok((search, approved_hosts))
}

pub(crate) struct PlanningToolDeps<'a> {
    pub(crate) commands: CommandRegistry,
    pub(crate) output_cap: Arc<OutputCap>,
    pub(crate) scratch_dir: Option<&'a Path>,
    pub(crate) reminisce: Option<ReminisceTool>,
    pub(crate) checks: ActionCheck,
}

/// `run` over `commands` and, with memory on, `reminisce`, each reporting refusals as
/// [`PlanModeTool`] does.
pub(crate) fn planning_tools(deps: PlanningToolDeps<'_>) -> PlanningTools {
    let PlanningToolDeps {
        commands,
        output_cap,
        scratch_dir,
        reminisce,
        checks,
    } = deps;
    let run = RunTool::new(Arc::new(commands), Arc::clone(&output_cap));
    let mut registry = ToolRegistry::with_output_cap(output_cap);
    registry.register(PlanModeTool::new(match scratch_dir {
        Some(scratch_dir) => run.with_scratch_dir(scratch_dir),
        None => run,
    }));
    if let Some(reminisce) = reminisce {
        registry.register(PlanModeTool::new(reminisce));
    }
    PlanningTools {
        registry: Arc::new(registry),
        checks,
    }
}

pub(crate) struct PlanningCommandDeps {
    pub(crate) bash_cfg: Arc<BashPolicyCfg>,
    pub(crate) approved_hosts: ApprovalGate,
    pub(crate) sandbox: Arc<SandboxInfo>,
    pub(crate) confirmation_gate: Arc<dyn ConfirmationGate>,
    pub(crate) search: Option<SearxngClient>,
}

/// The commands of a planning turn: the readers, `web`, and `bash`, with no "always allow"
/// approvals, so a script or text program run unasked is read-only by its own class.
pub(crate) fn planning_commands(deps: PlanningCommandDeps) -> CommandRegistry {
    let PlanningCommandDeps {
        bash_cfg,
        approved_hosts,
        sandbox,
        confirmation_gate,
        search,
    } = deps;
    let unapproved = BashPolicyCfg {
        allowlist: Arc::new(Allowlist::unsaved(sandbox.search_path())),
        ..BashPolicyCfg::clone(&bash_cfg)
    };
    let unapproved = Arc::new(unapproved);
    let mut commands = CommandRegistry::new();
    register_readers(&mut commands, &unapproved, &sandbox, &confirmation_gate);
    commands.register(WebCommand::new(approved_hosts, search));
    commands.register(BashCommand::new(unapproved, sandbox, confirmation_gate));
    commands
}

struct BuiltinCommandDeps {
    bash_cfg: Arc<BashPolicyCfg>,
    write_cfg: Arc<WritePolicyCfg>,
    approved_hosts: ApprovalGate,
    approved_apps: Arc<Approvals>,
    screenshot_cfg: Arc<ScreenshotPolicyCfg>,
    sandbox: Arc<SandboxInfo>,
    confirmation_gate: Arc<dyn ConfirmationGate>,
    vision_gate: Arc<VisionGate>,
    window_manager: Arc<dyn WindowManager>,
    search: Option<SearxngClient>,
}

fn builtin_commands(deps: BuiltinCommandDeps) -> CommandRegistry {
    let BuiltinCommandDeps {
        bash_cfg,
        write_cfg,
        approved_hosts,
        approved_apps,
        screenshot_cfg,
        sandbox,
        confirmation_gate,
        vision_gate,
        window_manager,
        search,
    } = deps;

    let mut commands = CommandRegistry::new();
    register_readers(&mut commands, &bash_cfg, &sandbox, &confirmation_gate);
    commands.register(WriteCommand::new(
        write_cfg,
        confirmation_gate.clone(),
        sandbox.clone(),
    ));
    commands.register(SeeCommand::new(vision_gate.clone()));
    commands.register(ScreenshotCommand::new(screenshot_cfg, vision_gate));
    commands.register(WebCommand::new(approved_hosts, search));
    commands.register(BashCommand::new(
        bash_cfg.clone(),
        sandbox.clone(),
        confirmation_gate.clone(),
    ));
    commands.register(WmCommand::new(
        window_manager,
        confirmation_gate,
        approved_apps,
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
    let named_like_spill = entry.file_name().to_str().is_some_and(is_spill_file_name);
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

/// Canonicalize each configured writable path, dropping (with a warning)
/// those that do not exist.
fn resolve_writable_paths(raw_paths: &[WritablePath]) -> Vec<PathBuf> {
    raw_paths
        .iter()
        .filter_map(|raw| {
            let expanded = expand_tilde_from_env(raw.as_str());
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clear_overflow_dir_removes_only_spill_files_and_restricts_the_dir() {
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path().join("assistd/output");
        std::fs::create_dir_all(dir.join("keep-1.txt")).unwrap();
        for spill in ["cmd-1.txt", "mcp-files-12.txt"] {
            std::fs::write(dir.join(spill), b"stale").unwrap();
        }
        std::fs::write(dir.join("notes.txt"), b"user data").unwrap();
        std::os::unix::fs::symlink("/etc/passwd", dir.join("cmd-2.txt")).unwrap();
        std::fs::set_permissions(&dir, Permissions::from_mode(0o755)).unwrap();

        clear_overflow_dir(&dir).unwrap();

        let mut left: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        left.sort();
        assert_eq!(left, ["keep-1.txt", "notes.txt"]);
        let mode = std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, OVERFLOW_DIR_MODE);
    }
}
