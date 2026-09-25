//! Command-execution policy: confirmation gates, command review, and
//! sandbox probing.

use std::collections::HashMap;
use std::ffi::OsStr;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;

use async_trait::async_trait;
use tokio::process::Command as ProcCommand;
use tokio::sync::{mpsc, oneshot};
use tracing::{info, warn};

use assistd_ipc::Event;

use crate::command::{CommandOutput, Hint, error_line};
use crate::exec::POLICY_DENIED_EXIT;

mod allowlist;
mod review;
mod shell;

pub use allowlist::{APPROVALS_FILE, Allowlist, AllowlistError, SearchPath};
pub use review::{Confirmation, DestructivePattern, Rules, check_argv, check_script};

/// Policy for the commands that spawn subprocesses.
///
/// A command runs without asking only when [`check_script`] finds every
/// program it can run on the allowlist and nothing destructive in it. The
/// denylist refuses outright: a literal backstop for obviously dangerous
/// invocations (`rm -rf /`, `mkfs`, …). Neither sees what an allowed
/// program does once running; the bwrap sandbox limits that.
#[derive(Debug, Clone)]
pub struct BashPolicyCfg {
    pub timeout: Duration,
    pub denylist: Vec<String>,
    pub destructive_patterns: Vec<DestructivePattern>,
    /// Programs that run without confirmation, shared with every command
    /// that spawns subprocesses so an approval applies to all of them.
    pub allowlist: Arc<Allowlist>,
    /// Directories a command may not name without confirmation:
    /// assistd's own configuration.
    pub protected: Vec<PathBuf>,
}

impl Default for BashPolicyCfg {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            denylist: Vec::new(),
            destructive_patterns: Vec::new(),
            allowlist: Arc::new(Allowlist::unsaved(
                assistd_config::defaults::default_bash_allowed_programs(),
                SandboxInfo::none().search_path(),
            )),
            protected: Vec::new(),
        }
    }
}

impl BashPolicyCfg {
    /// The rules [`check_script`] and [`check_argv`] apply.
    pub fn rules(&self) -> Rules<'_> {
        Rules {
            patterns: &self.destructive_patterns,
            allowlist: &self.allowlist,
            protected: &self.protected,
        }
    }
}

/// Everything a command needs to run model-chosen argv: the policy,
/// the sandbox to wrap it in, and the gate that confirms what the policy
/// does not let through on its own.
pub(crate) struct SubprocessPolicy {
    pub(crate) cfg: Arc<BashPolicyCfg>,
    pub(crate) sandbox: Arc<SandboxInfo>,
    pub(crate) gate: Arc<dyn ConfirmationGate>,
}

impl SubprocessPolicy {
    /// Refuse `script` when it hits the denylist, or when it needs
    /// confirmation and the gate declines. An "always" answer adds the
    /// programs the prompt offered to the allowlist. `tool` and `op` name
    /// the command and operation in the error line. `confirmation` is the
    /// review's verdict, computed by the command because what it reviews
    /// (a script, an argv) differs per command.
    pub(crate) async fn authorize(
        &self,
        tool: &str,
        op: &str,
        script: &str,
        confirmation: Option<Confirmation>,
    ) -> Result<(), CommandOutput> {
        if let Some(pat) = matches_denylist(script, &self.cfg.denylist) {
            warn!(
                target: "assistd::policy",
                tool = %tool,
                script = %script,
                matched = %pat,
                "denied by denylist"
            );
            return Err(CommandOutput::failed(
                POLICY_DENIED_EXIT,
                error_line(
                    tool,
                    format_args!("{op} denied by policy. Matched denylist pattern: {pat}"),
                    Hint::Try,
                    "a non-destructive alternative",
                )
                .into_bytes(),
            ));
        }
        let Some(confirmation) = confirmation else {
            return Ok(());
        };
        let offered = confirmation.always_allow().to_vec();
        let approval = self
            .gate
            .confirm(ConfirmationRequest {
                tool: tool.to_string(),
                script: script.to_string(),
                matched_pattern: confirmation.to_string(),
                always_allow: offered.clone(),
            })
            .await;
        match approval {
            Approval::Always if !offered.is_empty() => {
                if let Err(e) = self.cfg.allowlist.approve(&offered).await {
                    warn!(
                        target: "assistd::policy",
                        error = %e,
                        "approval holds until the daemon exits but was not saved"
                    );
                }
                return Ok(());
            }
            Approval::Once | Approval::Always => return Ok(()),
            Approval::Deny => {}
        }
        let reason = match &confirmation {
            Confirmation::Pattern(pattern) => format!("Matched destructive pattern: {pattern}"),
            Confirmation::Unverifiable(why) => {
                format!("Could not rule out a destructive command: {why}")
            }
            Confirmation::Unlisted { programs, .. } => {
                format!("Not on the allowlist: {}", programs.join(", "))
            }
        };
        Err(CommandOutput::failed(
            POLICY_DENIED_EXIT,
            error_line(
                tool,
                format_args!("{op} cancelled by user. {reason}"),
                Hint::Try,
                "a different approach",
            )
            .into_bytes(),
        ))
    }
}

/// Describes a request for the user's confirmation before a command
/// runs. Passed to [`ConfirmationGate::confirm`].
#[derive(Debug, Clone)]
pub struct ConfirmationRequest {
    /// Tool name requesting confirmation (e.g. `"bash"`).
    pub tool: String,
    /// Verbatim script the tool is about to execute.
    pub script: String,
    /// Why the command needs confirmation, for display: the destructive
    /// pattern it matches (`"rm -rf"`), the programs not on the
    /// allowlist, or why it could not be checked.
    pub matched_pattern: String,
    /// Programs an [`Approval::Always`] answer adds to the allowlist.
    /// Empty when the prompt cannot be settled that way.
    pub always_allow: Vec<String>,
}

/// The user's answer to a [`ConfirmationRequest`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Approval {
    /// Do not run the command.
    Deny,
    /// Run the command this once.
    Once,
    /// Run the command, and add the request's `always_allow` programs to
    /// the allowlist. The same as [`Approval::Once`] when it offered none.
    Always,
}

impl Approval {
    /// The approval an IPC client's answer stands for.
    pub fn from_answer(allow: bool, always: bool) -> Self {
        match (allow, always) {
            (false, _) => Self::Deny,
            (true, false) => Self::Once,
            (true, true) => Self::Always,
        }
    }
}

/// Decides whether a command that needs confirmation may run.
#[async_trait]
pub trait ConfirmationGate: Send + Sync + 'static {
    /// Ask for confirmation.
    ///
    /// Implementations must convert *every* failure mode (channel drop, UI
    /// shutdown, timeout) into [`Approval::Deny`] so a turn never hangs.
    async fn confirm(&self, req: ConfirmationRequest) -> Approval;
}

/// Gate that never approves, logging each denial.
#[cfg(any(test, feature = "test-support"))]
#[derive(Debug, Default)]
pub struct DenyAllGate;

#[cfg(any(test, feature = "test-support"))]
#[async_trait]
impl ConfirmationGate for DenyAllGate {
    async fn confirm(&self, req: ConfirmationRequest) -> Approval {
        warn!(
            target: "assistd::policy",
            tool = %req.tool,
            pattern = %req.matched_pattern,
            "command denied: no interactive confirmation gate attached"
        );
        Approval::Deny
    }
}

/// Gate that approves every request once, defeating the confirmation
/// layer.
#[cfg(any(test, feature = "test-support"))]
#[derive(Debug, Default)]
pub struct AlwaysAllowGate;

#[cfg(any(test, feature = "test-support"))]
#[async_trait]
impl ConfirmationGate for AlwaysAllowGate {
    async fn confirm(&self, _req: ConfirmationRequest) -> Approval {
        Approval::Once
    }
}

/// Cap on confirmation prompts in flight on one connection. A client
/// that never answers would otherwise leak a `oneshot::Sender` per ask.
pub const MAX_PENDING_CONFIRMS: usize = 32;

/// How long a prompt waits for the client's answer before it is denied.
pub const CONFIRM_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Default)]
struct PendingPrompts {
    /// Set once the client can no longer answer; every later ask is
    /// denied without touching the wire.
    closed: bool,
    prompts: HashMap<String, oneshot::Sender<Approval>>,
}

/// A confirmation answer named a `confirm_id` with no prompt in flight:
/// it never existed, was already answered, or was denied on timeout.
#[derive(Debug, thiserror::Error)]
#[error("no pending confirm for this confirm_id")]
pub struct NoPendingConfirm;

/// Per-connection routing table for in-flight confirmation prompts,
/// installed in the [`CONFIRM_ROUTER`] task-local so
/// [`IpcConfirmationGate`] can find it without plumbing. Each `ask`
/// gets a fresh `confirm_id`, and beyond [`MAX_PENDING_CONFIRMS`] in
/// flight further asks are denied rather than queued. An answer only
/// reaches a prompt asked on the same connection.
pub struct ConfirmRouter {
    /// Id of the connection's originating request, carried on every
    /// emitted [`Event::ConfirmRequest`].
    request_id: String,
    wire: mpsc::Sender<Event>,
    timeout: Duration,
    pending: Mutex<PendingPrompts>,
}

impl ConfirmRouter {
    /// A router whose prompts are denied after `timeout` without an
    /// answer.
    pub fn new(request_id: String, wire: mpsc::Sender<Event>, timeout: Duration) -> Arc<Self> {
        Arc::new(Self {
            request_id,
            wire,
            timeout,
            pending: Mutex::new(PendingPrompts::default()),
        })
    }

    /// Forward the prompt to the connected client and await the answer.
    /// Every failure mode (channel drop, cap reached, disconnect, closed
    /// router, timeout) is [`Approval::Deny`].
    pub async fn ask(&self, req: ConfirmationRequest) -> Approval {
        let confirm_id = uuid::Uuid::new_v4().to_string();
        let (tx, rx) = oneshot::channel();
        {
            let mut pending = self.pending.lock();
            if pending.closed {
                warn!(
                    target: "assistd::policy",
                    tool = %req.tool,
                    pattern = %req.matched_pattern,
                    "command denied: client cannot answer prompts"
                );
                return Approval::Deny;
            }
            if pending.prompts.len() >= MAX_PENDING_CONFIRMS {
                warn!(
                    target: "assistd::policy",
                    tool = %req.tool,
                    in_flight = pending.prompts.len(),
                    cap = MAX_PENDING_CONFIRMS,
                    "command denied: pending-confirm cap reached"
                );
                return Approval::Deny;
            }
            pending.prompts.insert(confirm_id.clone(), tx);
        }

        let event = Event::ConfirmRequest {
            id: self.request_id.clone(),
            confirm_id: confirm_id.clone(),
            tool: req.tool.clone(),
            script: req.script.clone(),
            matched_pattern: req.matched_pattern.clone(),
            always_allow: req.always_allow.clone(),
        };
        if self.wire.send(event).await.is_err() {
            self.pending.lock().prompts.remove(&confirm_id);
            warn!(
                target: "assistd::policy",
                tool = %req.tool,
                "command denied: client disconnected before confirm"
            );
            return Approval::Deny;
        }

        match tokio::time::timeout(self.timeout, rx).await {
            Ok(Ok(approval)) => approval,
            Ok(Err(_)) => {
                self.pending.lock().prompts.remove(&confirm_id);
                warn!(
                    target: "assistd::policy",
                    tool = %req.tool,
                    "command denied: confirmation channel dropped"
                );
                Approval::Deny
            }
            Err(_) => {
                self.pending.lock().prompts.remove(&confirm_id);
                warn!(
                    target: "assistd::policy",
                    tool = %req.tool,
                    timeout_secs = self.timeout.as_secs(),
                    "command denied: no answer before timeout"
                );
                Approval::Deny
            }
        }
    }

    /// Deliver a client's answer to the matching pending prompt.
    ///
    /// # Errors
    ///
    /// [`NoPendingConfirm`] when no prompt with that id is in flight.
    pub fn route_response(
        &self,
        confirm_id: &str,
        approval: Approval,
    ) -> Result<(), NoPendingConfirm> {
        let tx = self
            .pending
            .lock()
            .prompts
            .remove(confirm_id)
            .ok_or(NoPendingConfirm)?;
        let _ = tx.send(approval);
        Ok(())
    }

    /// Deny every prompt in flight and every later ask, for when the
    /// client can no longer answer.
    pub fn close(&self) {
        let drained = {
            let mut pending = self.pending.lock();
            pending.closed = true;
            std::mem::take(&mut pending.prompts)
        };
        for (_, tx) in drained {
            let _ = tx.send(Approval::Deny);
        }
    }

    #[cfg(test)]
    fn pending_len(&self) -> usize {
        self.pending.lock().prompts.len()
    }
}

tokio::task_local! {
    /// The [`ConfirmRouter`] of the IPC connection whose request is
    /// being dispatched.
    pub static CONFIRM_ROUTER: Arc<ConfirmRouter>;
}

/// Wrap `fut` so it runs under the caller's [`CONFIRM_ROUTER`], if one
/// is in scope. Task-locals do not survive `tokio::spawn`; call this at
/// the spawn site so the spawned task can still reach the connection
/// that asked for it.
pub fn inherit_confirm_router<F: Future>(fut: F) -> impl Future<Output = F::Output> {
    let router = CONFIRM_ROUTER.try_with(Arc::clone).ok();
    async move {
        match router {
            Some(router) => CONFIRM_ROUTER.scope(router, fut).await,
            None => fut.await,
        }
    }
}

/// Gate that round-trips prompts through the [`CONFIRM_ROUTER`] in
/// scope, denying when there is none.
#[derive(Debug, Default)]
pub struct IpcConfirmationGate;

#[async_trait]
impl ConfirmationGate for IpcConfirmationGate {
    async fn confirm(&self, req: ConfirmationRequest) -> Approval {
        match CONFIRM_ROUTER.try_with(Arc::clone) {
            Ok(router) => router.ask(req).await,
            Err(_) => {
                warn!(
                    target: "assistd::policy",
                    tool = %req.tool,
                    pattern = %req.matched_pattern,
                    "command denied: no IPC client attached to ask"
                );
                Approval::Deny
            }
        }
    }
}

/// Case-insensitive literal-substring search over a bash script,
/// returning the first matching pattern.
///
/// A pattern ending in anything but an ASCII letter or digit only
/// matches where a shell word ends (at whitespace, an operator, a quote
/// or the end of the script), so `rm -rf /` matches `rm -rf / ;` but not
/// `rm -rf /tmp/build`. One ending in a letter or digit also matches the
/// start of a longer word, so `mkfs` matches `mkfs.ext4`. Empty patterns
/// are ignored (they would match every script).
pub fn matches_denylist<'a>(script: &str, patterns: &'a [String]) -> Option<&'a str> {
    let haystack = script.to_ascii_lowercase();
    patterns
        .iter()
        .find(|p| {
            let needle = p.to_ascii_lowercase();
            let Some(last) = needle.chars().next_back() else {
                return false;
            };
            if last.is_ascii_alphanumeric() {
                return haystack.contains(&needle);
            }
            haystack
                .char_indices()
                .filter_map(|(at, _)| haystack[at..].strip_prefix(needle.as_str()))
                .any(ends_word)
        })
        .map(String::as_str)
}

fn ends_word(rest: &str) -> bool {
    rest.chars()
        .next()
        .is_none_or(|c| c.is_whitespace() || ";&|()<>'\"`".contains(c))
}

/// How sandboxing was requested for subprocess-spawning commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SandboxRequest {
    /// Use bwrap if found on `PATH`; fall back to unsandboxed with a warn.
    Auto,
    /// Require bwrap; fail startup if missing.
    Bwrap,
    /// Never wrap; run bash under the daemon's own user.
    None,
}

/// How subprocesses are wrapped, as resolved once by [`probe_sandbox`].
#[derive(Debug, Clone)]
pub enum ResolvedSandboxMode {
    /// No wrapping; bash is spawned directly.
    None,
    /// `bwrap` was found at the given absolute path.
    Bwrap { path: PathBuf },
}

/// Resolved sandbox configuration shared by every subprocess-spawning
/// command.
#[derive(Debug)]
pub struct SandboxInfo {
    pub mode: ResolvedSandboxMode,
    /// Extra args appended verbatim to the bwrap invocation before `--`.
    pub extra_args: Vec<String>,
    /// What sandboxed commands may not change.
    pub protected: Protected,
    /// Absolute directories unsandboxed commands look programs up in.
    host_path: Vec<PathBuf>,
}

/// Paths commands must not change, even inside the sandbox's writable
/// binds.
#[derive(Debug, Clone, Default)]
pub struct Protected {
    /// Directories mounted read-only: assistd's own configuration.
    pub dirs: Vec<PathBuf>,
    /// Sockets hidden from sandboxed commands: the daemon's IPC socket,
    /// through which a command could answer its own prompts.
    pub sockets: Vec<PathBuf>,
}

/// The `PATH` sandboxed commands run with, and the fallback when the
/// daemon has no usable one.
const SANDBOX_PATH: &str = "/usr/local/bin:/usr/bin:/bin";

/// Session resources a sandboxed command needs beyond the default
/// profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SandboxAccess {
    /// The default profile. `/run` is a fresh tmpfs, so the compositor
    /// and D-Bus session sockets are unreachable.
    Default,
    /// Additionally bind `$XDG_RUNTIME_DIR` back over the `/run` tmpfs.
    /// GUI applications need this: without the Wayland and D-Bus sockets
    /// they fail to start at all.
    Session,
}

impl SandboxInfo {
    /// A configuration that never wraps.
    pub fn none() -> Arc<Self> {
        Arc::new(Self {
            mode: ResolvedSandboxMode::None,
            extra_args: Vec::new(),
            protected: Protected::default(),
            host_path: host_path(&std::env::var_os("PATH").unwrap_or_default()),
        })
    }

    /// Where the commands this configuration runs look up bare program
    /// names.
    pub fn search_path(&self) -> SearchPath {
        match self.mode {
            ResolvedSandboxMode::None => SearchPath {
                dirs: self.host_path.clone(),
                read_only: false,
            },
            ResolvedSandboxMode::Bwrap { .. } => SearchPath {
                dirs: std::env::split_paths(SANDBOX_PATH).collect(),
                read_only: true,
            },
        }
    }

    /// Build the [`ProcCommand`] that runs `program` with `args`, wrapped
    /// in bubblewrap when the resolved mode is
    /// [`ResolvedSandboxMode::Bwrap`].
    ///
    /// Flag order is load-bearing: the default profile first, then the
    /// `access` binds (which must land *after* `--tmpfs /run` to be
    /// visible), then the protections over both, then the operator's
    /// `extra_args` so they always win.
    ///
    /// Unwrapped, the command runs with `PATH` limited to its absolute
    /// entries, the directories [`SandboxInfo::search_path`] reports, so
    /// a bare name runs the program the review resolved it to.
    pub fn command<I, S>(&self, access: SandboxAccess, program: &str, args: I) -> ProcCommand
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        match &self.mode {
            ResolvedSandboxMode::None => {
                let mut cmd = ProcCommand::new(program);
                cmd.args(args);
                if let Ok(path) = std::env::join_paths(&self.host_path) {
                    cmd.env("PATH", path);
                }
                cmd
            }
            ResolvedSandboxMode::Bwrap { path } => {
                let home = std::env::var("HOME").ok();
                let runtime_dir = std::env::var("XDG_RUNTIME_DIR").ok();
                let mut cmd = ProcCommand::new(path);
                cmd.args(default_bwrap_flags_for(home.clone()).iter().map(OsStr::new));
                let mut writable = vec![PathBuf::from("/tmp")];
                writable.extend(home.filter(|h| is_bindable_home(h)).map(PathBuf::from));
                if access == SandboxAccess::Session {
                    cmd.args(
                        session_bind_flags_for(runtime_dir.clone())
                            .iter()
                            .map(OsStr::new),
                    );
                    writable.extend(runtime_dir.map(PathBuf::from));
                }
                cmd.args(protection_flags(&self.protected, &writable));
                cmd.args(self.extra_args.iter().map(OsStr::new));
                cmd.arg("--");
                cmd.arg(program).args(args);
                cmd
            }
        }
    }
}

/// Read-only binds for the protected directories, and `/dev/null` over
/// the protected sockets, for those that exist under one of the
/// sandbox's `writable` binds. Anything else is already read-only or
/// out of sight, and binding onto it would make `bwrap` create mount
/// points or fail.
fn protection_flags(protected: &Protected, writable: &[PathBuf]) -> Vec<PathBuf> {
    let visible = |path: &Path| writable.iter().any(|root| path.starts_with(root)) && path.exists();
    let dirs = protected
        .dirs
        .iter()
        .filter(|dir| visible(dir))
        .flat_map(|dir| [PathBuf::from("--ro-bind"), dir.clone(), dir.clone()]);
    let sockets = protected
        .sockets
        .iter()
        .filter(|s| visible(s))
        .flat_map(|socket| {
            [
                PathBuf::from("--ro-bind"),
                PathBuf::from("/dev/null"),
                socket.clone(),
            ]
        });
    dirs.chain(sockets).collect()
}

/// Default bubblewrap flags applied before any user `bwrap_extra_args`.
/// Read-only root, writable `/tmp` and (when it is a usable non-root
/// directory) `$HOME`, standard `/dev` and `/proc`,
/// isolated pid/ipc/uts namespaces, dies with the daemon. Crucially *not*
/// `--unshare-net`: the assistant legitimately needs curl/pip/etc., so
/// network isolation is opt-in via `bwrap_extra_args`.
fn default_bwrap_flags_for(home: Option<String>) -> Vec<String> {
    let mut flags: Vec<String> = vec!["--ro-bind".into(), "/".into(), "/".into()];
    let home = home.filter(|h| is_bindable_home(h));
    match &home {
        Some(home) => flags.extend(["--bind".into(), home.clone(), home.clone()]),
        None => warn!(
            target: "assistd::policy",
            "HOME is unset or not a non-root absolute directory; sandboxed commands \
             will have no writable home"
        ),
    }
    flags.extend([
        "--bind".into(),
        "/tmp".into(),
        "/tmp".into(),
        "--dev".into(),
        "/dev".into(),
        "--proc".into(),
        "/proc".into(),
        "--tmpfs".into(),
        "/run".into(),
        "--unshare-pid".into(),
        "--unshare-ipc".into(),
        "--unshare-uts".into(),
        "--new-session".into(),
        "--die-with-parent".into(),
    ]);
    if let Some(home) = home {
        flags.extend(["--setenv".into(), "HOME".into(), home]);
    }
    flags.extend(["--setenv".into(), "PATH".into(), SANDBOX_PATH.into()]);
    flags
}

fn is_bindable_home(home: &str) -> bool {
    let path = std::path::Path::new(home);
    path.is_absolute()
        && std::fs::canonicalize(path).is_ok_and(|real| real.is_dir() && real.parent().is_some())
}

/// Bind flags for [`SandboxAccess::Session`], re-exposing the compositor
/// and D-Bus session sockets that the default profile's `--tmpfs /run`
/// hides. Empty when `runtime_dir` is unset or not a directory: `bwrap`
/// aborts on a missing bind source, so a stale value would take every
/// launch down with it rather than merely leaving the sandbox tight.
fn session_bind_flags_for(runtime_dir: Option<String>) -> Vec<String> {
    match runtime_dir {
        Some(dir) if std::path::Path::new(&dir).is_dir() => {
            vec!["--bind".into(), dir.clone(), dir]
        }
        _ => {
            warn!(
                target: "assistd::policy",
                "XDG_RUNTIME_DIR is unset or not a directory; sandboxed launches will have \
                 no compositor or D-Bus session socket"
            );
            Vec::new()
        }
    }
}

/// Why [`probe_sandbox`] could not satisfy the requested sandbox.
#[derive(Debug, thiserror::Error)]
pub enum SandboxError {
    /// `Bwrap` was required but no `bwrap` executable is on `PATH`.
    #[error(
        "tools.bash.sandbox = \"bwrap\" but `bwrap` was not found on PATH. \
         Install bubblewrap or change sandbox to \"auto\" / \"none\"."
    )]
    BwrapNotFound,
}

/// Resolve `request` against the environment once, for the whole
/// process lifetime. `Auto` falls back to unsandboxed with a warning
/// when `bwrap` is missing.
///
/// # Errors
///
/// [`SandboxError::BwrapNotFound`] when `request` is `Bwrap` and no
/// `bwrap` is on `PATH`.
pub fn probe_sandbox(
    request: SandboxRequest,
    extra_args: Vec<String>,
    protected: Protected,
) -> Result<Arc<SandboxInfo>, SandboxError> {
    let path_env = std::env::var_os("PATH").unwrap_or_default();
    probe_sandbox_with_path(request, extra_args, protected, &path_env)
}

fn probe_sandbox_with_path(
    request: SandboxRequest,
    extra_args: Vec<String>,
    protected: Protected,
    path_env: &std::ffi::OsStr,
) -> Result<Arc<SandboxInfo>, SandboxError> {
    let mode = match request {
        SandboxRequest::None => {
            info!(target: "assistd::policy", "bash sandbox: disabled by config");
            ResolvedSandboxMode::None
        }
        SandboxRequest::Auto => match find_executable("bwrap", path_env) {
            Some(path) => {
                info!(
                    target: "assistd::policy",
                    path = %path.display(),
                    "bash sandbox: bubblewrap enabled (auto-detected)"
                );
                ResolvedSandboxMode::Bwrap { path }
            }
            None => {
                warn!(
                    target: "assistd::policy",
                    "bubblewrap not found on PATH; bash commands will run unsandboxed under the current user. \
                     Install bubblewrap (package: bubblewrap) for defence-in-depth."
                );
                ResolvedSandboxMode::None
            }
        },
        SandboxRequest::Bwrap => match find_executable("bwrap", path_env) {
            Some(path) => {
                info!(
                    target: "assistd::policy",
                    path = %path.display(),
                    "bash sandbox: bubblewrap enabled (required by config)"
                );
                ResolvedSandboxMode::Bwrap { path }
            }
            None => return Err(SandboxError::BwrapNotFound),
        },
    };
    Ok(Arc::new(SandboxInfo {
        mode,
        extra_args,
        protected,
        host_path: host_path(path_env),
    }))
}

/// The absolute directories of `path_env`, or the sandbox's `PATH` when
/// it has none. Relative and empty entries are dropped: they resolve
/// against whatever directory a command happens to be in.
fn host_path(path_env: &OsStr) -> Vec<PathBuf> {
    let dirs: Vec<PathBuf> = std::env::split_paths(path_env)
        .filter(|dir| dir.is_absolute())
        .collect();
    if dirs.is_empty() {
        std::env::split_paths(SANDBOX_PATH).collect()
    } else {
        dirs
    }
}

fn find_executable(name: &str, path_env: &std::ffi::OsStr) -> Option<PathBuf> {
    for dir in std::env::split_paths(path_env) {
        if dir.as_os_str().is_empty() {
            continue;
        }
        let candidate = dir.join(name);
        if is_executable_file(&candidate) {
            return Some(candidate);
        }
    }
    None
}

#[cfg(unix)]
fn is_executable_file(path: &std::path::Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    match std::fs::metadata(path) {
        Ok(md) => md.is_file() && md.permissions().mode() & 0o111 != 0,
        Err(_) => false,
    }
}

#[cfg(not(unix))]
fn is_executable_file(path: &std::path::Path) -> bool {
    path.is_file()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_request() -> ConfirmationRequest {
        ConfirmationRequest {
            tool: "bash".into(),
            script: "rm -rf foo".into(),
            matched_pattern: "rm -rf".into(),
            always_allow: Vec::new(),
        }
    }

    async fn recv_confirm_id(rx: &mut mpsc::Receiver<Event>) -> String {
        match rx.recv().await.expect("prompt on the wire") {
            Event::ConfirmRequest { confirm_id, .. } => confirm_id,
            other => panic!("expected ConfirmRequest, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn router_denies_and_forgets_prompt_after_timeout() {
        let (tx, mut rx) = mpsc::channel(4);
        let router = ConfirmRouter::new("r".into(), tx, Duration::from_millis(50));
        let ask = router.ask(sample_request());
        let (approval, confirm_id) = tokio::join!(ask, recv_confirm_id(&mut rx));
        assert_eq!(approval, Approval::Deny);
        assert_eq!(router.pending_len(), 0);
        router
            .route_response(&confirm_id, Approval::Once)
            .expect_err("a timed-out prompt is no longer routable");
    }

    #[tokio::test]
    async fn router_close_denies_in_flight_prompt_and_later_asks() {
        let (tx, mut rx) = mpsc::channel(4);
        let router = ConfirmRouter::new("r".into(), tx, Duration::from_secs(60));
        let asker = Arc::clone(&router);
        let in_flight = tokio::spawn(async move { asker.ask(sample_request()).await });
        recv_confirm_id(&mut rx).await;

        router.close();
        assert_eq!(in_flight.await.expect("ask task"), Approval::Deny);
        assert_eq!(router.pending_len(), 0);

        assert_eq!(router.ask(sample_request()).await, Approval::Deny);
        rx.try_recv()
            .expect_err("a closed router must not put prompts on the wire");
    }

    #[tokio::test]
    async fn router_denies_without_asking_once_pending_cap_is_reached() {
        let (tx, mut rx) = mpsc::channel(4);
        let router = ConfirmRouter::new("r".into(), tx, CONFIRM_TIMEOUT);
        {
            let mut pending = router.pending.lock();
            for i in 0..MAX_PENDING_CONFIRMS {
                pending
                    .prompts
                    .insert(format!("preloaded-{i}"), oneshot::channel().0);
            }
        }

        assert_eq!(router.ask(sample_request()).await, Approval::Deny);
        assert_eq!(router.pending_len(), MAX_PENDING_CONFIRMS);
        rx.try_recv()
            .expect_err("a denied ask must not put a prompt on the wire");
    }

    #[tokio::test]
    async fn inherit_confirm_router_carries_router_across_spawn() {
        let (tx, mut rx) = mpsc::channel(4);
        let router = ConfirmRouter::new("r".into(), tx, Duration::from_secs(60));
        let answer = CONFIRM_ROUTER.sync_scope(Arc::clone(&router), || {
            tokio::spawn(inherit_confirm_router(async {
                IpcConfirmationGate.confirm(sample_request()).await
            }))
        });
        let confirm_id = recv_confirm_id(&mut rx).await;
        router
            .route_response(&confirm_id, Approval::Always)
            .expect("routed");
        assert_eq!(answer.await.expect("spawned gate"), Approval::Always);
    }

    #[tokio::test]
    async fn bare_spawn_loses_router_and_gate_denies() {
        let (tx, mut rx) = mpsc::channel(4);
        let router = ConfirmRouter::new("r".into(), tx, Duration::from_secs(60));
        let answer = CONFIRM_ROUTER.sync_scope(router, || {
            tokio::spawn(async { IpcConfirmationGate.confirm(sample_request()).await })
        });
        assert_eq!(answer.await.expect("spawned gate"), Approval::Deny);
        rx.try_recv()
            .expect_err("a gate with no router must not reach the wire");
    }

    #[test]
    fn an_answer_is_always_only_when_it_allows() {
        assert_eq!(Approval::from_answer(false, true), Approval::Deny);
        assert_eq!(Approval::from_answer(true, false), Approval::Once);
        assert_eq!(Approval::from_answer(true, true), Approval::Always);
    }

    #[test]
    fn denylist_returns_first_case_insensitive_substring_match() {
        let patterns = ["", "rm -rf /", "mkfs", "> /dev/nvme"].map(String::from);
        for (script, expected) in [
            ("rm -rf /", Some("rm -rf /")),
            ("RM -RF /", Some("rm -rf /")),
            ("sudo mkfs.ext4 /dev/sda1", Some("mkfs")),
            ("mkfs /dev/sda1 && rm -rf /", Some("rm -rf /")),
            ("ls -l /tmp", None),
            ("rm -rf / --no-preserve-root", Some("rm -rf /")),
            ("rm -rf /;true", Some("rm -rf /")),
            ("bash -c 'rm -rf /'", Some("rm -rf /")),
            ("rm -rf /tmp/build", None),
            ("rm -rf /tmp/build && rm -rf /", Some("rm -rf /")),
            ("echo x > /dev/nvme0n1", Some("> /dev/nvme")),
        ] {
            assert_eq!(matches_denylist(script, &patterns), expected, "{script:?}");
        }
    }

    fn argv_of(cmd: &tokio::process::Command) -> Vec<String> {
        cmd.as_std()
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn unsandboxed_command_passes_argv_through_verbatim() {
        let cmd =
            SandboxInfo::none().command(SandboxAccess::Default, "firefox", ["--new-window", "a b"]);
        assert_eq!(cmd.as_std().get_program(), "firefox");
        assert_eq!(argv_of(&cmd), ["--new-window", "a b"]);
    }

    /// Operator extra args come last so they override the profile, and
    /// session binds must follow `--tmpfs /run` or the tmpfs shadows them.
    #[test]
    fn bwrap_argv_is_profile_then_access_binds_then_extra_args_then_program() {
        let info = SandboxInfo {
            mode: ResolvedSandboxMode::Bwrap {
                path: PathBuf::from("/usr/bin/bwrap"),
            },
            extra_args: vec!["--unshare-net".into()],
            protected: Protected::default(),
            host_path: Vec::new(),
        };
        let profile = default_bwrap_flags_for(std::env::var("HOME").ok());
        assert!(profile.windows(2).any(|w| w == ["--tmpfs", "/run"]));
        let tail = ["--unshare-net", "--", "firefox", "--new-window"].map(String::from);
        for (access, binds) in [
            (SandboxAccess::Default, Vec::new()),
            (
                SandboxAccess::Session,
                session_bind_flags_for(std::env::var("XDG_RUNTIME_DIR").ok()),
            ),
        ] {
            let cmd = info.command(access, "firefox", ["--new-window"]);
            assert_eq!(cmd.as_std().get_program(), "/usr/bin/bwrap");
            let expected = [profile.clone(), binds, tail.to_vec()].concat();
            assert_eq!(argv_of(&cmd), expected, "{access:?}");
        }
    }

    /// Protections go on after every writable bind they sit under, and
    /// only for paths that exist there.
    #[test]
    fn protections_cover_existing_paths_under_writable_binds() {
        let scratch = tempfile::tempdir().expect("tempdir");
        let config = scratch.path().join("config");
        std::fs::create_dir(&config).expect("config dir");
        let socket = scratch.path().join("assistd.sock");
        std::fs::write(&socket, b"").expect("socket stand-in");
        let protected = Protected {
            dirs: vec![config.clone(), scratch.path().join("missing")],
            sockets: vec![socket.clone(), PathBuf::from("/nonexistent/assistd.sock")],
        };
        let writable = [scratch.path().to_path_buf()];
        assert_eq!(
            protection_flags(&protected, &writable),
            [
                PathBuf::from("--ro-bind"),
                config.clone(),
                config,
                PathBuf::from("--ro-bind"),
                PathBuf::from("/dev/null"),
                socket,
            ]
        );
        assert!(protection_flags(&protected, &[PathBuf::from("/elsewhere")]).is_empty());
    }

    #[test]
    fn unsandboxed_path_keeps_only_absolute_entries() {
        assert_eq!(
            host_path(OsStr::new("/usr/bin::.:bin:/bin")),
            [PathBuf::from("/usr/bin"), PathBuf::from("/bin")]
        );
        assert_eq!(
            host_path(OsStr::new(".:")),
            std::env::split_paths(SANDBOX_PATH).collect::<Vec<_>>()
        );
    }

    #[test]
    fn session_binds_only_an_existing_runtime_dir() {
        let dir = std::env::temp_dir().to_string_lossy().into_owned();
        assert_eq!(
            session_bind_flags_for(Some(dir.clone())),
            ["--bind", &dir, &dir]
        );
        assert!(session_bind_flags_for(None).is_empty());
        assert!(
            session_bind_flags_for(Some("/nonexistent/assistd-runtime-dir".into())).is_empty(),
            "a stale XDG_RUNTIME_DIR must not be passed to bwrap, which aborts on a \
             missing bind source"
        );
    }

    fn writable_binds(flags: &[String]) -> Vec<&str> {
        flags
            .windows(2)
            .filter(|w| w[0] == "--bind")
            .map(|w| w[1].as_str())
            .collect()
    }

    #[test]
    fn home_is_bound_writable_when_usable() {
        let dir = std::env::temp_dir();
        let dir_str = dir.to_string_lossy().into_owned();
        let flags = default_bwrap_flags_for(Some(dir_str.clone()));
        assert!(writable_binds(&flags).contains(&dir_str.as_str()));
        let setenv = flags
            .windows(3)
            .find(|w| w[0] == "--setenv" && w[1] == "HOME")
            .expect("HOME is exported into the sandbox");
        assert_eq!(setenv[2], dir_str);
    }

    #[test]
    fn unusable_home_never_binds_the_root_writable() {
        for home in [
            None,
            Some(""),
            Some("/"),
            Some("//"),
            Some("/tmp/.."),
            Some("relative"),
        ] {
            let flags = default_bwrap_flags_for(home.map(str::to_string));
            assert_eq!(
                writable_binds(&flags),
                ["/tmp"],
                "HOME={home:?} must only leave /tmp writable: {flags:?}"
            );
            assert!(
                !flags
                    .windows(2)
                    .any(|w| w[0] == "--setenv" && w[1] == "HOME"),
                "HOME={home:?} must not be exported: {flags:?}"
            );
        }
    }

    #[test]
    fn probe_sandbox_runs_unwrapped_when_disabled_or_bwrap_is_absent() {
        for request in [SandboxRequest::None, SandboxRequest::Auto] {
            let info =
                probe_sandbox_with_path(request, Vec::new(), Protected::default(), OsStr::new(""))
                    .unwrap_or_else(|e| panic!("{request:?}: {e}"));
            assert!(
                matches!(info.mode, ResolvedSandboxMode::None),
                "{request:?}"
            );
        }
    }

    #[test]
    fn probe_sandbox_bwrap_missing_fails_startup() {
        probe_sandbox_with_path(
            SandboxRequest::Bwrap,
            Vec::new(),
            Protected::default(),
            OsStr::new(""),
        )
        .expect_err("bwrap is required but absent from PATH");
    }
}
