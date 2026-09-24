//! Command-execution policy: confirmation gates, pattern matchers, and
//! sandbox probing.

use std::collections::HashMap;
use std::ffi::OsStr;
use std::future::Future;
use std::path::PathBuf;
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

/// Policy for the commands that spawn subprocesses. Destructive
/// patterns are pre-tokenized so no invocation re-parses them.
///
/// The denylist and destructive patterns are syntactic backstops for
/// obviously dangerous invocations (`rm -rf /`, `mkfs`, …). Variable
/// expansion, here-docs or command substitution defeat them, so they
/// are not the real defense; the bwrap sandbox is.
#[derive(Debug, Clone)]
pub struct BashPolicyCfg {
    pub timeout: Duration,
    pub denylist: Vec<String>,
    pub destructive_patterns: Vec<Vec<String>>,
}

impl Default for BashPolicyCfg {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            denylist: Vec::new(),
            destructive_patterns: Vec::new(),
        }
    }
}

/// Everything a command needs to run model-chosen argv: the policy,
/// the sandbox to wrap it in, and the gate that confirms destructive
/// invocations.
pub(crate) struct SubprocessPolicy {
    pub(crate) cfg: Arc<BashPolicyCfg>,
    pub(crate) sandbox: Arc<SandboxInfo>,
    pub(crate) gate: Arc<dyn ConfirmationGate>,
}

impl SubprocessPolicy {
    /// Refuse `script` when it hits the denylist or when the gate
    /// declines a destructive match. `tool` and `op` name the command and
    /// operation in the error line. `destructive` is the destructive
    /// pattern match, computed by the command because what it matches (a
    /// script, an argv) differs per command.
    pub(crate) async fn authorize(
        &self,
        tool: &str,
        op: &str,
        script: &str,
        destructive: Option<&[String]>,
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
        let Some(matched) = destructive else {
            return Ok(());
        };
        let pattern_display = matched.join(" ");
        let approved = self
            .gate
            .confirm(ConfirmationRequest {
                tool: tool.to_string(),
                script: script.to_string(),
                matched_pattern: pattern_display.clone(),
            })
            .await;
        if approved {
            return Ok(());
        }
        Err(CommandOutput::failed(
            POLICY_DENIED_EXIT,
            error_line(
                tool,
                format_args!(
                    "{op} cancelled by user. Matched destructive pattern: {pattern_display}"
                ),
                Hint::Try,
                "a different approach",
            )
            .into_bytes(),
        ))
    }
}

/// Describes a request for user confirmation before executing a destructive
/// command. Passed to [`ConfirmationGate::confirm`].
#[derive(Debug, Clone)]
pub struct ConfirmationRequest {
    /// Tool name requesting confirmation (e.g. `"bash"`).
    pub tool: String,
    /// Verbatim script the tool is about to execute.
    pub script: String,
    /// The configured destructive pattern that triggered the prompt
    /// (rendered for display, e.g. `"rm -rf"`).
    pub matched_pattern: String,
}

/// Decides whether a destructive command may run.
#[async_trait]
pub trait ConfirmationGate: Send + Sync + 'static {
    /// Ask for confirmation. `true` = proceed, `false` = cancel.
    ///
    /// Implementations must convert *every* failure mode (channel drop, UI
    /// shutdown, timeout) into `false` so a turn never hangs.
    async fn confirm(&self, req: ConfirmationRequest) -> bool;
}

/// Gate that never approves, logging each denial.
#[cfg(any(test, feature = "test-support"))]
#[derive(Debug, Default)]
pub struct DenyAllGate;

#[cfg(any(test, feature = "test-support"))]
#[async_trait]
impl ConfirmationGate for DenyAllGate {
    async fn confirm(&self, req: ConfirmationRequest) -> bool {
        warn!(
            target: "assistd::policy",
            tool = %req.tool,
            pattern = %req.matched_pattern,
            "destructive command denied: no interactive confirmation gate attached"
        );
        false
    }
}

/// Gate that always approves, defeating the confirmation layer.
#[cfg(any(test, feature = "test-support"))]
#[derive(Debug, Default)]
pub struct AlwaysAllowGate;

#[cfg(any(test, feature = "test-support"))]
#[async_trait]
impl ConfirmationGate for AlwaysAllowGate {
    async fn confirm(&self, _req: ConfirmationRequest) -> bool {
        true
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
    prompts: HashMap<String, oneshot::Sender<bool>>,
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
/// flight further asks are denied rather than queued.
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
    /// router, timeout) is `false`.
    pub async fn ask(&self, req: ConfirmationRequest) -> bool {
        let confirm_id = uuid::Uuid::new_v4().to_string();
        let (tx, rx) = oneshot::channel();
        {
            let mut pending = self.pending.lock();
            if pending.closed {
                warn!(
                    target: "assistd::policy",
                    tool = %req.tool,
                    pattern = %req.matched_pattern,
                    "destructive command denied: client cannot answer prompts"
                );
                return false;
            }
            if pending.prompts.len() >= MAX_PENDING_CONFIRMS {
                warn!(
                    target: "assistd::policy",
                    tool = %req.tool,
                    in_flight = pending.prompts.len(),
                    cap = MAX_PENDING_CONFIRMS,
                    "destructive command denied: pending-confirm cap reached"
                );
                return false;
            }
            pending.prompts.insert(confirm_id.clone(), tx);
        }

        let event = Event::ConfirmRequest {
            id: self.request_id.clone(),
            confirm_id: confirm_id.clone(),
            tool: req.tool.clone(),
            script: req.script.clone(),
            matched_pattern: req.matched_pattern.clone(),
        };
        if self.wire.send(event).await.is_err() {
            self.pending.lock().prompts.remove(&confirm_id);
            warn!(
                target: "assistd::policy",
                tool = %req.tool,
                "destructive command denied: client disconnected before confirm"
            );
            return false;
        }

        match tokio::time::timeout(self.timeout, rx).await {
            Ok(Ok(allow)) => allow,
            Ok(Err(_)) => {
                self.pending.lock().prompts.remove(&confirm_id);
                warn!(
                    target: "assistd::policy",
                    tool = %req.tool,
                    "destructive command denied: confirmation channel dropped"
                );
                false
            }
            Err(_) => {
                self.pending.lock().prompts.remove(&confirm_id);
                warn!(
                    target: "assistd::policy",
                    tool = %req.tool,
                    timeout_secs = self.timeout.as_secs(),
                    "destructive command denied: no answer before timeout"
                );
                false
            }
        }
    }

    /// Deliver a client's answer to the matching pending prompt.
    ///
    /// # Errors
    ///
    /// [`NoPendingConfirm`] when no prompt with that id is in flight.
    pub fn route_response(&self, confirm_id: &str, allow: bool) -> Result<(), NoPendingConfirm> {
        let tx = self
            .pending
            .lock()
            .prompts
            .remove(confirm_id)
            .ok_or(NoPendingConfirm)?;
        let _ = tx.send(allow);
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
            let _ = tx.send(false);
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
    async fn confirm(&self, req: ConfirmationRequest) -> bool {
        match CONFIRM_ROUTER.try_with(Arc::clone) {
            Ok(router) => router.ask(req).await,
            Err(_) => {
                warn!(
                    target: "assistd::policy",
                    tool = %req.tool,
                    pattern = %req.matched_pattern,
                    "destructive command denied: no IPC client attached to ask"
                );
                false
            }
        }
    }
}

/// Case-insensitive literal-substring search over a bash script,
/// returning the first matching pattern. Empty patterns are ignored
/// (they would match every script).
pub fn matches_denylist<'a>(script: &str, patterns: &'a [String]) -> Option<&'a str> {
    let haystack = script.to_ascii_lowercase();
    patterns.iter().find_map(|p| {
        if p.is_empty() {
            return None;
        }
        let needle = p.to_ascii_lowercase();
        if haystack.contains(&needle) {
            Some(p.as_str())
        } else {
            None
        }
    })
}

const COMMAND_SEPARATORS: &[&str] = &["|", "||", ";", "&&", "&"];

/// Commands that run their arguments as another command. Every token
/// after one is a potential command word, since its own options (`sudo
/// -u root`, `nice -n 10`) cannot be told apart from the command without
/// knowing each wrapper's flags.
const PASS_THROUGH_WRAPPERS: &[&str] = &[
    "sudo", "doas", "env", "exec", "xargs", "nohup", "nice", "time", "timeout", "command",
    "setsid", "stdbuf",
];

/// Whether any command in the script starts with a configured
/// destructive prefix, compared case-insensitively. Returns the matched
/// prefix.
///
/// The script is split into lines at unquoted newlines, and each line
/// into commands at `|`, `||`, `;`, `&&` and `&`. A command's words are
/// checked after any leading `NAME=value` assignments, and every word
/// after a pass-through wrapper (`sudo`, `env`, `xargs`, …) is checked
/// too. Quoted arguments (`echo "rm -rf"`) stay single tokens and so do
/// not match `["rm", "-rf"]`. A line shlex cannot parse counts as no
/// match; bash will surface the syntax error itself.
pub fn matches_destructive<'a>(script: &str, prefixes: &'a [Vec<String>]) -> Option<&'a [String]> {
    script_lines(script)
        .into_iter()
        .filter_map(shlex::split)
        .find_map(|tokens| {
            let anchors = command_anchors(&tokens);
            prefixes
                .iter()
                .filter(|prefix| !prefix.is_empty())
                .find(|prefix| {
                    anchors.iter().any(|&at| {
                        tokens.get(at..at + prefix.len()).is_some_and(|words| {
                            words
                                .iter()
                                .zip(prefix.iter())
                                .all(|(word, pat)| word.eq_ignore_ascii_case(pat))
                        })
                    })
                })
                .map(Vec::as_slice)
        })
}

fn script_lines(script: &str) -> Vec<&str> {
    #[derive(Clone, Copy)]
    enum State {
        Plain,
        Single,
        Double,
        Comment,
    }

    let bytes = script.as_bytes();
    let mut lines = Vec::new();
    let mut state = State::Plain;
    let mut start = 0;
    let mut end = None;
    let mut i = 0;
    while i < bytes.len() {
        match (state, bytes[i]) {
            (State::Plain | State::Double, b'\\') => i += 1,
            (State::Plain, b'\'') => state = State::Single,
            (State::Plain, b'"') => state = State::Double,
            (State::Single, b'\'') | (State::Double, b'"') => state = State::Plain,
            (State::Plain, b'#') if i == 0 || b" \t\n;&|()".contains(&bytes[i - 1]) => {
                state = State::Comment;
                end = Some(i);
            }
            (State::Plain | State::Comment, b'\n') => {
                lines.push(&script[start..end.unwrap_or(i)]);
                state = State::Plain;
                start = i + 1;
                end = None;
            }
            _ => {}
        }
        i += 1;
    }
    lines.push(&script[start..end.unwrap_or(bytes.len())]);
    lines
}

fn command_anchors(tokens: &[String]) -> Vec<usize> {
    let mut anchors = Vec::new();
    let mut at_command = true;
    let mut wrapped = false;
    for (i, tok) in tokens.iter().enumerate() {
        if COMMAND_SEPARATORS.contains(&tok.as_str()) {
            at_command = true;
            wrapped = false;
            continue;
        }
        if at_command || wrapped {
            anchors.push(i);
        }
        if at_command {
            wrapped |= PASS_THROUGH_WRAPPERS
                .iter()
                .any(|w| w.eq_ignore_ascii_case(tok));
            at_command = is_assignment(tok);
        }
    }
    anchors
}

fn is_assignment(tok: &str) -> bool {
    tok.split_once('=').is_some_and(|(name, _)| {
        name.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
            && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    })
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
}

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
        })
    }

    /// Build the [`ProcCommand`] that runs `program` with `args`, wrapped
    /// in bubblewrap when the resolved mode is
    /// [`ResolvedSandboxMode::Bwrap`].
    ///
    /// Flag order is load-bearing: the default profile first, then the
    /// `access` binds (which must land *after* `--tmpfs /run` to be
    /// visible), then the operator's `extra_args` so they always win.
    pub fn command<I, S>(&self, access: SandboxAccess, program: &str, args: I) -> ProcCommand
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        match &self.mode {
            ResolvedSandboxMode::None => {
                let mut cmd = ProcCommand::new(program);
                cmd.args(args);
                cmd
            }
            ResolvedSandboxMode::Bwrap { path } => {
                let mut cmd = ProcCommand::new(path);
                cmd.args(default_bwrap_flags().iter().map(OsStr::new));
                if access == SandboxAccess::Session {
                    cmd.args(session_bind_flags().iter().map(OsStr::new));
                }
                cmd.args(self.extra_args.iter().map(OsStr::new));
                cmd.arg("--");
                cmd.arg(program).args(args);
                cmd
            }
        }
    }
}

/// Default bubblewrap flags applied before any user `bwrap_extra_args`.
/// Read-only root, writable `/tmp` and (when it is a usable non-root
/// directory) `$HOME`, standard `/dev` and `/proc`,
/// isolated pid/ipc/uts namespaces, dies with the daemon. Crucially *not*
/// `--unshare-net`: the assistant legitimately needs curl/pip/etc., so
/// network isolation is opt-in via `bwrap_extra_args`.
fn default_bwrap_flags() -> Vec<String> {
    default_bwrap_flags_for(std::env::var("HOME").ok())
}

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
    flags.extend([
        "--setenv".into(),
        "PATH".into(),
        "/usr/local/bin:/usr/bin:/bin".into(),
    ]);
    flags
}

fn is_bindable_home(home: &str) -> bool {
    let path = std::path::Path::new(home);
    path.is_absolute()
        && std::fs::canonicalize(path).is_ok_and(|real| real.is_dir() && real.parent().is_some())
}

/// Bind flags for [`SandboxAccess::Session`], re-exposing the compositor
/// and D-Bus session sockets that the default profile's `--tmpfs /run`
/// hides.
fn session_bind_flags() -> Vec<String> {
    session_bind_flags_for(std::env::var("XDG_RUNTIME_DIR").ok())
}

/// Empty when `runtime_dir` is unset or not a directory: `bwrap` aborts
/// on a missing bind source, so a stale value would take every launch
/// down with it rather than merely leaving the sandbox tight.
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
) -> Result<Arc<SandboxInfo>, SandboxError> {
    let path_env = std::env::var_os("PATH").unwrap_or_default();
    probe_sandbox_with_path(request, extra_args, &path_env)
}

fn probe_sandbox_with_path(
    request: SandboxRequest,
    extra_args: Vec<String>,
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
    Ok(Arc::new(SandboxInfo { mode, extra_args }))
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
        let (allowed, confirm_id) = tokio::join!(ask, recv_confirm_id(&mut rx));
        assert!(!allowed);
        assert_eq!(router.pending_len(), 0);
        router
            .route_response(&confirm_id, true)
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
        assert!(!in_flight.await.expect("ask task"));
        assert_eq!(router.pending_len(), 0);

        assert!(!router.ask(sample_request()).await);
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

        assert!(!router.ask(sample_request()).await);
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
        router.route_response(&confirm_id, true).expect("routed");
        assert!(answer.await.expect("spawned gate"));
    }

    #[tokio::test]
    async fn bare_spawn_loses_router_and_gate_denies() {
        let (tx, mut rx) = mpsc::channel(4);
        let router = ConfirmRouter::new("r".into(), tx, Duration::from_secs(60));
        let answer = CONFIRM_ROUTER.sync_scope(router, || {
            tokio::spawn(async { IpcConfirmationGate.confirm(sample_request()).await })
        });
        assert!(!answer.await.expect("spawned gate"));
        rx.try_recv()
            .expect_err("a gate with no router must not reach the wire");
    }

    #[test]
    fn denylist_returns_first_case_insensitive_substring_match() {
        let patterns = ["".to_string(), "rm -rf /".to_string(), "mkfs".to_string()];
        for (script, expected) in [
            ("rm -rf /", Some("rm -rf /")),
            ("RM -RF /", Some("rm -rf /")),
            ("sudo mkfs.ext4 /dev/sda1", Some("mkfs")),
            ("mkfs /dev/sda1 && rm -rf /", Some("rm -rf /")),
            ("ls -l /tmp", None),
        ] {
            assert_eq!(matches_denylist(script, &patterns), expected, "{script:?}");
        }
    }

    /// The empty prefix would match every script if it were not skipped.
    fn destructive_prefixes() -> Vec<Vec<String>> {
        vec![
            Vec::new(),
            vec!["shutdown".into()],
            vec!["rm".into(), "-rf".into()],
        ]
    }

    fn destructive_match(script: &str) -> Option<String> {
        matches_destructive(script, &destructive_prefixes()).map(|m| m.join(" "))
    }

    #[test]
    fn destructive_matches_a_prefix_in_any_command_position() {
        for (script, expected) in [
            ("rm -rf foo", "rm -rf"),
            ("RM -RF foo", "rm -rf"),
            ("shutdown -h now", "shutdown"),
            ("touch foo && rm -rf bar", "rm -rf"),
            ("false || rm -rf bar", "rm -rf"),
            ("echo hi ; rm -rf bar", "rm -rf"),
            ("ls | rm -rf foo", "rm -rf"),
            ("true\nrm -rf ~", "rm -rf"),
            ("cd /tmp\n\n  rm -rf ~\n", "rm -rf"),
            // A quote inside a comment must not open a string that
            // swallows the following lines.
            ("true # it's fine\nrm -rf ~", "rm -rf"),
            ("rm -rf ~;# it's fine", "rm -rf"),
            // An unparseable line does not hide the lines before it.
            ("rm -rf ~\necho \"oops", "rm -rf"),
            ("sudo rm -rf ~", "rm -rf"),
            ("sudo -u root rm -rf ~", "rm -rf"),
            ("exec rm -rf ~", "rm -rf"),
            ("env FOO=1 rm -rf ~", "rm -rf"),
            ("find . -print0 | xargs -0 rm -rf", "rm -rf"),
            ("nice -n 10 sudo rm -rf ~", "rm -rf"),
            ("FOO=1 BAR=2 rm -rf ~", "rm -rf"),
        ] {
            assert_eq!(
                destructive_match(script).as_deref(),
                Some(expected),
                "{script:?}"
            );
        }
    }

    #[test]
    fn destructive_ignores_words_outside_command_position() {
        for script in [
            "ls -l /tmp",
            "echo \"rm -rf\"",
            "echo 'shutdown'",
            "echo rm -rf ~",
            "sudo true ; echo rm -rf ~",
            "echo \"a\nrm -rf ~\"",
            "echo 'a\nrm -rf ~'",
            "echo \\\nrm -rf ~",
            "rm -rf \"unterminated",
        ] {
            assert_eq!(destructive_match(script), None, "{script:?}");
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
        };
        let profile = default_bwrap_flags();
        assert!(profile.windows(2).any(|w| w == ["--tmpfs", "/run"]));
        let tail = ["--unshare-net", "--", "firefox", "--new-window"].map(String::from);
        for (access, binds) in [
            (SandboxAccess::Default, Vec::new()),
            (SandboxAccess::Session, session_bind_flags()),
        ] {
            let cmd = info.command(access, "firefox", ["--new-window"]);
            assert_eq!(cmd.as_std().get_program(), "/usr/bin/bwrap");
            let expected = [profile.clone(), binds, tail.to_vec()].concat();
            assert_eq!(argv_of(&cmd), expected, "{access:?}");
        }
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
            let info = probe_sandbox_with_path(request, Vec::new(), OsStr::new(""))
                .unwrap_or_else(|e| panic!("{request:?}: {e}"));
            assert!(
                matches!(info.mode, ResolvedSandboxMode::None),
                "{request:?}"
            );
        }
    }

    #[test]
    fn probe_sandbox_bwrap_missing_fails_startup() {
        probe_sandbox_with_path(SandboxRequest::Bwrap, Vec::new(), OsStr::new(""))
            .expect_err("bwrap is required but absent from PATH");
    }
}
