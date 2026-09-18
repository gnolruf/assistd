//! Command-execution policy: confirmation gates, pattern matchers, and
//! sandbox probing.
//!
//! The denylist and destructive-pattern checks are syntactic backstops
//! for obvious dangerous invocations (`rm -rf /`, `mkfs`, …). A
//! sufficiently clever script defeats them through variable expansion,
//! here-docs, or command substitution, so they are not the real
//! defense; the bwrap sandbox is.

use std::collections::HashMap;
use std::ffi::OsStr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;

use async_trait::async_trait;
use tokio::process::Command as ProcCommand;
use tokio::sync::{mpsc, oneshot};
use tracing::{info, warn};

use assistd_ipc::Event;

use crate::command::{CommandOutput, error_line};
use crate::exec::POLICY_DENIED_EXIT;

/// Policy for the commands that spawn subprocesses. Destructive
/// patterns are pre-tokenized so no invocation re-parses them.
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
    /// declines a destructive match. `tool` and `op` name the caller in
    /// the error line; `destructive` is the caller's own match result,
    /// since `bash` matches its script and `wm open` matches argv.
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
                    "Try",
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
                "Try",
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
    /// shutdown, timeout) into `false` so the agent loop never hangs.
    async fn confirm(&self, req: ConfirmationRequest) -> bool;
}

/// Gate that never approves, logging each denial.
#[derive(Debug, Default)]
pub struct DenyAllGate;

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

/// Test-only gate that always approves. Do not use in production: it
/// defeats the entire confirmation layer.
#[derive(Debug, Default)]
pub struct AlwaysAllowGate;

#[async_trait]
impl ConfirmationGate for AlwaysAllowGate {
    async fn confirm(&self, _req: ConfirmationRequest) -> bool {
        true
    }
}

/// Cap on confirmation prompts in flight on one connection. A client
/// that never answers would otherwise leak a `oneshot::Sender` per ask.
pub const MAX_PENDING_CONFIRMS: usize = 32;

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
    pending: Mutex<HashMap<String, oneshot::Sender<bool>>>,
}

impl ConfirmRouter {
    pub fn new(request_id: String, wire: mpsc::Sender<Event>) -> Arc<Self> {
        Arc::new(Self {
            request_id,
            wire,
            pending: Mutex::new(HashMap::new()),
        })
    }

    /// Forward the prompt to the connected client and await the answer.
    /// Every failure mode (channel drop, cap reached, disconnect) is
    /// `false`.
    pub async fn ask(&self, req: ConfirmationRequest) -> bool {
        let confirm_id = uuid::Uuid::new_v4().to_string();
        let (tx, rx) = oneshot::channel();
        {
            let mut pending = self.pending.lock();
            if pending.len() >= MAX_PENDING_CONFIRMS {
                warn!(
                    target: "assistd::policy",
                    tool = %req.tool,
                    in_flight = pending.len(),
                    cap = MAX_PENDING_CONFIRMS,
                    "destructive command denied: pending-confirm cap reached"
                );
                return false;
            }
            pending.insert(confirm_id.clone(), tx);
        }

        let event = Event::ConfirmRequest {
            id: self.request_id.clone(),
            confirm_id: confirm_id.clone(),
            tool: req.tool.clone(),
            script: req.script.clone(),
            matched_pattern: req.matched_pattern.clone(),
        };
        if self.wire.send(event).await.is_err() {
            self.pending.lock().remove(&confirm_id);
            warn!(
                target: "assistd::policy",
                tool = %req.tool,
                "destructive command denied: client disconnected before confirm"
            );
            return false;
        }

        match rx.await {
            Ok(allow) => allow,
            Err(_) => {
                self.pending.lock().remove(&confirm_id);
                warn!(
                    target: "assistd::policy",
                    tool = %req.tool,
                    "destructive command denied: confirmation channel dropped"
                );
                false
            }
        }
    }

    /// Deliver a client's answer to the matching pending prompt. `Err`
    /// means no prompt with that id is in flight.
    pub fn route_response(&self, confirm_id: &str, allow: bool) -> Result<(), &'static str> {
        let sender = self.pending.lock().remove(confirm_id);
        match sender {
            Some(tx) => {
                let _ = tx.send(allow);
                Ok(())
            }
            None => Err("no pending confirm for this confirm_id"),
        }
    }
}

tokio::task_local! {
    /// The [`ConfirmRouter`] of the IPC connection whose request is
    /// being dispatched.
    pub static CONFIRM_ROUTER: Arc<ConfirmRouter>;
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

/// Case-insensitive literal-substring search over a bash script. Returns
/// the *first* matching pattern so the caller can surface it to the user
/// verbatim (per the denylist error-message contract).
///
/// Patterns are compared as lowercase; empty patterns are ignored (they
/// would match every script).
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

/// Whether any command segment of the shlex-tokenized script starts with
/// a configured destructive prefix. Returns the matched prefix.
///
/// Quoted arguments (`echo "rm -rf"`) stay single tokens and so do not
/// match `["rm", "-rf"]`. Unparseable scripts count as no match; bash
/// will surface the syntax error itself.
pub fn matches_destructive<'a>(script: &str, prefixes: &'a [Vec<String>]) -> Option<&'a [String]> {
    let tokens = shlex::split(script)?;
    if tokens.is_empty() {
        return None;
    }
    let lower: Vec<String> = tokens.iter().map(|t| t.to_ascii_lowercase()).collect();

    let separators: &[&str] = &["|", "||", ";", "&&", "&"];
    let mut anchors: Vec<usize> = vec![0];
    for (i, tok) in lower.iter().enumerate() {
        if separators.iter().any(|s| *s == tok) && i + 1 < lower.len() {
            anchors.push(i + 1);
        }
    }

    for prefix in prefixes {
        if prefix.is_empty() {
            continue;
        }
        let lower_prefix: Vec<String> = prefix.iter().map(|t| t.to_ascii_lowercase()).collect();
        for &anchor in &anchors {
            if anchor + lower_prefix.len() > lower.len() {
                continue;
            }
            if lower[anchor..anchor + lower_prefix.len()] == lower_prefix[..] {
                return Some(prefix.as_slice());
            }
        }
    }
    None
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

/// Resolved sandbox state, cached in an `Arc<SandboxInfo>` and shared across
/// every bash invocation so we pay the probe cost once.
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
    /// and D-Bus session sockets are unreachable. Used by `bash`.
    Default,
    /// Additionally bind `$XDG_RUNTIME_DIR` back over the `/run` tmpfs.
    /// Used by `wm open`, which launches GUI applications: without the
    /// Wayland and D-Bus sockets they fail to start at all.
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
/// Read-only root, writable `$HOME` + `/tmp`, standard `/dev` and `/proc`,
/// isolated pid/ipc/uts namespaces, dies with the daemon. Crucially *not*
/// `--unshare-net`: the assistant legitimately needs curl/pip/etc., so
/// network isolation is opt-in via `bwrap_extra_args`.
fn default_bwrap_flags() -> Vec<String> {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/".to_string());
    vec![
        "--ro-bind".into(),
        "/".into(),
        "/".into(),
        "--bind".into(),
        home.clone(),
        home.clone(),
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
        "--setenv".into(),
        "HOME".into(),
        home,
        "--setenv".into(),
        "PATH".into(),
        "/usr/local/bin:/usr/bin:/bin".into(),
    ]
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

/// Resolve `request` against the environment once, for the whole
/// process lifetime. `Auto` falls back to unsandboxed with a warning
/// when `bwrap` is missing; `Bwrap` errors instead.
pub fn probe_sandbox(
    request: SandboxRequest,
    extra_args: Vec<String>,
) -> anyhow::Result<Arc<SandboxInfo>> {
    let path_env = std::env::var_os("PATH").unwrap_or_default();
    probe_sandbox_with_path(request, extra_args, &path_env)
}

fn probe_sandbox_with_path(
    request: SandboxRequest,
    extra_args: Vec<String>,
    path_env: &std::ffi::OsStr,
) -> anyhow::Result<Arc<SandboxInfo>> {
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
            None => {
                anyhow::bail!(
                    "tools.bash.sandbox = \"bwrap\" but `bwrap` was not found on PATH. \
                     Install bubblewrap or change sandbox to \"auto\" / \"none\"."
                );
            }
        },
    };
    Ok(Arc::new(SandboxInfo { mode, extra_args }))
}

/// Minimal `which` over an explicit PATH value.
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

    #[tokio::test]
    async fn deny_all_gate_refuses_everything() {
        let gate = DenyAllGate;
        let req = ConfirmationRequest {
            tool: "bash".into(),
            script: "rm -rf foo".into(),
            matched_pattern: "rm -rf".into(),
        };
        assert!(!gate.confirm(req).await);
    }

    #[tokio::test]
    async fn always_allow_gate_approves_everything() {
        let gate = AlwaysAllowGate;
        let req = ConfirmationRequest {
            tool: "bash".into(),
            script: "rm -rf /".into(),
            matched_pattern: "rm -rf".into(),
        };
        assert!(gate.confirm(req).await);
    }

    #[tokio::test]
    async fn confirm_router_denies_when_pending_cap_reached() {
        // Drive `MAX_PENDING_CONFIRMS` into the pending map without
        // letting them resolve. The next `ask` must deny immediately
        // rather than insert and leak. Using a wide channel so the
        // wire send doesn't block (the cap path triggers before send).
        let (wire_tx, _wire_rx) = mpsc::channel::<Event>(MAX_PENDING_CONFIRMS * 2);
        let router = ConfirmRouter::new("req-cap-test".into(), wire_tx);

        // Pre-load the pending map up to the cap. Bypass `ask` so we
        // don't have to keep the wire sender alive; we just want the
        // router to think it's full.
        {
            let mut pending = router.pending.lock();
            for i in 0..MAX_PENDING_CONFIRMS {
                let (tx, _rx) = oneshot::channel();
                pending.insert(format!("preloaded-{i}"), tx);
            }
        }

        let req = ConfirmationRequest {
            tool: "bash".into(),
            script: "rm -rf foo".into(),
            matched_pattern: "rm -rf".into(),
        };
        let result = router.ask(req).await;
        assert!(!result, "ask must deny when pending cap is reached");
        // The cap path returns before insert, so the map size is unchanged.
        let pending_len = router.pending.lock().len();
        assert_eq!(
            pending_len, MAX_PENDING_CONFIRMS,
            "denied ask must not insert into pending"
        );
    }

    #[test]
    fn denylist_matches_exact_substring() {
        let patterns = vec!["rm -rf /".to_string()];
        assert_eq!(matches_denylist("rm -rf /", &patterns), Some("rm -rf /"));
    }

    #[test]
    fn denylist_matches_within_larger_script() {
        let patterns = vec!["mkfs".to_string()];
        assert_eq!(
            matches_denylist("sudo mkfs.ext4 /dev/sda1", &patterns),
            Some("mkfs")
        );
    }

    #[test]
    fn denylist_is_case_insensitive() {
        let patterns = vec!["rm -rf /".to_string()];
        assert_eq!(matches_denylist("RM -RF /", &patterns), Some("rm -rf /"));
    }

    #[test]
    fn denylist_no_match_returns_none() {
        let patterns = vec!["rm -rf /".to_string()];
        assert!(matches_denylist("ls -l /tmp", &patterns).is_none());
    }

    #[test]
    fn denylist_empty_pattern_is_ignored() {
        let patterns = vec!["".to_string(), "mkfs".to_string()];
        assert_eq!(matches_denylist("anything", &patterns), None);
        assert_eq!(matches_denylist("mkfs.ext4", &patterns), Some("mkfs"));
    }

    #[test]
    fn destructive_matches_command_prefix() {
        let prefixes = vec![vec!["rm".into(), "-rf".into()]];
        let m = matches_destructive("rm -rf foo", &prefixes).expect("match");
        assert_eq!(m, &["rm".to_string(), "-rf".to_string()]);
    }

    #[test]
    fn destructive_ignores_quoted_literal() {
        // The bash script `echo "rm -rf"` tokenizes to three tokens:
        // ["echo", "rm -rf"]; the second is a single quoted arg and must
        // not match the prefix ["rm", "-rf"].
        let prefixes = vec![vec!["rm".into(), "-rf".into()]];
        assert!(matches_destructive("echo \"rm -rf\"", &prefixes).is_none());
    }

    #[test]
    fn destructive_matches_second_command_in_chain() {
        let prefixes = vec![vec!["rm".into(), "-rf".into()]];
        // `touch foo && rm -rf bar`: the prefix should anchor at the
        // start of the second command.
        let m = matches_destructive("touch foo && rm -rf bar", &prefixes).expect("match");
        assert_eq!(m, &["rm".to_string(), "-rf".to_string()]);
    }

    #[test]
    fn destructive_matches_after_pipe() {
        let prefixes = vec![vec!["rm".into()]];
        let m = matches_destructive("ls | rm foo", &prefixes).expect("match");
        assert_eq!(m, &["rm".to_string()]);
    }

    #[test]
    fn destructive_no_match_on_distinct_command() {
        let prefixes = vec![vec!["rm".into(), "-rf".into()]];
        assert!(matches_destructive("ls -l /tmp", &prefixes).is_none());
    }

    #[test]
    fn destructive_unparseable_script_returns_none() {
        // Unterminated quote; shlex returns None.
        let prefixes = vec![vec!["rm".into(), "-rf".into()]];
        assert!(matches_destructive("rm -rf \"unterminated", &prefixes).is_none());
    }

    #[test]
    fn destructive_single_word_prefix_matches_only_as_first_token() {
        let prefixes = vec![vec!["shutdown".into()]];
        assert!(matches_destructive("shutdown -h now", &prefixes).is_some());
        // "shutdown" inside a quoted arg doesn't anchor at a command slot.
        assert!(matches_destructive("echo 'shutdown'", &prefixes).is_none());
    }

    fn bwrap_sandbox(extra_args: Vec<String>) -> SandboxInfo {
        SandboxInfo {
            mode: ResolvedSandboxMode::Bwrap {
                path: PathBuf::from("/usr/bin/bwrap"),
            },
            extra_args,
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
        let info = SandboxInfo {
            mode: ResolvedSandboxMode::None,
            extra_args: Vec::new(),
        };
        let cmd = info.command(SandboxAccess::Default, "firefox", ["--new-window", "a b"]);
        assert_eq!(cmd.as_std().get_program(), "firefox");
        assert_eq!(argv_of(&cmd), vec!["--new-window", "a b"]);
    }

    #[test]
    fn bwrap_command_puts_program_after_the_separator() {
        let info = bwrap_sandbox(vec!["--unshare-net".into()]);
        let cmd = info.command(SandboxAccess::Default, "firefox", ["--new-window"]);
        assert_eq!(cmd.as_std().get_program(), "/usr/bin/bwrap");
        let argv = argv_of(&cmd);
        let sep = argv.iter().position(|a| a == "--").expect("separator");
        assert_eq!(&argv[sep + 1..], &["firefox", "--new-window"]);
        // Operator extra args are the last thing before `--`, so they
        // override anything in the default profile.
        assert_eq!(argv[sep - 1], "--unshare-net");
    }

    #[test]
    fn default_access_leaves_the_run_tmpfs_empty() {
        let info = bwrap_sandbox(Vec::new());
        let argv = argv_of(&info.command(SandboxAccess::Default, "bash", ["-c", "true"]));
        let runtime_dir = std::env::var("XDG_RUNTIME_DIR").unwrap_or_default();
        assert!(
            runtime_dir.is_empty() || !argv.contains(&runtime_dir),
            "default profile must not bind the session runtime dir: {argv:?}"
        );
    }

    #[test]
    fn session_bind_lands_after_the_run_tmpfs() {
        // Order matters: a bind listed before `--tmpfs /run` would be
        // shadowed by the tmpfs and the socket would be invisible.
        let dir = std::env::temp_dir();
        let dir_str = dir.to_string_lossy().into_owned();
        let flags = session_bind_flags_for(Some(dir_str.clone()));
        assert_eq!(flags, vec!["--bind".to_string(), dir_str.clone(), dir_str]);

        let profile = default_bwrap_flags();
        let tmpfs = profile
            .iter()
            .position(|f| f == "--tmpfs")
            .expect("default profile mounts a tmpfs");
        assert_eq!(profile[tmpfs + 1], "/run");
    }

    #[test]
    fn session_bind_is_skipped_when_runtime_dir_is_unusable() {
        assert!(session_bind_flags_for(None).is_empty());
        assert!(
            session_bind_flags_for(Some("/nonexistent/assistd-runtime-dir".into())).is_empty(),
            "a stale XDG_RUNTIME_DIR must not be passed to bwrap, which aborts on a \
             missing bind source"
        );
    }

    #[test]
    fn probe_sandbox_none_always_returns_none() {
        let info = probe_sandbox(SandboxRequest::None, Vec::new()).expect("probe none");
        assert!(matches!(info.mode, ResolvedSandboxMode::None));
    }

    #[test]
    fn probe_sandbox_bwrap_missing_fails_startup() {
        let result =
            probe_sandbox_with_path(SandboxRequest::Bwrap, Vec::new(), std::ffi::OsStr::new(""));
        assert!(
            result.is_err(),
            "expected bwrap probe to fail with empty PATH"
        );
    }

    #[test]
    fn sandbox_info_none_helper_is_usable() {
        let info = SandboxInfo::none();
        assert!(matches!(info.mode, ResolvedSandboxMode::None));
        assert!(info.extra_args.is_empty());
    }
}
