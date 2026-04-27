#![allow(unsafe_code)] // libc / env / fd primitives — each unsafe block is locally justified

//! Command-execution policy: confirmation gates, pattern matchers, and
//! sandbox probing.
//!
//! The types here are consumed by [`crate::commands::BashCommand`] (denylist
//! check, destructive-pattern confirmation, bwrap wrapping) and
//! [`crate::commands::WriteCommand`] (writable-path allowlist — resolved at
//! build time by the caller, not by this module). They are deliberately
//! decoupled from `assistd-core::config` to avoid a circular crate
//! dependency: the caller in `assistd-core::build_tools` constructs the
//! primitive policy types here from its own `Config`.
//!
//! Honest caveat on syntactic checks: the denylist and destructive-pattern
//! list are backstops for *obvious* dangerous invocations (`rm -rf /`,
//! `mkfs`, …). They can be defeated by a sufficiently clever caller
//! (variable expansion, here-docs, command substitution), so they are not
//! the real defense — the bwrap sandbox is.

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use tracing::{info, warn};

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

/// A policy authority that decides whether a destructive command may run.
///
/// Two built-in implementations are provided:
/// - [`DenyAllGate`]: the headless/IPC default. Always returns `false`. Logs
///   a warn when it denies so the operator can see why a command was blocked.
/// - [`AlwaysAllowGate`]: a test-only bypass.
///
/// Real interactive use wires up a TUI-side gate that forwards the request
/// to the user through an `mpsc` channel and awaits a `oneshot` response.
/// The trait is `Send + Sync + 'static` so the same `Arc<dyn ConfirmationGate>`
/// can be handed to every command.
#[async_trait]
pub trait ConfirmationGate: Send + Sync + 'static {
    /// Ask for confirmation. `true` = proceed, `false` = cancel.
    ///
    /// Implementations must convert *every* failure mode (channel drop, UI
    /// shutdown, timeout) into `false` so the agent loop never hangs.
    async fn confirm(&self, req: ConfirmationRequest) -> bool;
}

/// Default gate for headless / IPC-connected paths. Never approves.
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

/// Tokenize the bash script with `shlex` and check whether any token
/// subsequence matches a configured destructive prefix. The first token of
/// a prefix must equal the first token of a script segment (roughly, a
/// command invocation), with subsequent prefix tokens appearing in order.
///
/// Quoted arguments (`echo "rm -rf"`) are left as single tokens by `shlex`
/// and therefore do **not** match `["rm", "-rf"]`. Unparseable scripts
/// (unbalanced quotes, etc.) are treated as "no match" so we don't block
/// legitimate constructs; the subsequent bash invocation will surface the
/// syntax error itself.
///
/// Returns the matched prefix (for display in the confirmation prompt) or
/// `None`.
pub fn matches_destructive<'a>(script: &str, prefixes: &'a [Vec<String>]) -> Option<&'a [String]> {
    let tokens = shlex::split(script)?;
    if tokens.is_empty() {
        return None;
    }
    let lower: Vec<String> = tokens.iter().map(|t| t.to_ascii_lowercase()).collect();

    // Split into "commands": anything that isn't a shell separator starts a
    // new command; these are the positions we try to anchor prefixes at.
    // Separators include `;`, `&&`, `||`, `|`, and `&`.
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

/// How the caller requested sandboxing for bash. Derived from
/// `config::BashSandboxMode` at startup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SandboxRequest {
    /// Use bwrap if found on `PATH`; fall back to unsandboxed with a warn.
    Auto,
    /// Require bwrap — fail startup if missing.
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

/// Cached sandbox configuration threaded into every `BashCommand`.
#[derive(Debug)]
pub struct SandboxInfo {
    pub mode: ResolvedSandboxMode,
    /// Extra args appended verbatim to the bwrap invocation before `--`.
    pub extra_args: Vec<String>,
}

impl SandboxInfo {
    /// Convenience constructor for tests that don't care about sandboxing.
    pub fn none() -> Arc<Self> {
        Arc::new(Self {
            mode: ResolvedSandboxMode::None,
            extra_args: Vec::new(),
        })
    }
}

/// Probe the environment once at daemon startup and return a shared
/// [`SandboxInfo`] for the entire process lifetime.
///
/// Behaviour by `request`:
/// - `None` → always returns [`ResolvedSandboxMode::None`].
/// - `Auto` → returns `Bwrap` if `bwrap` is found on `PATH`; otherwise
///   logs a `warn!` and returns `None` (degraded mode per the AC).
/// - `Bwrap` → returns `Bwrap` if found; otherwise returns an error so
///   daemon startup fails fast.
pub fn probe_sandbox(
    request: SandboxRequest,
    extra_args: Vec<String>,
) -> anyhow::Result<Arc<SandboxInfo>> {
    let mode = match request {
        SandboxRequest::None => {
            info!(target: "assistd::policy", "bash sandbox: disabled by config");
            ResolvedSandboxMode::None
        }
        SandboxRequest::Auto => match which_in_path("bwrap") {
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
        SandboxRequest::Bwrap => match which_in_path("bwrap") {
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

/// Minimal `which`: first PATH entry that contains an executable file with
/// the given basename. Avoids a dependency on the `which` crate.
fn which_in_path(name: &str) -> Option<PathBuf> {
    let path_env = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_env) {
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
        // ["echo", "rm -rf"] — the second is a single quoted arg and must
        // not match the prefix ["rm", "-rf"].
        let prefixes = vec![vec!["rm".into(), "-rf".into()]];
        assert!(matches_destructive("echo \"rm -rf\"", &prefixes).is_none());
    }

    #[test]
    fn destructive_matches_second_command_in_chain() {
        let prefixes = vec![vec!["rm".into(), "-rf".into()]];
        // `touch foo && rm -rf bar` — the prefix should anchor at the
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
        // Unterminated quote — shlex returns None.
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

    #[test]
    fn probe_sandbox_none_always_returns_none() {
        let info = probe_sandbox(SandboxRequest::None, Vec::new()).expect("probe none");
        assert!(matches!(info.mode, ResolvedSandboxMode::None));
    }

    #[test]
    fn probe_sandbox_bwrap_missing_fails_startup() {
        // Force an empty PATH so bwrap can't be found.
        let saved = std::env::var_os("PATH");
        // SAFETY: tests run single-threaded by default for std::env mutation;
        // we restore on exit. Accept the test smell here because isolating
        // the probe otherwise requires injecting PATH, which would clutter
        // the public API.
        unsafe {
            std::env::set_var("PATH", "");
        }
        let result = probe_sandbox(SandboxRequest::Bwrap, Vec::new());
        if let Some(p) = saved {
            unsafe {
                std::env::set_var("PATH", p);
            }
        } else {
            unsafe {
                std::env::remove_var("PATH");
            }
        }
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
