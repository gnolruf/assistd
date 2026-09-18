use std::num::{NonZeroU32, NonZeroU64};
use std::path::PathBuf;

use crate::defaults::{
    DEFAULT_BASH_TIMEOUT_SECS, DEFAULT_TOOLS_MAX_KB, DEFAULT_TOOLS_MAX_LINES,
    DEFAULT_TOOLS_OVERFLOW_DIR, default_bash_denylist, default_bash_destructive_patterns,
    default_writable_paths,
};
use serde::{Deserialize, Serialize};

/// Tools subsystem configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default, deny_unknown_fields)]
pub struct ToolsConfig {
    /// Output presentation limits applied before handing results to the LLM.
    pub output: ToolsOutputConfig,
    /// Bash command execution policy (timeout, denylist, sandbox mode).
    pub bash: ToolsBashConfig,
    /// File-write command policy (allowlist of writable path prefixes).
    pub write: ToolsWriteConfig,
    /// Screenshot capture settings (backend selector, timeout).
    pub screenshot: ToolsScreenshotConfig,
}

/// Limits applied to a `run` result before it is handed to the LLM.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct ToolsOutputConfig {
    /// Max lines of stdout surfaced to the LLM before overflow spill.
    pub max_lines: NonZeroU32,
    /// Max bytes of the truncated head, in KB.
    pub max_kb: NonZeroU32,
    /// Directory where overflow output is spilled as `cmd-<n>.txt`.
    /// Cleared + recreated on daemon startup.
    pub overflow_dir: PathBuf,
}

impl Default for ToolsOutputConfig {
    fn default() -> Self {
        Self {
            max_lines: DEFAULT_TOOLS_MAX_LINES,
            max_kb: DEFAULT_TOOLS_MAX_KB,
            overflow_dir: DEFAULT_TOOLS_OVERFLOW_DIR.into(),
        }
    }
}

impl ToolsOutputConfig {
    /// `max_kb` expressed in bytes, ready to pass to the presentation layer.
    pub fn max_bytes(&self) -> usize {
        (self.max_kb.get() as usize) * 1024
    }
}

/// Sandbox mode for bash subprocess execution.
///
/// * `Auto`: use bubblewrap if `bwrap` is found on `PATH` at daemon startup;
///   log a warn and run unsandboxed if not.
/// * `Bwrap`: require bubblewrap; fail daemon startup if `bwrap` is missing.
/// * `None`: never wrap; run bash directly under the daemon's own user.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum BashSandboxMode {
    #[default]
    Auto,
    Bwrap,
    None,
}

/// Bash-command policy. The denylist and destructive patterns are
/// syntactic backstops for the obvious cases; the sandbox is the real
/// defence.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct ToolsBashConfig {
    /// Subprocess timeout in seconds. Must be > 0. Exceeding the timeout
    /// kills the process group and returns exit 137.
    pub timeout_secs: NonZeroU64,
    /// Literal substrings that, if present in a bash script (case-insensitive),
    /// cause immediate rejection before spawn. Use for patterns that should
    /// never be executed under any circumstances.
    pub denylist: Vec<String>,
    /// Shell-tokenized word prefixes that trigger interactive confirmation
    /// before executing (when a confirmation gate is wired up) or reject by
    /// default over IPC. Example: `"rm -rf"` matches `rm -rf foo` but not
    /// `echo "rm -rf"`.
    pub destructive_patterns: Vec<String>,
    /// Sandbox mode. See [`BashSandboxMode`].
    pub sandbox: BashSandboxMode,
    /// Extra arguments appended to the bubblewrap invocation (before the
    /// trailing `--`). Useful for widening binds (e.g.
    /// `["--bind", "/srv", "/srv"]`) or tightening the sandbox (e.g.
    /// `["--unshare-net"]`).
    pub bwrap_extra_args: Vec<String>,
}

impl Default for ToolsBashConfig {
    fn default() -> Self {
        Self {
            timeout_secs: DEFAULT_BASH_TIMEOUT_SECS,
            denylist: default_bash_denylist(),
            destructive_patterns: default_bash_destructive_patterns(),
            sandbox: BashSandboxMode::default(),
            bwrap_extra_args: Vec::new(),
        }
    }
}

/// Write-command policy: the allowlist of path prefixes under which the
/// `write` command is permitted to create or overwrite files. Attempts
/// outside every entry return exit 126.
///
/// Supports `~` / `~user` expansion. Relative paths are rejected outright
/// because the daemon's cwd is not a meaningful anchor. Non-existent
/// allowlist entries are dropped with a warning at daemon startup.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct ToolsWriteConfig {
    /// Path prefixes (supporting `~` expansion) under which the `write` command may operate.
    pub writable_paths: Vec<String>,
}

impl Default for ToolsWriteConfig {
    fn default() -> Self {
        Self {
            writable_paths: default_writable_paths(),
        }
    }
}

/// Screenshot capture backend selector. `Auto` picks Wayland when
/// `XDG_SESSION_TYPE`/`WAYLAND_DISPLAY` is set, X11 otherwise. Override
/// only when the auto-detect picks the wrong tool on a hybrid session.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum ScreenshotBackend {
    #[default]
    Auto,
    X11,
    Wayland,
}

/// Screenshot-command policy. Bytes are kept in memory and never written
/// to disk by this command.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct ToolsScreenshotConfig {
    /// Which capture backend to use.
    pub backend: ScreenshotBackend,
}
