use crate::defaults::{
    DEFAULT_BASH_TIMEOUT_SECS, DEFAULT_TOOLS_MAX_KB, DEFAULT_TOOLS_MAX_LINES,
    DEFAULT_TOOLS_OVERFLOW_DIR, default_bash_denylist, default_bash_destructive_patterns,
    default_writable_paths,
};
use serde::{Deserialize, Serialize};

/// Tools subsystem configuration. Nested container so future tool-related
/// knobs (sandboxing, per-command timeouts, etc.) can slot in alongside the
/// output-presentation limits.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ToolsConfig {
    #[serde(default)]
    pub output: ToolsOutputConfig,
    #[serde(default)]
    pub bash: ToolsBashConfig,
    #[serde(default)]
    pub write: ToolsWriteConfig,
}

/// Layer-2 presentation limits applied to the final output of `run` before
/// it is handed to the LLM.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolsOutputConfig {
    /// Max lines of stdout surfaced to the LLM before overflow spill.
    #[serde(default = "default_tools_max_lines")]
    pub max_lines: u32,
    /// Max bytes of the truncated head, in KB.
    #[serde(default = "default_tools_max_kb")]
    pub max_kb: u32,
    /// Directory where overflow output is spilled as `cmd-<n>.txt`.
    /// Cleared + recreated on daemon startup.
    #[serde(default = "default_tools_overflow_dir")]
    pub overflow_dir: String,
}

impl Default for ToolsOutputConfig {
    fn default() -> Self {
        Self {
            max_lines: DEFAULT_TOOLS_MAX_LINES,
            max_kb: DEFAULT_TOOLS_MAX_KB,
            overflow_dir: DEFAULT_TOOLS_OVERFLOW_DIR.to_string(),
        }
    }
}

impl ToolsOutputConfig {
    /// `max_kb` expressed in bytes, ready to pass to the presentation layer.
    pub fn max_bytes(&self) -> usize {
        (self.max_kb as usize) * 1024
    }
}

fn default_tools_max_lines() -> u32 {
    DEFAULT_TOOLS_MAX_LINES
}

fn default_tools_max_kb() -> u32 {
    DEFAULT_TOOLS_MAX_KB
}

fn default_tools_overflow_dir() -> String {
    DEFAULT_TOOLS_OVERFLOW_DIR.to_string()
}

/// Sandbox mode for bash subprocess execution.
///
/// * `Auto` — use bubblewrap if `bwrap` is found on `PATH` at daemon startup;
///   log a warn and run unsandboxed if not.
/// * `Bwrap` — require bubblewrap; fail daemon startup if `bwrap` is missing.
/// * `None` — never wrap; run bash directly under the daemon's own user.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum BashSandboxMode {
    #[default]
    Auto,
    Bwrap,
    None,
}

/// Bash-command policy: timeout ceiling, denylist of literal substrings that
/// are rejected before spawn, destructive-pattern prefixes that require user
/// confirmation, and sandbox mode.
///
/// Honest caveat: once bash is available, any syntactic pre-check can be
/// defeated by a sufficiently clever caller (variable expansion, here-docs,
/// command substitution). The denylist and destructive patterns are a
/// backstop for the *obvious* cases; the sandbox is the real defense.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolsBashConfig {
    /// Subprocess timeout in seconds. Must be > 0. Exceeding the timeout
    /// kills the process group and returns exit 137.
    #[serde(default = "default_bash_timeout_secs")]
    pub timeout_secs: u64,
    /// Literal substrings that, if present in a bash script (case-insensitive),
    /// cause immediate rejection before spawn. Use for patterns that should
    /// never be executed under any circumstances.
    #[serde(default = "default_bash_denylist_fn")]
    pub denylist: Vec<String>,
    /// Shell-tokenized word prefixes that trigger interactive confirmation
    /// before executing (when a confirmation gate is wired up) or reject by
    /// default over IPC. Example: `"rm -rf"` matches `rm -rf foo` but not
    /// `echo "rm -rf"`.
    #[serde(default = "default_bash_destructive_patterns_fn")]
    pub destructive_patterns: Vec<String>,
    /// Sandbox mode. See [`BashSandboxMode`].
    #[serde(default)]
    pub sandbox: BashSandboxMode,
    /// Extra arguments appended to the bubblewrap invocation (before the
    /// trailing `--`). Useful for widening binds (e.g.
    /// `["--bind", "/srv", "/srv"]`) or tightening the sandbox (e.g.
    /// `["--unshare-net"]`).
    #[serde(default)]
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

fn default_bash_timeout_secs() -> u64 {
    DEFAULT_BASH_TIMEOUT_SECS
}

fn default_bash_denylist_fn() -> Vec<String> {
    default_bash_denylist()
}

fn default_bash_destructive_patterns_fn() -> Vec<String> {
    default_bash_destructive_patterns()
}

/// Write-command policy: the allowlist of path prefixes under which the
/// `write` command is permitted to create or overwrite files. Attempts
/// outside every entry return exit 126.
///
/// Supports `~` / `~user` expansion. Relative paths are rejected outright
/// because the daemon's cwd is not a meaningful anchor. Non-existent
/// allowlist entries are dropped with a warning at daemon startup.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolsWriteConfig {
    #[serde(default = "default_writable_paths_fn")]
    pub writable_paths: Vec<String>,
}

impl Default for ToolsWriteConfig {
    fn default() -> Self {
        Self {
            writable_paths: default_writable_paths(),
        }
    }
}

fn default_writable_paths_fn() -> Vec<String> {
    default_writable_paths()
}
