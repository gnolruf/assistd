use std::num::{NonZeroU32, NonZeroU64};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::defaults::{
    DEFAULT_BASH_TIMEOUT_SECS, DEFAULT_TOOLS_MAX_KB, DEFAULT_TOOLS_MAX_LINES,
    DEFAULT_TOOLS_OVERFLOW_DIR, default_bash_allowed_programs, default_bash_denylist,
    default_bash_destructive_patterns, default_writable_paths,
};

/// Tools subsystem configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct ToolsConfig {
    pub output: ToolsOutputConfig,
    pub bash: ToolsBashConfig,
    pub write: ToolsWriteConfig,
    pub screenshot: ToolsScreenshotConfig,
}

/// Limits on a `run` result before it reaches the LLM; the excess spills
/// to a file whose path the model is given.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ToolsOutputConfig {
    /// Max output lines shown to the LLM.
    pub max_lines: NonZeroU32,
    /// Max size of the shown head, in KB.
    pub max_kb: NonZeroU32,
    /// Spill directory for overflow (`cmd-<n>.txt`); recreated empty on
    /// daemon startup. Must not be empty.
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
    /// `max_kb` expressed in bytes.
    pub fn max_bytes(&self) -> usize {
        (self.max_kb.get() as usize) * 1024
    }
}

/// Sandbox mode for spawned commands.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum BashSandboxMode {
    /// Use bubblewrap if `bwrap` is on `PATH` at startup; otherwise warn
    /// and run unsandboxed.
    #[default]
    Auto,
    /// Require bubblewrap; startup fails without `bwrap`.
    Bwrap,
    /// Never sandbox.
    None,
}

/// Policy for every spawned command (`bash` and `wm open`). A command runs
/// without confirmation only when every program it can run is allowed and
/// it matches no destructive pattern; the denylist refuses outright.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ToolsBashConfig {
    /// `bash` timeout in seconds; on expiry the process group is killed
    /// (exit 137). Does not apply to `wm open`.
    pub timeout_secs: NonZeroU64,
    /// Case-insensitive substrings that reject a script before spawn (exit
    /// 126). An entry ending in a non-alphanumeric must also end a shell
    /// word: `"rm -rf /"` rejects `rm -rf /` but not `rm -rf /tmp/build`.
    pub denylist: Vec<String>,
    /// Programs that run without confirmation: absolute paths, or bare
    /// names resolved on `PATH` that count only if the file is not
    /// user-writable or is the one approved. "Always allow" approvals
    /// persist in `allowed_programs.toml` beside the config file.
    pub allowed_programs: Vec<String>,
    /// Commands that need confirmation, and are refused when no one can
    /// confirm. Each is a command name (matching any path ending in it)
    /// then arguments that must all appear, in any order; `a|b` lists
    /// alternatives. `-rf` matches those short flags in any cluster,
    /// `--force` also its abbreviations and `--force=…`, `of=` any argument
    /// with that prefix; other words match exactly. Quoted text is one
    /// word, so `"rm -r|--recursive"` matches `rm -vfr x` but not
    /// `echo "rm -rf"`.
    pub destructive_patterns: Vec<String>,
    pub sandbox: BashSandboxMode,
    /// Extra bubblewrap arguments, inserted before the trailing `--`: e.g.
    /// `["--unshare-net"]` (the network is shared by default), or a `--bind`
    /// making a dot entry of `$HOME` writable (they are read-only).
    pub bwrap_extra_args: Vec<String>,
}

impl Default for ToolsBashConfig {
    fn default() -> Self {
        Self {
            timeout_secs: DEFAULT_BASH_TIMEOUT_SECS,
            denylist: default_bash_denylist(),
            allowed_programs: default_bash_allowed_programs(),
            destructive_patterns: default_bash_destructive_patterns(),
            sandbox: BashSandboxMode::default(),
            bwrap_extra_args: Vec::new(),
        }
    }
}

/// Write-command policy.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ToolsWriteConfig {
    /// Non-empty path prefixes `write` may create files under; symlinks and dot
    /// entries directly inside a prefix are refused (list one to allow it).
    /// `~` / `~user` expand; relative entries error, missing ones are dropped.
    pub writable_paths: Vec<String>,
}

impl Default for ToolsWriteConfig {
    fn default() -> Self {
        Self {
            writable_paths: default_writable_paths(),
        }
    }
}

/// Screenshot capture backend.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum ScreenshotBackend {
    /// Wayland when `XDG_SESSION_TYPE`/`WAYLAND_DISPLAY` say so, else X11.
    #[default]
    Auto,
    /// `maim`; `--focused` also needs `xdotool`.
    X11,
    /// `grim`; `--focused` needs sway or Hyprland.
    Wayland,
}

/// Screenshot-command settings.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ToolsScreenshotConfig {
    pub backend: ScreenshotBackend,
}
