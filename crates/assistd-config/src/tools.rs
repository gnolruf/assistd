//! Tool subsystem configuration: scratch directory, output limits and per-command policy.

use std::num::{NonZeroU32, NonZeroU64};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::defaults::{
    DEFAULT_BASH_TIMEOUT_SECS, DEFAULT_TOOLS_MAX_KB, DEFAULT_TOOLS_MAX_LINES,
    DEFAULT_TOOLS_SCRATCH_RETENTION_DAYS, default_tools_overflow_dir, default_tools_scratch_dir,
    default_writable_paths,
};
use crate::settings::{DescribeSettings, describe_fields};

/// Tools subsystem configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct ToolsConfig {
    pub scratch: ToolsScratchConfig,
    pub output: ToolsOutputConfig,
    pub bash: ToolsBashConfig,
    pub write: ToolsWriteConfig,
    /// Whether the chat's permission-mode toggle offers Auto, where the safety judge answers
    /// for the user on commands the path guards and edit scope do not settle. `false` keeps
    /// every connection in manual or accept-edits mode.
    pub allow_auto_mode: bool,
}

impl DescribeSettings for ToolsConfig {
    describe_fields! {
        scratch => nested,
        output => nested,
        bash => nested,
        write => nested,
        allow_auto_mode => (
            "offer auto mode",
            "Let the chat toggle into Auto, where a model judge approves commands"
        ),
    }
}

/// The directory `bash` scripts and the other commands share.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ToolsScratchConfig {
    /// Bound writable into the sandbox, and writable by `write` without
    /// asking. Created owner-only on daemon startup. Must be absolute or
    /// start with `~/`, and must not be `/` or hold `$HOME` or assistd's config
    /// or data directory. Defaults to `$XDG_RUNTIME_DIR/assistd/scratch`, else
    /// under the user cache dir.
    #[serde(deserialize_with = "crate::home_path::deserialize")]
    pub dir: PathBuf,
    /// Days an entry in `dir` is kept after it last changed; older ones,
    /// and directories that leaves empty, are removed on daemon startup when
    /// assistd created `dir`. 0 empties the directory at every startup.
    pub retention_days: u32,
}

impl DescribeSettings for ToolsScratchConfig {
    describe_fields! {
        dir => (
            "scratch folder",
            "Folder tools may write to freely; absolute or starting with ~/",
            advanced
        ),
        retention_days => (
            "keep scratch files (days)",
            "Days scratch files are kept; 0 empties it at every startup",
            advanced
        ),
    }
}

impl Default for ToolsScratchConfig {
    fn default() -> Self {
        Self {
            dir: default_tools_scratch_dir(),
            retention_days: DEFAULT_TOOLS_SCRATCH_RETENTION_DAYS,
        }
    }
}

/// Limits on every tool result before it reaches the LLM; the excess
/// spills to a file whose path the model is given.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ToolsOutputConfig {
    /// Max lines of one tool result, notices included.
    pub max_lines: NonZeroU32,
    /// Max size of one tool result in KB, notices included. At about 256
    /// tokens per KB, must not exceed a quarter of `chat.max_history_tokens`.
    pub max_kb: NonZeroU32,
    /// Spill directory for overflow (`cmd-<n>.txt` for `run`,
    /// `<tool>-<n>.txt` for any other tool); made owner-only on daemon
    /// startup, when earlier spill files are removed and nothing else is.
    /// Each new spill deletes the oldest ones beyond the newest 64 files
    /// or 128 MiB. Must be absolute or start with `~/`, and must not be `/`
    /// or hold `$HOME` or assistd's config or data directory. Defaults to
    /// `$XDG_RUNTIME_DIR/assistd/output`, else under the user cache dir.
    #[serde(deserialize_with = "crate::home_path::deserialize")]
    pub overflow_dir: PathBuf,
}

impl DescribeSettings for ToolsOutputConfig {
    describe_fields! {
        max_lines => ("max result lines", "Most lines of one tool result", advanced),
        max_kb => (
            "max result size (KB)",
            "Largest tool result; at most a quarter of the history budget",
            advanced
        ),
        overflow_dir => ("overflow folder", "Where cut-off tool output is saved in full", advanced),
    }
}

impl Default for ToolsOutputConfig {
    fn default() -> Self {
        Self {
            max_lines: DEFAULT_TOOLS_MAX_LINES,
            max_kb: DEFAULT_TOOLS_MAX_KB,
            overflow_dir: default_tools_overflow_dir(),
        }
    }
}

impl ToolsOutputConfig {
    /// `max_kb` expressed in bytes.
    pub fn max_bytes(&self) -> usize {
        (self.max_kb.get() as usize) * 1024
    }
}

/// Sandbox mode for spawned commands. Tools never run unsandboxed: without
/// bubblewrap the model is offered no tools at all.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum BashSandboxMode {
    /// Use bubblewrap if `bwrap` is on `PATH` at startup; otherwise disable
    /// every tool.
    #[default]
    Auto,
    /// Require bubblewrap; startup fails without `bwrap`.
    Bwrap,
    /// Never sandbox, so every tool is disabled.
    None,
}

/// Policy for the scripts `bash` runs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ToolsBashConfig {
    /// `bash` timeout in seconds; on expiry the process group is killed
    /// (exit 137).
    pub timeout_secs: NonZeroU64,
    /// How spawned commands are sandboxed.
    pub sandbox: BashSandboxMode,
    /// Extra bubblewrap arguments, inserted before the trailing `--`: e.g.
    /// `["--share-net"]` giving `bash` the network (it has none by default;
    /// needs Linux 6.12+ Landlock scoping to keep abstract sockets out, or
    /// tools are disabled), or a `--bind` making a dot entry of `$HOME`
    /// writable (read-only).
    pub bwrap_extra_args: Vec<String>,
}

impl DescribeSettings for ToolsBashConfig {
    describe_fields! {
        timeout_secs => ("command timeout (s)", "Longest a bash command runs before it is killed"),
        sandbox => ("sandbox", "auto uses bubblewrap if installed; tools are disabled unsandboxed"),
        bwrap_extra_args => (
            "extra bubblewrap arguments",
            "Extra bubblewrap arguments, e.g. --share-net for network access",
            advanced
        ),
    }
}

impl Default for ToolsBashConfig {
    fn default() -> Self {
        Self {
            timeout_secs: DEFAULT_BASH_TIMEOUT_SECS,
            sandbox: BashSandboxMode::default(),
            bwrap_extra_args: Vec::new(),
        }
    }
}

/// Write-command policy.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ToolsWriteConfig {
    /// Non-empty path prefixes `write` may create files under; anywhere else
    /// outside `tools.scratch.dir` is refused. Symlinks and dot entries at any
    /// depth below a prefix are refused (list one to allow it). Writes here ask
    /// first; scratch is writable without asking. `~` and `~/` expand to
    /// `$HOME`; any other relative entry errors, missing ones are dropped.
    pub writable_paths: Vec<String>,
}

impl DescribeSettings for ToolsWriteConfig {
    describe_fields! {
        writable_paths => (
            "writable folders",
            "Folders write may create files under, after asking; nowhere else outside scratch"
        ),
    }
}

impl Default for ToolsWriteConfig {
    fn default() -> Self {
        Self {
            writable_paths: default_writable_paths(),
        }
    }
}
