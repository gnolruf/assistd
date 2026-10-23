use std::num::{NonZeroU32, NonZeroU64};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::defaults::{
    DEFAULT_BASH_TIMEOUT_SECS, DEFAULT_TOOLS_MAX_KB, DEFAULT_TOOLS_MAX_LINES,
    DEFAULT_TOOLS_SCRATCH_RETENTION_DAYS, default_bash_allowed_programs, default_bash_denylist,
    default_bash_destructive_patterns, default_tools_overflow_dir, default_tools_scratch_dir,
    default_writable_paths,
};
use crate::settings::DescribeSettings;

/// Tools subsystem configuration.
#[derive(DescribeSettings, Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct ToolsConfig {
    #[setting(nested)]
    pub scratch: ToolsScratchConfig,
    #[setting(nested)]
    pub output: ToolsOutputConfig,
    #[setting(nested)]
    pub bash: ToolsBashConfig,
    #[setting(nested)]
    pub write: ToolsWriteConfig,
}

/// The directory `bash` scripts and the other commands share.
#[derive(DescribeSettings, Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ToolsScratchConfig {
    /// Bound writable into the sandbox, and writable by `write` without
    /// asking. Created owner-only on daemon startup. Must be absolute or
    /// start with `~/`. Defaults to `$XDG_RUNTIME_DIR/assistd/scratch`, else
    /// under the user cache dir.
    #[serde(deserialize_with = "crate::home_path::deserialize")]
    #[setting(
        label = "scratch folder",
        description = "Folder tools may write to freely; absolute or starting with ~/",
        advanced
    )]
    pub dir: PathBuf,
    /// Days an entry in `dir` is kept after it last changed; older ones,
    /// and directories that leaves empty, are removed on daemon startup.
    /// 0 empties the directory at every startup.
    #[setting(
        label = "keep scratch files (days)",
        description = "Days scratch files are kept; 0 empties it at every startup",
        advanced
    )]
    pub retention_days: u32,
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
#[derive(DescribeSettings, Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ToolsOutputConfig {
    /// Max lines of one tool result, notices included.
    #[setting(
        label = "max result lines",
        description = "Most lines of one tool result",
        advanced
    )]
    pub max_lines: NonZeroU32,
    /// Max size of one tool result in KB, notices included. At about 256
    /// tokens per KB, must not exceed a quarter of `chat.max_history_tokens`.
    #[setting(
        label = "max result size (KB)",
        description = "Largest tool result; at most a quarter of the history budget",
        advanced
    )]
    pub max_kb: NonZeroU32,
    /// Spill directory for overflow (`cmd-<n>.txt` for `run`,
    /// `<tool>-<n>.txt` for any other tool); made owner-only on daemon
    /// startup, when earlier spill files are removed and nothing else is.
    /// Each new spill deletes the oldest ones beyond the newest 64 files
    /// or 128 MiB. Must be absolute or start with `~/`. Defaults to
    /// `$XDG_RUNTIME_DIR/assistd/output`, else under the user cache dir.
    #[serde(deserialize_with = "crate::home_path::deserialize")]
    #[setting(
        label = "overflow folder",
        description = "Where cut-off tool output is saved in full",
        advanced
    )]
    pub overflow_dir: PathBuf,
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

/// Policy for the scripts `bash` runs. A script runs
/// without confirmation only when every program it can run is allowed and
/// it matches no destructive pattern; the denylist refuses outright.
#[derive(DescribeSettings, Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ToolsBashConfig {
    /// `bash` timeout in seconds; on expiry the process group is killed
    /// (exit 137).
    #[setting(
        label = "command timeout (s)",
        description = "Longest a bash command runs before it is killed"
    )]
    pub timeout_secs: NonZeroU64,
    /// Case-insensitive substrings that reject a script before spawn (exit
    /// 126). An entry ending in a non-alphanumeric must also end a shell
    /// word: `"rm -rf /"` rejects `rm -rf /` but not `rm -rf /tmp/build`.
    #[setting(
        label = "blocked commands",
        description = "Case-insensitive substrings that reject a script outright",
        advanced
    )]
    pub denylist: Vec<String>,
    /// Programs that run without confirmation: absolute paths, or bare
    /// names resolved on `PATH` that count only if the file is not
    /// user-writable or is the one approved. "Always allow" approvals
    /// persist in `allowed_programs.toml` beside the config file.
    #[setting(
        label = "programs run without asking",
        description = "Programs that run without asking for confirmation",
        advanced
    )]
    pub allowed_programs: Vec<String>,
    /// Commands that need confirmation, and are refused when no one can
    /// confirm. Each is a command name (matching any path ending in it)
    /// then arguments that must all appear, in any order; `a|b` lists
    /// alternatives. `-rf` matches those short flags in any cluster,
    /// `--force` also its abbreviations and `--force=…`, `of=` any argument
    /// with that prefix; other words match exactly. Quoted text is one
    /// word, so `"rm -r|--recursive"` matches `rm -vfr x` but not
    /// `echo "rm -rf"`.
    #[setting(
        label = "commands that always ask",
        description = "Commands that always ask, and are refused when no one can confirm",
        advanced
    )]
    pub destructive_patterns: Vec<String>,
    #[setting(
        label = "sandbox",
        description = "auto uses bubblewrap if installed; tools are disabled unsandboxed"
    )]
    pub sandbox: BashSandboxMode,
    /// Extra bubblewrap arguments, inserted before the trailing `--`: e.g.
    /// `["--share-net"]` giving `bash` the network (it has none by default;
    /// needs Linux 6.12+ Landlock scoping to keep abstract sockets out, or
    /// tools are disabled), or a `--bind` making a dot entry of `$HOME`
    /// writable (read-only).
    #[setting(
        label = "extra bubblewrap arguments",
        description = "Extra bubblewrap arguments, e.g. --share-net for network access",
        advanced
    )]
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
#[derive(DescribeSettings, Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ToolsWriteConfig {
    /// Non-empty path prefixes `write` may create files under; symlinks and dot
    /// entries at any depth below a prefix are refused (list one to allow it).
    /// Writes here ask the user first; `tools.scratch.dir` is always
    /// writable, without asking. `~` and `~/` expand to
    /// `$HOME`; any other relative entry errors, missing ones are dropped.
    #[setting(
        label = "writable folders",
        description = "Folders write may create files under, after asking"
    )]
    pub writable_paths: Vec<String>,
}

impl Default for ToolsWriteConfig {
    fn default() -> Self {
        Self {
            writable_paths: default_writable_paths(),
        }
    }
}
