//! Tool subsystem configuration: scratch directory, output limits and per-command policy.

use std::num::{NonZeroU32, NonZeroU64};
use std::path::PathBuf;
use std::str::FromStr;

use assistd_utils::path::tilde_remainder;
use serde::{Deserialize, Serialize};

use crate::defaults::{
    DEFAULT_BASH_TIMEOUT_SECS, DEFAULT_TOOLS_MAX_KB, DEFAULT_TOOLS_MAX_LINES,
    DEFAULT_TOOLS_SCRATCH_RETENTION_DAYS, DEFAULT_WRITABLE_PATHS, default_tools_dir,
};
use crate::errors::{NoDefaultDirError, WritablePathError};
use crate::settings::{DescribeSettings, choice_setting, describe_fields};
use crate::values::{AbsoluteDir, NonEmpty};

const SCRATCH_DIR_KEY: &str = "tools.scratch.dir";
const OVERFLOW_DIR_KEY: &str = "tools.output.overflow_dir";

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
    /// asking. Created owner-only on daemon startup. Absolute or starting
    /// with `~/`; must not be `/` or hold `$HOME` or assistd's config or data
    /// directory. `None` uses `$XDG_RUNTIME_DIR/assistd/scratch`, else
    /// `assistd/scratch` under the user cache dir.
    pub dir: Option<AbsoluteDir>,
    /// Days an entry in `dir` is kept after it last changed; older ones,
    /// and directories that leaves empty, are removed on daemon startup when
    /// assistd created `dir`. 0 empties the directory at every startup.
    pub retention_days: u32,
}

impl DescribeSettings for ToolsScratchConfig {
    describe_fields! {
        dir => (
            "scratch folder",
            "Folder tools may write to freely; unset uses the runtime or cache dir",
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
            dir: None,
            retention_days: DEFAULT_TOOLS_SCRATCH_RETENTION_DAYS,
        }
    }
}

impl ToolsScratchConfig {
    /// `dir`, or its default. Errors when it is unset and this machine has
    /// neither a runtime nor a cache directory to default it under.
    pub fn resolved_dir(&self) -> Result<PathBuf, NoDefaultDirError> {
        resolve_tools_dir(self.dir.as_ref(), "scratch", SCRATCH_DIR_KEY)
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
    /// or 128 MiB. Absolute or starting with `~/`; must not be `/` or hold
    /// `$HOME` or assistd's config or data directory. `None` uses
    /// `$XDG_RUNTIME_DIR/assistd/output`, else `assistd/output` under the
    /// user cache dir.
    pub overflow_dir: Option<AbsoluteDir>,
}

impl DescribeSettings for ToolsOutputConfig {
    describe_fields! {
        max_lines => ("max result lines", "Most lines of one tool result", advanced),
        max_kb => (
            "max result size (KB)",
            "Largest tool result; at most a quarter of the history budget",
            advanced
        ),
        overflow_dir => (
            "overflow folder",
            "Where cut-off tool output is saved in full; unset uses the runtime or cache dir",
            advanced
        ),
    }
}

impl Default for ToolsOutputConfig {
    fn default() -> Self {
        Self {
            max_lines: DEFAULT_TOOLS_MAX_LINES,
            max_kb: DEFAULT_TOOLS_MAX_KB,
            overflow_dir: None,
        }
    }
}

impl ToolsOutputConfig {
    /// `max_kb` expressed in bytes.
    pub fn max_bytes(&self) -> usize {
        (self.max_kb.get() as usize) * 1024
    }

    /// `overflow_dir`, or its default. Errors when it is unset and this
    /// machine has neither a runtime nor a cache directory to default it under.
    pub fn resolved_overflow_dir(&self) -> Result<PathBuf, NoDefaultDirError> {
        resolve_tools_dir(self.overflow_dir.as_ref(), "output", OVERFLOW_DIR_KEY)
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

choice_setting!(BashSandboxMode {
    Auto => "auto",
    Bwrap => "bwrap",
    None => "none",
});

/// Policy for the scripts `bash` runs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ToolsBashConfig {
    /// `bash` timeout in seconds, `1..=180`; on expiry the process group is
    /// killed (exit 137). The cap keeps a run that waited out a confirmation
    /// prompt under the 300 s tool-call deadline.
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
        timeout_secs => ("command timeout (s)", "Longest a bash command runs before it is killed, at most 180"),
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
    /// Path prefixes `write` may create files under; anywhere else outside
    /// `tools.scratch.dir` is refused. Symlinks and dot entries at any depth
    /// below a prefix are refused (list one to allow it). Writes here ask
    /// first; scratch is writable without asking. Missing ones are dropped.
    pub writable_paths: NonEmpty<WritablePath>,
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
            writable_paths: DEFAULT_WRITABLE_PATHS
                .map(|path| WritablePath(path.to_owned()))
                .to_vec()
                .try_into()
                .expect("the default writable paths are not empty"),
        }
    }
}

/// A path prefix as written: absolute, or `~` or `~/…` for under `$HOME`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct WritablePath(String);

impl WritablePath {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl FromStr for WritablePath {
    type Err = WritablePathError;

    fn from_str(path: &str) -> Result<Self, Self::Err> {
        Self::try_from(path.to_owned())
    }
}

impl TryFrom<String> for WritablePath {
    type Error = WritablePathError;

    fn try_from(path: String) -> Result<Self, Self::Error> {
        if tilde_remainder(&path).is_some() || PathBuf::from(&path).is_absolute() {
            Ok(Self(path))
        } else {
            Err(WritablePathError(path))
        }
    }
}

impl From<WritablePath> for String {
    fn from(path: WritablePath) -> Self {
        path.0
    }
}

fn resolve_tools_dir(
    configured: Option<&AbsoluteDir>,
    leaf: &str,
    key: &'static str,
) -> Result<PathBuf, NoDefaultDirError> {
    configured
        .map(|dir| dir.as_path().to_path_buf())
        .or_else(|| default_tools_dir(leaf))
        .ok_or(NoDefaultDirError { key })
}
