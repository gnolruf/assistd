//! Errors from loading, editing and validating configuration.

use std::net::IpAddr;
use std::path::{Path, PathBuf};

use assistd_utils::hf::InvalidHfId;
use thiserror::Error;

/// Errors produced while loading, writing, or validating configuration.
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("no config directory: neither XDG_CONFIG_HOME nor a home directory is available")]
    NoConfigDir,

    #[error("failed to read config file {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// Carries the line and toml's message but not the source text, which
    /// may hold the very credential that failed to parse.
    #[error("failed to parse config file {path}{}: {message}", at_line(*.line))]
    Parse {
        path: PathBuf,
        line: Option<usize>,
        message: String,
    },

    #[error("failed to parse config file {path}: {source}")]
    Document {
        path: PathBuf,
        #[source]
        source: toml_edit::TomlError,
    },

    #[error("{} sets `{key}`, which was removed: {replacement}; delete it", path.display())]
    Removed {
        path: PathBuf,
        key: &'static str,
        replacement: &'static str,
    },

    #[error("cannot set `{key}`: {reason}")]
    Edit { key: String, reason: String },

    #[error("failed to serialize default config: {0}")]
    Serialize(#[from] toml::ser::Error),

    #[error("failed to create directory {path}: {source}")]
    CreateDir {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to write config file {path}: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("config file already exists at {0} (not overwriting)")]
    AlreadyExists(PathBuf),

    #[error("{}", format_validation_errors(.0))]
    Validation(Vec<ConfigIssue>),
}

impl ConfigError {
    pub(crate) fn parse(path: &Path, text: &str, error: &toml::de::Error) -> Self {
        let line = error
            .span()
            .and_then(|span| text.get(..span.start))
            .map(|before| before.matches('\n').count() + 1);
        Self::Parse {
            path: path.to_path_buf(),
            line,
            message: error.message().to_owned(),
        }
    }
}

/// A rule between keys, or between a key and this machine, that a parsed
/// config breaks.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ConfigIssue {
    #[error(
        "chat.max_history_tokens + chat.max_response_tokens ({total}) must not exceed 90% of \
         model.context_length ({budget})"
    )]
    HistoryAndResponseOverBudget { total: u64, budget: u32 },

    #[error("chat.max_reasoning_tokens must be strictly less than chat.max_response_tokens")]
    ReasoningNotBelowResponse,

    #[error("chat.summary_target_tokens must be strictly less than chat.max_history_tokens")]
    SummaryNotBelowHistory,

    #[error(
        "sleep.idle_to_sleep_mins must be greater than sleep.idle_to_drowsy_mins \
         (set either to 0 to disable that transition)"
    )]
    SleepNotAfterDrowsy,

    #[error(transparent)]
    NoDefaultDir(#[from] NoDefaultDirError),

    #[error(
        "{key} must be a directory of its own, not {} (assistd removes old files from it)",
        dir.display()
    )]
    SharedToolDir { key: &'static str, dir: PathBuf },

    #[error(
        "tools.output.max_kb ({max_kb} KB, about {head_tokens} tokens) must not exceed a \
         quarter of chat.max_history_tokens ({allowed} tokens)"
    )]
    ToolOutputOverShare {
        max_kb: u32,
        head_tokens: u64,
        allowed: u64,
    },

    #[error(
        "{key} ({timeout_secs}) must be at most {max_secs}: a tool call is cut off after \
         {valve_secs} s, {confirmation_secs} s of which are kept for a confirmation prompt"
    )]
    ToolTimeoutOverValve {
        key: String,
        timeout_secs: u64,
        max_secs: u64,
        valve_secs: u64,
        confirmation_secs: u64,
    },

    #[error("{key} must differ from {other}")]
    PortClash {
        key: &'static str,
        other: &'static str,
    },

    #[error("mcp.servers[{index}].name '{name}' is duplicated; names must be unique")]
    DuplicateMcpServer { index: usize, name: String },
}

/// A tool directory left unset on a machine with nowhere to default it to.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error(
    "{key} has no default: neither XDG_RUNTIME_DIR, XDG_CACHE_HOME nor a home directory is \
     available; set it to an absolute path"
)]
pub struct NoDefaultDirError {
    pub key: &'static str,
}

/// Why a `custom_args` string was refused.
#[derive(Debug, Error)]
pub enum CustomArgsError {
    #[error("unterminated quote or trailing backslash")]
    Unparseable,

    #[error("argument {0:?} contains a control character")]
    ControlCharacter(String),

    #[error("`{0}` is set by assistd or relied on by its client")]
    Managed(String),

    #[error("`{0}` exposes files, tools or state through the server's HTTP API")]
    Exposing(String),
}

/// Why a `search.command` string was refused.
#[derive(Debug, Error)]
pub enum SearchCommandError {
    #[error("unterminated quote or trailing backslash")]
    Unparseable,

    #[error("names no program")]
    Empty,
}

/// A number refused by [`crate::voice::UnitInterval`].
#[derive(Debug, Error)]
#[error("{0} is outside the range 0.0..=1.0")]
pub struct UnitIntervalError(pub f32);

/// A number refused by a bounded decimal such as [`crate::values::Temperature`].
#[derive(Debug, Error)]
#[error("{value} must be {expected}")]
pub struct DecimalRangeError {
    pub value: f32,
    pub expected: &'static str,
}

/// A number refused by [`crate::values::Bounded`].
#[derive(Debug, Error)]
#[error("{value} is outside the range {min}..={max}")]
pub struct IntegerRangeError {
    pub value: u32,
    pub min: u32,
    pub max: u32,
}

/// An address refused by [`crate::values::LoopbackAddr`].
#[derive(Debug, Error)]
#[error("{0} is not a loopback address (127.0.0.0/8 or ::1)")]
pub struct LoopbackAddrError(pub IpAddr);

/// A string refused by [`crate::values::NonBlankString`].
#[derive(Debug, Error)]
#[error("must not be empty or blank")]
pub struct BlankStringError;

/// A list refused by [`crate::values::NonEmpty`].
#[derive(Debug, Error)]
#[error("must have at least one entry")]
pub struct EmptyListError;

/// A path refused by [`crate::values::AbsoluteDir`].
#[derive(Debug, Error)]
#[error("`{0}` must be an absolute path or start with `~/`")]
pub struct AbsoluteDirError(pub String);

/// A string refused by [`crate::values::HfFile`].
#[derive(Debug, Error)]
#[error("`{id}` must be of the form '<owner>/<repo>[@<revision>]:<file>': {reason}")]
pub struct HfFileError {
    pub id: String,
    pub reason: InvalidHfId,
}

/// A string refused by [`crate::voice::SttEncoder`].
#[derive(Debug, Error)]
pub enum SttEncoderError {
    #[error(transparent)]
    Id(#[from] HfFileError),

    #[error("`{0}` must name the model's encoder .onnx, a file whose name starts with `encoder`")]
    NotAnEncoder(String),
}

/// A string refused by [`crate::embedding::EmbeddingModel`].
#[derive(Debug, Error)]
pub enum EmbeddingModelError {
    #[error(transparent)]
    Id(#[from] HfFileError),

    #[error("`{0}` pins a revision, which the embedding model does not support")]
    Pinned(String),
}

/// A string refused by [`crate::voice::WakePhrase`].
#[derive(Debug, Error)]
#[error("{0:?} must be 1 to 6 words of letters and apostrophes")]
pub struct WakePhraseError(pub String);

/// A string refused by [`crate::tools::WritablePath`].
#[derive(Debug, Error)]
#[error("{0:?} must be absolute or start with `~/`")]
pub struct WritablePathError(pub String);

/// A string refused by [`crate::theme::ThemeName`].
#[derive(Debug, Error)]
pub enum ThemeNameError {
    #[error("a theme name is ASCII letters, digits, _ and -")]
    Malformed,

    #[error("{0} is a built-in theme")]
    BuiltIn(String),
}

/// A string refused by [`crate::mcp::McpServerName`].
#[derive(Debug, Error)]
#[error(
    "MCP server name `{0}` must be non-empty and use only ASCII letters, digits, '_' or '-', \
     with no '__' and no trailing '_' (it becomes the `<name>` in tool names `mcp__<name>__<tool>`)"
)]
pub struct McpServerNameError(pub String);

/// A string refused by [`crate::theme::ThemeColor`].
#[derive(Debug, Error)]
#[error(
    "`{0}` is not a color: use `default`, a named ANSI color such as `dark_gray`, or `#rrggbb`"
)]
pub struct ThemeColorError(pub String);

fn at_line(line: Option<usize>) -> String {
    line.map(|line| format!(" at line {line}"))
        .unwrap_or_default()
}

fn format_validation_errors(errors: &[ConfigIssue]) -> String {
    let mut report = format!("configuration has {} error(s):", errors.len());
    for (i, error) in errors.iter().enumerate() {
        report.push_str(&format!("\n  {}: {}", i + 1, error));
    }
    report
}
