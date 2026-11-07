//! Errors from loading, editing and validating configuration.

use std::path::{Path, PathBuf};

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
    Validation(Vec<String>),
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

fn format_validation_errors(errors: &[String]) -> String {
    let mut report = format!("configuration has {} error(s):", errors.len());
    for (i, error) in errors.iter().enumerate() {
        report.push_str(&format!("\n  {}: {}", i + 1, error));
    }
    report
}
