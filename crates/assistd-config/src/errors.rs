use std::path::PathBuf;
use thiserror::Error;

/// Errors produced while loading, writing, or validating configuration.
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("HOME environment variable not set")]
    HomeNotSet,

    #[error("failed to read config file {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to parse config file {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },

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

fn format_validation_errors(errors: &[String]) -> String {
    let mut s = format!("configuration has {} error(s):", errors.len());
    for (i, e) in errors.iter().enumerate() {
        s.push_str(&format!("\n  {}: {}", i + 1, e));
    }
    s
}
