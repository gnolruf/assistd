//! Persistent memory configuration.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::defaults::{DEFAULT_MEMORY_ENABLED, default_memory_db_path};

/// `[memory]` section of `config.toml`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct MemoryConfig {
    /// Master switch. When `false` the SQLite file is never opened.
    pub enabled: bool,
    /// Path to the SQLite database file. Default resolves to
    /// `$XDG_DATA_HOME/assistd/memory.db` (or
    /// `$HOME/.local/share/assistd/memory.db`).
    pub db_path: PathBuf,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            enabled: DEFAULT_MEMORY_ENABLED,
            db_path: default_memory_db_path(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_db_path_ends_in_assistd_memory_db() {
        let path = MemoryConfig::default().db_path;
        assert!(
            path.ends_with("assistd/memory.db"),
            "unexpected default db_path: {}",
            path.display()
        );
    }
}
