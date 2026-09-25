//! Persistent memory configuration.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::defaults::{DEFAULT_MEMORY_ENABLED, default_memory_db_path};

/// SQLite-backed memory settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct MemoryConfig {
    /// When `false`, the database is never opened.
    pub enabled: bool,
    /// SQLite database file. Must not be empty when enabled.
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
