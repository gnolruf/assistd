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
    /// SQLite database file; a leading `~/` expands to `$HOME`. Must be
    /// absolute when enabled.
    #[serde(deserialize_with = "crate::home_path::deserialize")]
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
