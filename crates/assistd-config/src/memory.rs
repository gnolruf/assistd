//! Persistent memory configuration.
//!
//! Owns the on-disk SQLite location and the high-level enable flag. The
//! daemon's startup path at `crates/assistd/src/daemon.rs` reads this
//! and either spins up `assistd_memory::SqliteHandle` or binds the
//! `NoMemoryStore` / `NoConversationStore` placeholders.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::defaults::{DEFAULT_MEMORY_ENABLED, default_memory_db_path};

/// `[memory]` section of `config.toml`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct MemoryConfig {
    /// Master switch. When `false` the daemon binds the `NoMemoryStore`
    /// placeholder and skips opening the SQLite file entirely; useful
    /// for ephemeral test daemons or read-only installs.
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
    fn default_round_trips_through_toml() {
        let cfg = MemoryConfig::default();
        let s = toml::to_string(&cfg).unwrap();
        let back: MemoryConfig = toml::from_str(&s).unwrap();
        assert_eq!(cfg, back);
    }

    #[test]
    fn omitted_section_uses_defaults() {
        // An empty `[memory]` block in user config should produce the
        // same value as the in-code default.
        #[derive(Deserialize)]
        struct Wrap {
            #[serde(default)]
            memory: MemoryConfig,
        }
        let parsed: Wrap = toml::from_str("[memory]").unwrap();
        assert_eq!(parsed.memory, MemoryConfig::default());
    }
}
