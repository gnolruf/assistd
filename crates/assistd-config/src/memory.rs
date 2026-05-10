//! Persistent memory configuration.
//!
//! Owns the on-disk SQLite location and the high-level enable flag. The
//! daemon's startup path at `crates/assistd/src/daemon.rs` reads this
//! and either spins up `assistd_memory::SqliteHandle` or binds the
//! `NoMemoryStore` / `NoConversationStore` placeholders.

use serde::{Deserialize, Serialize};

use crate::defaults::{
    DEFAULT_MEMORY_ENABLED, DEFAULT_MEMORY_RETENTION_DAYS, default_memory_db_path,
};

/// `[memory]` section of `config.toml`. New optional fields can be
/// added with `#[serde(default = "…")]` without bumping the
/// `[memory]` block of any user's existing config.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryConfig {
    /// Master switch. When `false` the daemon binds the `NoMemoryStore`
    /// placeholder and skips opening the SQLite file entirely; useful
    /// for ephemeral test daemons or read-only installs.
    #[serde(default = "default_memory_enabled")]
    pub enabled: bool,
    /// Path to the SQLite database file. Default resolves to
    /// `$XDG_DATA_HOME/assistd/memory.db` (or
    /// `$HOME/.local/share/assistd/memory.db`). Stored as `String`
    /// because `Config` round-trips through TOML and `PathBuf` is
    /// quirkier across that boundary; converted to a `Path` at use site.
    #[serde(default = "default_memory_db_path")]
    pub db_path: String,
    /// Days to retain history. `0` means keep forever. The daemon does
    /// not yet ship a sweeper that honours this; present so adding one
    /// later doesn't need a schema-version bump.
    #[serde(default = "default_memory_retention_days")]
    pub retention_days: u32,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            enabled: default_memory_enabled(),
            db_path: default_memory_db_path(),
            retention_days: default_memory_retention_days(),
        }
    }
}

fn default_memory_enabled() -> bool {
    DEFAULT_MEMORY_ENABLED
}

fn default_memory_retention_days() -> u32 {
    DEFAULT_MEMORY_RETENTION_DAYS
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
