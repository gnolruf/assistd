//! Persistent memory configuration.

use serde::{Deserialize, Serialize};

use crate::defaults::DEFAULT_MEMORY_ENABLED;

/// SQLite-backed memory settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct MemoryConfig {
    /// When `false`, the model gets no `remember` or `reminisce` tool and no
    /// recalled past context; chats are saved either way.
    pub enabled: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            enabled: DEFAULT_MEMORY_ENABLED,
        }
    }
}
