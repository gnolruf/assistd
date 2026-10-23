//! Persistent memory configuration.

use serde::{Deserialize, Serialize};

use crate::defaults::DEFAULT_MEMORY_ENABLED;
use crate::settings::DescribeSettings;

/// SQLite-backed memory settings.
#[derive(DescribeSettings, Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct MemoryConfig {
    /// When `false`, the model gets no `remember` or `reminisce` tool and no
    /// recalled past context; chats are saved either way.
    #[setting(
        label = "remember memories and past chats",
        description = "Gives the model remember and reminisce tools"
    )]
    pub enabled: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            enabled: DEFAULT_MEMORY_ENABLED,
        }
    }
}
