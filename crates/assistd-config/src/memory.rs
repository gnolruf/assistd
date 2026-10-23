//! Persistent memory configuration.

use serde::{Deserialize, Serialize};

use crate::defaults::DEFAULT_MEMORY_ENABLED;
use crate::settings::{DescribeSettings, describe_fields};

/// SQLite-backed memory settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct MemoryConfig {
    /// When `false`, the model gets no `remember` or `reminisce` tool and no
    /// recalled past context; chats are saved either way.
    pub enabled: bool,
}

impl DescribeSettings for MemoryConfig {
    describe_fields! {
        enabled => (
            "recall memories and past chats",
            "Gives the model remember and reminisce tools"
        ),
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            enabled: DEFAULT_MEMORY_ENABLED,
        }
    }
}
