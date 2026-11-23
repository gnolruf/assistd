//! Persistent memory configuration.

use serde::{Deserialize, Serialize};

use crate::settings::{DescribeSettings, describe_fields};

/// SQLite-backed memory settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct MemoryConfig {
    /// When `false`, the model gets no `remember` tool, `search` leaves out sessions
    /// and memories, and no past context is recalled; chats are saved either way.
    pub enabled: bool,
}

impl DescribeSettings for MemoryConfig {
    describe_fields! {
        enabled => (
            "recall memories and past chats",
            "Lets the model save memories and search them and past chats"
        ),
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}
