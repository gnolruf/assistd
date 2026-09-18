use crate::defaults::DEFAULT_PRESENCE_HOTKEY;
use serde::{Deserialize, Serialize};

/// Manual presence-control settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct PresenceConfig {
    /// Global hotkey that cycles `Active → Drowsy → Sleeping → Active`.
    /// Empty string disables the in-daemon global hotkey listener.
    pub hotkey: String,
}

impl Default for PresenceConfig {
    fn default() -> Self {
        Self {
            hotkey: DEFAULT_PRESENCE_HOTKEY.to_string(),
        }
    }
}
