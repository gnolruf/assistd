use crate::defaults::DEFAULT_PRESENCE_HOTKEY;
use serde::{Deserialize, Serialize};

/// Manual presence-control settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PresenceConfig {
    /// Global hotkey that cycles `Active → Drowsy → Sleeping → Active`.
    /// Empty string disables the in-daemon global hotkey listener.
    #[serde(default = "default_presence_hotkey")]
    pub hotkey: String,
}

impl Default for PresenceConfig {
    fn default() -> Self {
        Self {
            hotkey: default_presence_hotkey(),
        }
    }
}

fn default_presence_hotkey() -> String {
    DEFAULT_PRESENCE_HOTKEY.to_string()
}
