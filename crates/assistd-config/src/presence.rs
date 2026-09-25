use serde::{Deserialize, Serialize};

use crate::defaults::DEFAULT_PRESENCE_HOTKEY;

/// Manual presence-control settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct PresenceConfig {
    /// Global hotkey cycling `Active → Drowsy → Sleeping`. Empty disables
    /// it; the listener works only on X11 (bind `assistd cycle` elsewhere).
    pub hotkey: String,
}

impl Default for PresenceConfig {
    fn default() -> Self {
        Self {
            hotkey: DEFAULT_PRESENCE_HOTKEY.to_string(),
        }
    }
}
