use crate::defaults::DEFAULT_VOICE_HOTKEY;
use serde::{Deserialize, Serialize};

/// Voice input settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VoiceConfig {
    /// Whether voice input is enabled.
    pub enabled: bool,
    /// ALSA/PulseAudio device name. `None` = system default.
    #[serde(default)]
    pub mic_device: Option<String>,
    /// Hotkey to activate voice input (e.g. "Super+V").
    pub hotkey: String,
}

impl Default for VoiceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mic_device: None,
            hotkey: DEFAULT_VOICE_HOTKEY.to_string(),
        }
    }
}
