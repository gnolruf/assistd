use serde::{Deserialize, Serialize};

use crate::defaults::DEFAULT_HOTKEY_PUSH_TO_TALK;
use crate::settings::DescribeSettings;

/// Global hotkeys, as `global-hotkey` specs like `"Super+Shift+L"`. Only
/// registered on X11; elsewhere bind the matching `assistd` subcommands in
/// the compositor. Empty disables a hotkey, as does turning its feature off.
#[derive(DescribeSettings, Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct HotkeysConfig {
    /// Hold to record a query. Needs `voice.enabled`.
    #[setting(label = "push-to-talk", description = "Hold to record a spoken query")]
    pub push_to_talk: String,
    /// Toggles wake-word listening. Needs `voice.wakeword.enabled`.
    #[setting(
        label = "toggle wake-word listening",
        description = "Turns wake-word listening on or off"
    )]
    pub toggle_listening: String,
    /// Toggles spoken replies; off cancels playback and drops sentences that
    /// arrive meanwhile. Needs `voice.synthesis.enabled`.
    #[setting(
        label = "toggle spoken replies",
        description = "Turns spoken replies on or off; off stops playback"
    )]
    pub toggle_speech: String,
    /// Stops playback and drops the current reply's queued sentences. Needs
    /// `voice.synthesis.enabled`.
    #[setting(
        label = "skip the current reply",
        description = "Stops speaking the current reply"
    )]
    pub skip_speech: String,
}

impl Default for HotkeysConfig {
    fn default() -> Self {
        Self {
            push_to_talk: DEFAULT_HOTKEY_PUSH_TO_TALK.to_string(),
            toggle_listening: String::new(),
            toggle_speech: String::new(),
            skip_speech: String::new(),
        }
    }
}
