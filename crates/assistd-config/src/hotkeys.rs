use serde::{Deserialize, Serialize};

use crate::defaults::DEFAULT_HOTKEY_PUSH_TO_TALK;
use crate::settings::{DescribeSettings, describe_fields};

/// Global hotkeys, as `global-hotkey` specs like `"Super+Shift+L"`. Only
/// registered on X11; elsewhere bind the matching `assistd` subcommands in
/// the compositor. Empty disables a hotkey, as does turning its feature off.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct HotkeysConfig {
    /// Hold to record a query. Needs `voice.enabled`.
    pub push_to_talk: String,
    /// Toggles wake-word listening. Needs `voice.wakeword.enabled`.
    pub toggle_listening: String,
    /// Toggles spoken replies; off cancels playback and drops sentences that
    /// arrive meanwhile. Needs `voice.synthesis.enabled`.
    pub toggle_speech: String,
    /// Stops playback and drops the current reply's queued sentences. Needs
    /// `voice.synthesis.enabled`.
    pub skip_speech: String,
}

impl DescribeSettings for HotkeysConfig {
    describe_fields! {
        push_to_talk => ("push-to-talk", "Hold to record a spoken query"),
        toggle_listening => ("toggle wake-word listening", "Turns wake-word listening on or off"),
        toggle_speech => (
            "toggle spoken replies",
            "Turns spoken replies on or off; off stops playback"
        ),
        skip_speech => ("skip the current reply", "Stops speaking the current reply"),
    }
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
