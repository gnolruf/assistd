//! What the microphone is doing: push-to-talk capture and the wake-word listener's phase.

use serde::{Deserialize, Serialize};

/// Push-to-talk capture state. `Idle` must stay the first variant.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VoiceCaptureState {
    Idle,
    Recording,
    Transcribing,
}

/// Wake-word listener phase.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ListenPhase {
    /// The listener is stopped.
    Off,
    /// The mic is open and waiting for the wake phrase.
    Armed,
    /// The wake phrase was heard; the next utterance is the command.
    Capturing,
}

impl ListenPhase {
    /// Whether the listener has the mic open.
    pub const fn is_on(self) -> bool {
        !matches!(self, ListenPhase::Off)
    }

    /// Lower-case name, matching the wire form.
    pub const fn as_str(self) -> &'static str {
        match self {
            ListenPhase::Off => "off",
            ListenPhase::Armed => "armed",
            ListenPhase::Capturing => "capturing",
        }
    }
}
