//! Whether the model is loaded: the state the daemon reports and the states a client may drive
//! it to.

use serde::{Deserialize, Serialize};

/// Coarse daemon lifecycle state.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PresenceState {
    /// Ready to answer queries.
    Active,
    /// llama-server running with its model weights unloaded.
    Drowsy,
    /// llama-server stopped.
    Sleeping,
    /// llama-server starting or loading the model; queries wait for it.
    Waking,
}

impl PresenceState {
    /// Lower-case name, matching the wire form.
    pub const fn as_str(self) -> &'static str {
        match self {
            PresenceState::Active => "active",
            PresenceState::Drowsy => "drowsy",
            PresenceState::Sleeping => "sleeping",
            PresenceState::Waking => "waking",
        }
    }
}

/// A presence state the daemon can be driven to and settle in.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PresenceTarget {
    Active,
    Drowsy,
    Sleeping,
}

impl From<PresenceTarget> for PresenceState {
    fn from(target: PresenceTarget) -> Self {
        match target {
            PresenceTarget::Active => PresenceState::Active,
            PresenceTarget::Drowsy => PresenceState::Drowsy,
            PresenceTarget::Sleeping => PresenceState::Sleeping,
        }
    }
}
