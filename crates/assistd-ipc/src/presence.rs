//! Whether the model is loaded: the state the daemon reports and the states a client may drive
//! it to.

use std::fmt;

use serde::{Deserialize, Serialize};

use assistd_utils::text::human_size;

/// Bytes of a model download fetched so far.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct DownloadProgress {
    pub done_bytes: u64,
    /// `0` until the size is known.
    pub total_bytes: u64,
}

impl fmt::Display for DownloadProgress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let done = human_size(self.done_bytes);
        if self.total_bytes == 0 {
            return f.write_str(&done);
        }
        let percent = u128::from(self.done_bytes) * 100 / u128::from(self.total_bytes);
        let total = human_size(self.total_bytes);
        write!(f, "{}% ({done} of {total})", percent.min(100))
    }
}

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
    /// A wake fetching the model's files before it can load them; queries
    /// wait for it.
    Downloading(DownloadProgress),
}

impl PresenceState {
    /// Lower-case name of the state.
    pub const fn as_str(self) -> &'static str {
        match self {
            PresenceState::Active => "active",
            PresenceState::Drowsy => "drowsy",
            PresenceState::Sleeping => "sleeping",
            PresenceState::Waking => "waking",
            PresenceState::Downloading(_) => "downloading",
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
