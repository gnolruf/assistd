//! Tray notification configuration.

use serde::{Deserialize, Serialize};

use crate::defaults::bounded;
use crate::settings::{DescribeSettings, describe_fields};
use crate::values::Bounded;

/// Desktop notifications from `assistd tray` showing a turn's activity, tool
/// calls and reply. No notification is sent while the chat has keyboard focus.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct NotificationsConfig {
    pub enabled: bool,
    /// Idle ms before the notification closes, `500..=60000`. Reset by
    /// each event and paused while a turn is in flight.
    pub auto_hide_ms: Bounded<500, 60_000>,
    /// Ask for a short plain-text reply to a voice turn while the chat is
    /// unfocused or closed.
    pub brief_when_away: bool,
}

impl DescribeSettings for NotificationsConfig {
    describe_fields! {
        #![consumer(Tray)]
        enabled => ("enabled", "Desktop notifications of each turn from assistd tray"),
        auto_hide_ms => (
            "hide after idle (ms)",
            "Idle time before a notification closes, 500 to 60000"
        ),
        #[consumer(Daemon)]
        brief_when_away => (
            "brief voice replies when away",
            "Ask for short replies to voice turns while the chat is unfocused"
        ),
    }
}

impl NotificationsConfig {
    /// Idle timeout while continuous listening is active: `3 × auto_hide_ms`,
    /// leaving time to hear the reply and answer verbally.
    pub fn listen_auto_hide_ms(&self) -> u64 {
        u64::from(self.auto_hide_ms.get()) * 3
    }
}

impl Default for NotificationsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            auto_hide_ms: bounded(3000),
            brief_when_away: true,
        }
    }
}
