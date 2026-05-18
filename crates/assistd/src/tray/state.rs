//! Tray state machine. Translates raw daemon events into the small
//! enum the icon renderer cares about, applying a fixed priority order
//! so concurrent signals (e.g. listening while generating) produce a
//! single deterministic icon.

use std::collections::HashSet;

use assistd_config::TrayConfig;
use assistd_ipc::{Event, PresenceState};

/// What the tray icon should currently display.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrayState {
    /// Daemon socket is unreachable.
    Disconnected,
    /// At least one query turn is in flight.
    Generating,
    /// Continuous listening is on.
    Listening,
    /// Daemon is awake and idle.
    Active,
    /// Daemon is drowsing or sleeping (rendered the same at-a-glance).
    Sleeping,
}

/// Mutable bag of raw signals received over IPC, plus the priority
/// resolution that derives a single [`TrayState`] from them.
#[derive(Debug, Clone)]
pub struct TrayTracker {
    presence: PresenceState,
    listening: bool,
    in_flight: HashSet<String>,
    connected: bool,
}

impl Default for TrayTracker {
    fn default() -> Self {
        Self {
            presence: PresenceState::Active,
            listening: false,
            in_flight: HashSet::new(),
            connected: false,
        }
    }
}

impl TrayTracker {
    /// Resolve the raw signals into a single tray state, applying the
    /// priority order disconnected → generating → listening → presence.
    pub fn current(&self) -> TrayState {
        if !self.connected {
            return TrayState::Disconnected;
        }
        if !self.in_flight.is_empty() {
            return TrayState::Generating;
        }
        if self.listening {
            return TrayState::Listening;
        }
        match self.presence {
            PresenceState::Active => TrayState::Active,
            PresenceState::Drowsy | PresenceState::Sleeping => TrayState::Sleeping,
        }
    }

    /// The daemon's most recently observed presence (independent of
    /// connection state). Used by the menu to label the "Sleep / Wake"
    /// toggle item.
    pub fn presence(&self) -> PresenceState {
        self.presence
    }

    /// Mark the connection up. Returns `true` when the resolved
    /// [`TrayState`] actually changed.
    pub fn set_connected(&mut self) -> bool {
        let before = self.current();
        self.connected = true;
        before != self.current()
    }

    /// Mark the connection down and drop all per-turn state we can no
    /// longer trust. Returns `true` when the resolved [`TrayState`]
    /// actually changed.
    pub fn set_disconnected(&mut self) -> bool {
        let before = self.current();
        self.connected = false;
        self.in_flight.clear();
        self.listening = false;
        before != self.current()
    }

    /// Apply an incoming wire event, updating the relevant raw signal.
    /// Returns `true` when the resolved [`TrayState`] actually changed,
    /// so the caller can skip redundant DBus property-changed broadcasts.
    pub fn ingest(&mut self, event: &Event) -> bool {
        let before = self.current();
        match event {
            Event::Delta { id, .. } | Event::ToolCall { id, .. } => {
                self.in_flight.insert(id.clone());
            }
            Event::Done { id, .. } | Event::Error { id, .. } => {
                self.in_flight.remove(id);
            }
            Event::Presence { state, .. } => {
                self.presence = *state;
            }
            Event::ListenState { active, .. } => {
                self.listening = *active;
            }
            _ => {}
        }
        before != self.current()
    }
}

/// Map a resolved state to the freedesktop icon-theme name configured
/// for it.
pub fn icon_name_for(state: TrayState, cfg: &TrayConfig) -> &str {
    match state {
        TrayState::Disconnected => &cfg.icon_disconnected,
        TrayState::Generating => &cfg.icon_generating,
        TrayState::Listening => &cfg.icon_listening,
        TrayState::Active => &cfg.icon_active,
        TrayState::Sleeping => &cfg.icon_sleeping,
    }
}

/// Short human-readable label for the tooltip.
pub fn tooltip_for(state: TrayState) -> &'static str {
    match state {
        TrayState::Disconnected => "assistd: daemon offline",
        TrayState::Generating => "assistd: thinking…",
        TrayState::Listening => "assistd: listening",
        TrayState::Active => "assistd: idle",
        TrayState::Sleeping => "assistd: sleeping",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn delta(id: &str) -> Event {
        Event::Delta {
            id: id.into(),
            text: String::new(),
        }
    }

    fn tool_call(id: &str) -> Event {
        Event::ToolCall {
            id: id.into(),
            name: "x".into(),
            args: serde_json::Value::Null,
        }
    }

    fn done(id: &str) -> Event {
        Event::Done { id: id.into() }
    }

    fn error(id: &str) -> Event {
        Event::Error {
            id: id.into(),
            message: "boom".into(),
        }
    }

    fn presence(state: PresenceState) -> Event {
        Event::Presence {
            id: "x".into(),
            state,
        }
    }

    fn listen(active: bool) -> Event {
        Event::ListenState {
            id: "x".into(),
            active,
        }
    }

    #[test]
    fn disconnected_takes_top_priority() {
        let t = TrayTracker::default();
        assert_eq!(t.current(), TrayState::Disconnected);
    }

    #[test]
    fn generating_outranks_listening_and_presence() {
        let mut t = TrayTracker::default();
        t.set_connected();
        t.ingest(&listen(true));
        t.ingest(&delta("a"));
        assert_eq!(t.current(), TrayState::Generating);
    }

    #[test]
    fn listening_outranks_presence() {
        let mut t = TrayTracker::default();
        t.set_connected();
        t.ingest(&presence(PresenceState::Active));
        t.ingest(&listen(true));
        assert_eq!(t.current(), TrayState::Listening);
    }

    #[test]
    fn drowsy_and_sleeping_both_render_as_sleeping() {
        let mut t = TrayTracker::default();
        t.set_connected();
        t.ingest(&presence(PresenceState::Drowsy));
        assert_eq!(t.current(), TrayState::Sleeping);
        t.ingest(&presence(PresenceState::Sleeping));
        assert_eq!(t.current(), TrayState::Sleeping);
        t.ingest(&presence(PresenceState::Active));
        assert_eq!(t.current(), TrayState::Active);
    }

    #[test]
    fn concurrent_turns_keep_generating_until_all_resolve() {
        let mut t = TrayTracker::default();
        t.set_connected();
        t.ingest(&delta("a"));
        t.ingest(&tool_call("b"));
        assert_eq!(t.current(), TrayState::Generating);
        t.ingest(&done("a"));
        assert_eq!(t.current(), TrayState::Generating);
        t.ingest(&error("b"));
        assert_eq!(t.current(), TrayState::Active);
    }

    #[test]
    fn done_for_unknown_id_is_harmless() {
        let mut t = TrayTracker::default();
        t.set_connected();
        let changed = t.ingest(&done("never-seen"));
        assert!(!changed);
        assert_eq!(t.current(), TrayState::Active);
    }

    #[test]
    fn disconnect_clears_in_flight_and_listening() {
        let mut t = TrayTracker::default();
        t.set_connected();
        t.ingest(&listen(true));
        t.ingest(&delta("a"));
        assert!(t.set_disconnected());
        assert_eq!(t.current(), TrayState::Disconnected);

        let changed = t.set_connected();
        assert!(changed);
        // After reconnect, in-flight and listening are gone; presence
        // defaults to Active until the daemon broadcasts otherwise.
        assert_eq!(t.current(), TrayState::Active);
    }

    #[test]
    fn ingest_returns_change_flag() {
        let mut t = TrayTracker::default();
        t.set_connected();
        // First Delta: enters Generating → state changed.
        assert!(t.ingest(&delta("a")));
        // Second Delta for the same id: still Generating → no change.
        assert!(!t.ingest(&delta("a")));
        // Done for the only in-flight id: back to Active → state changed.
        assert!(t.ingest(&done("a")));
    }

    #[test]
    fn unrelated_events_do_not_change_state() {
        let mut t = TrayTracker::default();
        t.set_connected();
        let changed = t.ingest(&Event::ToolResult {
            id: "a".into(),
            name: "x".into(),
            result: serde_json::Value::Null,
        });
        assert!(!changed);
        let changed = t.ingest(&Event::LastDelta {
            id: "a".into(),
            text: "x".into(),
        });
        assert!(!changed);
    }

    #[test]
    fn icon_name_for_threads_through_config() {
        let cfg = TrayConfig::default();
        assert_eq!(icon_name_for(TrayState::Active, &cfg), &cfg.icon_active);
        assert_eq!(
            icon_name_for(TrayState::Disconnected, &cfg),
            &cfg.icon_disconnected
        );
        assert_eq!(
            icon_name_for(TrayState::Generating, &cfg),
            &cfg.icon_generating
        );
    }
}
