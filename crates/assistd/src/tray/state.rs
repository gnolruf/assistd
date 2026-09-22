//! Tray state: raw daemon signals resolved to one icon by priority.

use std::collections::HashSet;

use assistd_ipc::{Event, PresenceState};

/// What the tray icon should currently display.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrayState {
    /// The tray's own config file failed to load or validate.
    ConfigError,
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

#[derive(Debug, Clone)]
pub struct TrayTracker {
    config_error: Option<String>,
    presence: PresenceState,
    listening: bool,
    in_flight: HashSet<String>,
    connected: bool,
}

impl Default for TrayTracker {
    fn default() -> Self {
        Self::new(None)
    }
}

impl TrayTracker {
    /// A tracker that reports [`TrayState::ConfigError`] for as long as
    /// `config_error` is `Some`, regardless of daemon state.
    pub fn new(config_error: Option<String>) -> Self {
        Self {
            config_error,
            presence: PresenceState::Active,
            listening: false,
            in_flight: HashSet::new(),
            connected: false,
        }
    }

    /// Priority: config error, disconnected, generating, listening, presence.
    pub fn current(&self) -> TrayState {
        if self.config_error.is_some() {
            return TrayState::ConfigError;
        }
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

    pub fn presence(&self) -> PresenceState {
        self.presence
    }

    pub fn connected(&self) -> bool {
        self.connected
    }

    pub fn config_error(&self) -> Option<&str> {
        self.config_error.as_deref()
    }

    /// Returns `true` when the resolved [`TrayState`] changed.
    pub fn set_connected(&mut self) -> bool {
        let before = self.current();
        self.connected = true;
        before != self.current()
    }

    /// Also drops per-turn state. Returns `true` when the resolved
    /// [`TrayState`] changed.
    pub fn set_disconnected(&mut self) -> bool {
        let before = self.current();
        self.connected = false;
        self.in_flight.clear();
        self.listening = false;
        before != self.current()
    }

    /// Returns `true` when the resolved [`TrayState`] changed.
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

/// freedesktop icon-theme names present in every major theme, so no
/// image assets ship.
pub fn icon_name_for(state: TrayState) -> &'static str {
    match state {
        TrayState::ConfigError => "dialog-error",
        TrayState::Disconnected => "network-offline",
        TrayState::Generating => "system-run",
        TrayState::Listening => "audio-input-microphone",
        TrayState::Active => "user-available",
        TrayState::Sleeping => "user-offline",
    }
}

pub fn tooltip_for(state: TrayState) -> &'static str {
    match state {
        TrayState::ConfigError => "assistd: config error",
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
    fn config_error_outranks_everything() {
        let mut t = TrayTracker::new(Some("unknown field `foo`".into()));
        assert_eq!(t.current(), TrayState::ConfigError);
        assert_eq!(t.config_error(), Some("unknown field `foo`"));

        assert!(!t.set_connected());
        assert!(t.connected());
        assert!(!t.ingest(&listen(true)));
        assert!(!t.ingest(&delta("a")));
        assert_eq!(t.current(), TrayState::ConfigError);

        assert!(!t.set_disconnected());
        assert_eq!(t.current(), TrayState::ConfigError);
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
        assert_eq!(t.current(), TrayState::Active);
    }

    #[test]
    fn ingest_returns_change_flag() {
        let mut t = TrayTracker::default();
        t.set_connected();
        assert!(t.ingest(&delta("a")));
        assert!(!t.ingest(&delta("a")));
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
    fn every_state_maps_to_a_distinct_icon() {
        let names: Vec<_> = [
            TrayState::ConfigError,
            TrayState::Disconnected,
            TrayState::Generating,
            TrayState::Listening,
            TrayState::Active,
            TrayState::Sleeping,
        ]
        .into_iter()
        .map(icon_name_for)
        .collect();
        let unique: std::collections::HashSet<_> = names.iter().collect();
        assert_eq!(unique.len(), names.len(), "{names:?}");
    }
}
