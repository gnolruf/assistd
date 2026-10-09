//! Tray state: raw daemon signals resolved to one icon by priority.

use std::collections::HashSet;

use assistd_config::TrayIconsConfig;
use assistd_ipc::{Event, PresenceState, StartupReadiness, VoiceCaptureState};

/// What the tray icon should currently display.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TrayState {
    /// The tray's own config file failed to load or validate.
    ConfigError,
    /// Daemon socket is unreachable.
    Disconnected,
    /// At least one query turn is in flight.
    Generating,
    /// Continuous listening is on, or a push-to-talk capture is underway.
    Listening,
    /// Daemon is awake and idle.
    Active,
    /// Daemon is drowsing, sleeping, or loading the model (rendered the
    /// same at-a-glance).
    Sleeping,
}

#[derive(Debug, Clone)]
pub(super) struct TrayTracker {
    config_error: Option<String>,
    presence: PresenceState,
    listening: bool,
    ptt_capture: VoiceCaptureState,
    startup: StartupReadiness,
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
    pub(super) fn new(config_error: Option<String>) -> Self {
        Self {
            config_error,
            presence: PresenceState::Active,
            listening: false,
            ptt_capture: VoiceCaptureState::Idle,
            startup: StartupReadiness::default(),
            in_flight: HashSet::new(),
            connected: false,
        }
    }

    /// Priority: config error, disconnected, generating, listening, presence.
    pub(super) fn current(&self) -> TrayState {
        if self.config_error.is_some() {
            return TrayState::ConfigError;
        }
        if !self.connected {
            return TrayState::Disconnected;
        }
        if !self.in_flight.is_empty() {
            return TrayState::Generating;
        }
        if self.listening || self.ptt_capture != VoiceCaptureState::Idle {
            return TrayState::Listening;
        }
        match self.presence {
            PresenceState::Active => TrayState::Active,
            PresenceState::Drowsy | PresenceState::Sleeping | PresenceState::Waking => {
                TrayState::Sleeping
            }
        }
    }

    pub(super) fn presence(&self) -> PresenceState {
        self.presence
    }

    pub(super) fn connected(&self) -> bool {
        self.connected
    }

    pub(super) fn config_error(&self) -> Option<&str> {
        self.config_error.as_deref()
    }

    /// One line per background subsystem with how far it has started;
    /// `None` until the daemon reports any.
    pub(super) fn startup_summary(&self) -> Option<String> {
        let lines: Vec<String> = self
            .startup
            .iter()
            .map(|(component, state)| format!("{component}: {state}"))
            .collect();
        (!lines.is_empty()).then(|| lines.join("\n"))
    }

    /// Returns `true` when the resolved [`TrayState`] changed.
    pub(super) fn set_connected(&mut self) -> bool {
        let before = self.current();
        self.connected = true;
        before != self.current()
    }

    /// Also drops per-turn state. Returns `true` when the resolved
    /// [`TrayState`] changed.
    pub(super) fn set_disconnected(&mut self) -> bool {
        let before = self.current();
        self.connected = false;
        self.in_flight.clear();
        self.listening = false;
        self.ptt_capture = VoiceCaptureState::Idle;
        self.startup = StartupReadiness::default();
        before != self.current()
    }

    /// Returns `true` when the resolved [`TrayState`] changed.
    pub(super) fn ingest(&mut self, event: &Event) -> bool {
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
            Event::VoiceState { state, .. } => {
                self.ptt_capture = *state;
            }
            Event::Readiness {
                component, state, ..
            } => {
                self.startup.record(component.clone(), state.clone());
            }
            _ => {}
        }
        before != self.current()
    }
}

/// The configured icon for `state`; a config error always shows `dialog-error`.
pub(super) fn icon_name_for(state: TrayState, icons: &TrayIconsConfig) -> &str {
    match state {
        TrayState::ConfigError => "dialog-error",
        TrayState::Disconnected => &icons.disconnected,
        TrayState::Generating => &icons.generating,
        TrayState::Listening => &icons.listening,
        TrayState::Active => &icons.active,
        TrayState::Sleeping => &icons.sleeping,
    }
}

pub(super) fn tooltip_for(state: TrayState) -> &'static str {
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
mod tests;
