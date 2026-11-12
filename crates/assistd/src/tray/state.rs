//! Tray state: raw daemon signals resolved to one icon by priority.

use std::collections::HashSet;

use assistd_config::TrayIconsConfig;
use assistd_ipc::{
    Event, EventBody, ListenPhase, PresenceState, QueueItem, StartupReadiness, TaskQueueSnapshot,
    TaskRunState, VoiceCaptureState,
};
use chrono::{DateTime, Utc};

/// What the tray icon should currently display.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TrayState {
    /// The tray's own config file failed to load or validate.
    ConfigError,
    /// Daemon socket is unreachable.
    Disconnected,
    /// At least one query turn is in flight.
    Generating,
    /// The wake phrase was heard, or a push-to-talk capture is underway.
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
    listen_phase: ListenPhase,
    ptt_capture: VoiceCaptureState,
    startup: StartupReadiness,
    in_flight: HashSet<String>,
    connected: bool,
    /// `None` until the daemon broadcasts the queue.
    task_queue: Option<TaskQueueSnapshot>,
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
            listen_phase: ListenPhase::Off,
            ptt_capture: VoiceCaptureState::Idle,
            startup: StartupReadiness::default(),
            in_flight: HashSet::new(),
            connected: false,
            task_queue: None,
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
        if self.listen_phase == ListenPhase::Capturing
            || self.ptt_capture != VoiceCaptureState::Idle
        {
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

    pub(super) fn task_queue(&self) -> Option<&TaskQueueSnapshot> {
        self.task_queue.as_ref()
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
        self.listen_phase = ListenPhase::Off;
        self.ptt_capture = VoiceCaptureState::Idle;
        self.startup = StartupReadiness::default();
        self.task_queue = None;
        before != self.current()
    }

    /// Returns `true` when the resolved [`TrayState`] changed.
    pub(super) fn ingest(&mut self, event: &Event) -> bool {
        let before = self.current();
        let Event { id, body } = event;
        match body {
            EventBody::Delta { .. }
            | EventBody::ReasoningDelta { .. }
            | EventBody::ToolCall { .. } => {
                self.in_flight.insert(id.clone());
            }
            EventBody::Done | EventBody::Error { .. } => {
                self.in_flight.remove(id);
            }
            EventBody::Presence { state } => {
                self.presence = *state;
            }
            EventBody::ListenState { phase } => {
                self.listen_phase = *phase;
            }
            EventBody::VoiceState { state } => {
                self.ptt_capture = *state;
            }
            EventBody::Readiness { component, state } => {
                self.startup.record(component.clone(), state.clone());
            }
            EventBody::TaskQueue { snapshot } => {
                self.task_queue = Some(snapshot.clone());
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
        TrayState::Disconnected => icons.disconnected.as_str(),
        TrayState::Generating => icons.generating.as_str(),
        TrayState::Listening => icons.listening.as_str(),
        TrayState::Active => icons.active.as_str(),
        TrayState::Sleeping => icons.sleeping.as_str(),
    }
}

/// The tray icon's hover text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Tooltip {
    pub title: String,
    pub description: String,
}

/// The state's one-line title over the config error, else the task queue's lines and how far
/// each subsystem has started.
pub(super) fn tooltip_for(tracker: &TrayTracker, now: DateTime<Utc>) -> Tooltip {
    let description = match tracker.config_error() {
        Some(error) => error.to_string(),
        None => tracker
            .task_queue()
            .map(|snapshot| task_lines(snapshot, now))
            .unwrap_or_default()
            .into_iter()
            .chain(tracker.startup_summary())
            .collect::<Vec<_>>()
            .join("\n"),
    };
    Tooltip {
        title: state_title(tracker.current()).to_string(),
        description,
    }
}

fn state_title(state: TrayState) -> &'static str {
    match state {
        TrayState::ConfigError => "assistd: config error",
        TrayState::Disconnected => "assistd: daemon offline",
        TrayState::Generating => "assistd: thinking…",
        TrayState::Listening => "assistd: listening",
        TrayState::Active => "assistd: idle",
        TrayState::Sleeping => "assistd: sleeping",
    }
}

/// Each running run with how long it has run, then the runs needing input, then how many
/// items wait and whether the queue is paused.
fn task_lines(snapshot: &TaskQueueSnapshot, now: DateTime<Utc>) -> Vec<String> {
    let mut running = Vec::new();
    let mut needing_input = Vec::new();
    let mut waiting = 0usize;
    for item in &snapshot.items {
        match item {
            QueueItem::Run {
                task_name,
                state: TaskRunState::Running,
                started_at,
                ..
            } => running.push(running_line(task_name.as_str(), *started_at, now)),
            QueueItem::Run {
                task_name,
                state: TaskRunState::NeedsInput,
                ..
            } => needing_input.push(task_name.as_str()),
            QueueItem::Run { .. } | QueueItem::UserTurn { .. } => waiting += 1,
        }
    }
    let needs_input = match needing_input.as_slice() {
        [] => None,
        [name] => Some(format!("1 task needs input: {name}")),
        names => Some(format!(
            "{} tasks need input: {}",
            names.len(),
            names.join(", ")
        )),
    };
    let queued = (waiting > 0 || snapshot.paused).then(|| {
        let paused = if snapshot.paused { " · paused" } else { "" };
        format!("{waiting} queued{paused}")
    });
    running
        .into_iter()
        .chain(needs_input)
        .chain(queued)
        .collect()
}

fn running_line(name: &str, started_at: Option<DateTime<Utc>>, now: DateTime<Utc>) -> String {
    let elapsed = started_at
        .and_then(|at| u64::try_from((now - at).num_minutes()).ok())
        .map_or_else(String::new, |minutes| {
            format!(" · running {}", elapsed_minutes(minutes))
        });
    format!("task: {name}{elapsed}")
}

/// `<1 min`, `4 min`, `1 hr 5 min`.
fn elapsed_minutes(minutes: u64) -> String {
    match (minutes / 60, minutes % 60) {
        (0, 0) => "<1 min".to_string(),
        (0, mins) => format!("{mins} min"),
        (hours, mins) => format!("{hours} hr {mins} min"),
    }
}

#[cfg(test)]
mod tests;
