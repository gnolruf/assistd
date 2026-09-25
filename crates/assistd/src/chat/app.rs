//! Chat application state and reducer. The `on_*` methods mutate state
//! only; I/O lives in the `spawn_*` methods and the attach loader.

use std::sync::Arc;
use std::time::{Duration, Instant};

use assistd_core::{PresenceState, SleepConfig};
use assistd_ipc::{Event, IpcClient, Request, VoiceCaptureState};
use assistd_tools::{Attachment, ConfirmationRequest};
use ratatui_image::picker::Picker;
use ratatui_image::protocol::StatefulProtocol;
use tokio::sync::mpsc;
use tokio::task::JoinSet;

use super::input::InputLine;
use super::output::OutputPane;
use super::throughput::ThroughputMeter;
use super::vram::ResourceState;

mod attach;
mod commands;
mod connection;
mod keys;
mod modal;
mod wire;

const SPINNER_CHARS: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
const NOTICE_HOLD: Duration = Duration::from_secs(3);
/// Approving keys are ignored for this long after the confirmation modal
/// opens, so a keystroke aimed at the input line cannot approve unseen.
const CONFIRM_ARM_DELAY: Duration = Duration::from_millis(750);

/// An image staged by `/attach` for the next submission.
pub struct PendingAttachment {
    /// File basename.
    pub name: String,
    pub mime: String,
    pub bytes: Vec<u8>,
    /// `None` on terminals without graphics support.
    pub protocol: Option<StatefulProtocol>,
}

impl PendingAttachment {
    fn into_parts(self) -> (Attachment, Option<StatefulProtocol>, String) {
        (
            Attachment::Image {
                mime: self.mime,
                bytes: self.bytes,
            },
            self.protocol,
            self.name,
        )
    }
}

/// Boxed inside [`ChatEvent`] so the graphics buffers do not bloat the
/// other variants.
pub struct AttachLoadedPayload {
    pub name: String,
    pub mime: String,
    pub size: usize,
    pub bytes: Vec<u8>,
    pub protocol: Option<StatefulProtocol>,
}

/// Which concurrent daemon connection an event arrived on, and so which
/// slice of `App` state its terminal event may retire.
#[derive(Debug, Clone, Copy)]
pub enum WireStream {
    /// A query dialog or push-to-talk turn; owns the assistant message,
    /// [`App::generating`] and the query writer.
    Reply,
    /// A branch command; owns the in-flight branch op and its rows.
    Branch,
    /// Polls and the F2 cycle; indicator state only.
    Status,
}

pub enum ChatEvent {
    Wire {
        stream: WireStream,
        event: Event,
    },
    /// The connection ended without a terminal event.
    WireError {
        stream: WireStream,
        message: String,
    },
    AttachLoaded(Box<AttachLoadedPayload>),
    AttachFailed {
        path: String,
        message: String,
    },
}

impl std::fmt::Debug for ChatEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatEvent::Wire { stream, event } => f
                .debug_struct("Wire")
                .field("stream", stream)
                .field("event", event)
                .finish(),
            ChatEvent::WireError { stream, message } => f
                .debug_struct("WireError")
                .field("stream", stream)
                .field("message", message)
                .finish(),
            ChatEvent::AttachLoaded(p) => f
                .debug_struct("AttachLoaded")
                .field("name", &p.name)
                .field("mime", &p.mime)
                .field("size", &p.size)
                .field("has_thumbnail", &p.protocol.is_some())
                .finish(),
            ChatEvent::AttachFailed { path, message } => f
                .debug_struct("AttachFailed")
                .field("path", path)
                .field("message", message)
                .finish(),
        }
    }
}

/// Command-confirmation prompt shown while the daemon's agent loop blocks
/// on the answer.
pub struct ConfirmationModal {
    pub request: ConfirmationRequest,
    /// Echoed back in the `Request::ConfirmResponse`.
    confirm_id: String,
    opened_at: Instant,
}

impl ConfirmationModal {
    /// Whether the modal has been visible long enough to accept approval.
    /// Denial is accepted at any time.
    pub fn armed(&self) -> bool {
        self.opened_at.elapsed() >= CONFIRM_ARM_DELAY
    }
}

pub struct App {
    pub output: OutputPane,
    pub input: InputLine,
    pub throughput: ThroughputMeter,
    pub resources: ResourceState,
    pub model_name: String,
    /// `true` while a reply stream is open.
    pub generating: bool,
    pub quitting: bool,
    pub spinner: usize,
    /// Transient status-bar message and when it was set.
    pub notice: Option<(String, Instant)>,
    pub last_output_height: u16,
    /// `None` until the first poll response.
    pub presence_state: Option<PresenceState>,
    pub sleep_cfg: SleepConfig,
    /// Local approximation of the daemon's idle clock.
    last_activity_at: Instant,
    pub vision_enabled: bool,
    pub modal: Option<ConfirmationModal>,
    pub listening: VoiceCaptureState,
    pub voice_output_enabled: bool,
    pub listen_active: bool,
    /// `(call id, command)` awaiting its result; the agent loop runs tools
    /// serially, so one slot suffices.
    pending_tool_call: Option<(String, String)>,
    pub pending_attachments: Vec<PendingAttachment>,
    picker: Option<Picker>,
    ipc: Arc<IpcClient>,
    /// The reply turn that owns the output pane; `None` between turns.
    active_reply: Option<ActiveReply>,
    /// An utterance transcribed while another turn owned the pane, held
    /// back so it is drawn above its own answer.
    queued_transcription: Option<QueuedTranscription>,
    in_flight_branch_op: Option<BranchOp>,
    /// Rows of a `/resume` listing, handed to the picker on `Done`.
    branches_buffer: Vec<BranchListEntry>,
    chat_tx: mpsc::Sender<ChatEvent>,
    /// Connection and attachment-loading tasks; aborted on drop.
    tasks: JoinSet<()>,
    slash_selected: usize,
    /// Set by Esc on the slash popup; cleared when the buffer leaves its
    /// `/` prefix.
    slash_dismissed: bool,
    /// `/resume` picker; `modal` takes precedence while both are open.
    pub picker_modal: Option<BranchPickerModal>,
    /// Throttles rewraps for a live thinking block's timer to 1 Hz.
    last_thinking_seconds: Option<u64>,
    pub session_title: Option<String>,
    /// Ctrl+O: force-expands every thinking and tool block without
    /// touching their per-item `expanded` flags.
    pub verbose: bool,
}

/// The reply turn that owns the output pane. The daemon serialises turns,
/// so a push-to-talk turn started mid-query is handed the pane afterwards.
struct ActiveReply {
    id: String,
    /// Answers `ConfirmRequest`; `None` for a push-to-talk turn, whose
    /// connection belongs to the voice proxy.
    writer: Option<mpsc::Sender<Request>>,
}

struct QueuedTranscription {
    id: String,
    text: String,
}

#[derive(Debug, Clone, Copy)]
enum BranchOp {
    Fork,
    Switch,
    Undo,
    Resume,
    New,
    ResumePicker,
}

#[derive(Debug, Clone)]
pub struct BranchListEntry {
    pub name: String,
    pub parent_branch_name: Option<String>,
    pub fork_point_seq: Option<i64>,
    pub message_count: i64,
    pub is_current_in_session: bool,
    pub is_active_session: bool,
    pub session_short: String,
    pub session_title: Option<String>,
}

/// Branch picker shown by `/resume`.
pub struct BranchPickerModal {
    pub entries: Vec<BranchListEntry>,
    pub selected: usize,
}

impl BranchPickerModal {
    /// The selected entry as a session-qualified `/switch` target.
    pub fn current_target(&self) -> Option<String> {
        self.entries
            .get(self.selected)
            .map(|e| format!("{}/{}", e.session_short, e.name))
    }
}

impl App {
    pub fn new(
        ipc: Arc<IpcClient>,
        chat_tx: mpsc::Sender<ChatEvent>,
        model_name: String,
        sleep_cfg: SleepConfig,
        vision_enabled: bool,
        picker: Option<Picker>,
    ) -> Self {
        Self {
            output: OutputPane::new(),
            input: InputLine::new(),
            throughput: ThroughputMeter::new(),
            resources: ResourceState::default(),
            model_name,
            generating: false,
            quitting: false,
            spinner: 0,
            notice: None,
            last_output_height: 10,
            presence_state: None,
            sleep_cfg,
            last_activity_at: Instant::now(),
            vision_enabled,
            modal: None,
            listening: VoiceCaptureState::Idle,
            voice_output_enabled: false,
            listen_active: false,
            pending_tool_call: None,
            pending_attachments: Vec::new(),
            picker,
            ipc,
            active_reply: None,
            queued_transcription: None,
            in_flight_branch_op: None,
            branches_buffer: Vec::new(),
            chat_tx,
            tasks: JoinSet::new(),
            slash_selected: 0,
            slash_dismissed: false,
            picker_modal: None,
            last_thinking_seconds: None,
            session_title: None,
            verbose: false,
        }
    }

    pub fn should_quit(&self) -> bool {
        self.quitting
    }

    pub fn spinner_char(&self) -> char {
        SPINNER_CHARS[self.spinner % SPINNER_CHARS.len()]
    }

    pub fn notice(&self) -> Option<&str> {
        self.notice.as_ref().map(|(s, _)| s.as_str())
    }

    pub fn set_output_height(&mut self, h: u16) {
        self.last_output_height = h;
    }

    pub fn on_resources(&mut self, v: ResourceState) {
        self.resources = v;
    }

    /// Local approximation of the daemon's countdown to its next idle
    /// transition. `None` when no transition is pending.
    pub fn local_time_until_next_transition(&self) -> Option<Duration> {
        let state = self.presence_state?;
        let next_threshold_secs = match state {
            PresenceState::Active => self.sleep_cfg.idle_to_drowsy_mins * 60,
            PresenceState::Drowsy => {
                (self.sleep_cfg.idle_to_drowsy_mins + self.sleep_cfg.idle_to_sleep_mins) * 60
            }
            PresenceState::Sleeping => return None,
        };
        if next_threshold_secs == 0 {
            return None;
        }
        let threshold = Duration::from_secs(next_threshold_secs);
        Some(threshold.saturating_sub(self.last_activity_at.elapsed()))
    }

    pub fn on_tick(&mut self) {
        while let Some(res) = self.tasks.try_join_next() {
            if let Err(e) = res {
                tracing::warn!("chat task failed: {e}");
            }
        }
        self.spinner = self.spinner.wrapping_add(1);
        if let Some((_, at)) = &self.notice
            && at.elapsed() > NOTICE_HOLD
        {
            self.notice = None;
        }
        let live_secs = self.output.live_thinking_seconds();
        if live_secs != self.last_thinking_seconds {
            self.last_thinking_seconds = live_secs;
            if live_secs.is_some() {
                self.output.refresh_live_thinking();
            }
        }
    }

    fn touch_activity(&mut self) {
        self.last_activity_at = Instant::now();
    }

    fn set_notice(&mut self, text: &str) {
        self.notice = Some((text.to_string(), Instant::now()));
    }
}

#[cfg(test)]
mod tests;
