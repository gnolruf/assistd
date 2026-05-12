//! Top-level chat application state and reducer.
//!
//! `App` owns the output pane, input line, throughput meter, VRAM
//! state, and an `IpcClient` handle for talking to the daemon. The
//! `on_*` methods are pure reducers; I/O is confined to `spawn_query`
//! (opens a daemon dialog connection for a Query) and `handle_attach`
//! (loads an image from disk).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use assistd_core::{PresenceState, SleepConfig};
use assistd_ipc::{Event, IpcClient, Request, VoiceCaptureState};
use assistd_tools::{Attachment, ConfirmationRequest, load_image_attachment};
use crossterm::event::{KeyCode, KeyEvent, MouseEvent, MouseEventKind};
use ratatui_image::picker::Picker;
use ratatui_image::protocol::StatefulProtocol;
use tokio::sync::mpsc;
use uuid::Uuid;

use super::input::{InputAction, InputLine};
use super::output::OutputPane;
use super::throughput::ThroughputMeter;
use super::vram::ResourceState;

const SPINNER_CHARS: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
const NOTICE_HOLD: Duration = Duration::from_secs(3);
/// Rows scrolled per mouse-wheel tick. Three matches the de facto
/// terminal-app convention (Claude Code, less, htop, …) and keeps a
/// single click responsive without launching past several messages.
const MOUSE_WHEEL_STEP: u16 = 3;

/// Slash commands surfaced by the input-line autocomplete popup.
/// Each entry is `(command, usage_hint)` where `usage_hint` is shown
/// in dim text next to the command name in the suggestion list.
pub const SLASH_COMMANDS: &[(&str, &str)] = &[
    ("/attach", "<path>"),
    ("/fork", "<name>"),
    ("/new", ""),
    ("/resume", ""),
    ("/switch", "<target>"),
    ("/undo", ""),
];

/// One image staged by `/attach`, waiting to ride along with the user's
/// next text submission.
pub struct PendingAttachment {
    /// Display name (file basename) shown in the `📎×N` indicator and
    /// the user-prompt tag.
    pub name: String,
    pub mime: String,
    pub bytes: Vec<u8>,
    /// Pre-built terminal-graphics protocol for inline thumbnail
    /// rendering. `None` on terminals without graphics support; the
    /// `📎 attached: ...` info line still appears either way.
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

/// Payload for [`ChatEvent::AttachLoaded`]. Boxed inside the enum so a
/// `Vec<u8>` of image bytes plus a `StatefulProtocol` (which holds its
/// own pre-built per-cell terminal-graphics buffers) doesn't bloat the
/// other variants; `Wire(Event)` and `WireError(String)` were paying
/// the maximum-variant size on every send before the box.
pub struct AttachLoadedPayload {
    #[allow(dead_code)]
    pub path: String,
    pub name: String,
    pub mime: String,
    pub size: usize,
    pub bytes: Vec<u8>,
    pub protocol: Option<StatefulProtocol>,
}

pub enum ChatEvent {
    /// Streaming event from the daemon over IPC. Includes both
    /// query-response events (Delta/ToolCall/ToolResult/Done) and
    /// status-poll events (Presence/VoiceState/ListenState/...).
    Wire(Event),
    /// The wire connection ended in an unexpected way (I/O error,
    /// daemon closed mid-stream without `Done`).
    WireError(String),
    /// `/attach <path>` finished reading + validating the file. Carries
    /// everything the App needs to update the UI.
    AttachLoaded(Box<AttachLoadedPayload>),
    /// `/attach <path>` failed (missing file, unsupported format, etc.).
    AttachFailed { path: String, message: String },
}

impl std::fmt::Debug for ChatEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatEvent::Wire(ev) => f.debug_tuple("Wire").field(ev).finish(),
            ChatEvent::WireError(msg) => f.debug_tuple("WireError").field(msg).finish(),
            ChatEvent::AttachLoaded(p) => f
                .debug_struct("AttachLoaded")
                .field("path", &p.path)
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

/// Pending destructive-command prompt displayed as an overlay. Only one
/// is ever active at a time; the agent loop on the daemon side blocks
/// on the gate until we send the `ConfirmResponse`.
pub struct ConfirmationModal {
    /// Renders as the modal body (script + matched_pattern). Reuses the
    /// `assistd-tools::ConfirmationRequest` shape so the UI code can
    /// stay agnostic of where the request came from.
    pub request: ConfirmationRequest,
    /// Routing key the daemon sent in `Event::ConfirmRequest`. Echoed
    /// back verbatim in the outgoing `Request::ConfirmResponse`.
    confirm_id: String,
}

/// Top-level TUI application state.
///
/// Owns the output pane, input line, throughput meter, resource stats, and
/// the IPC connection to the daemon. The `on_*` methods are pure reducers;
/// I/O is confined to [`App::spawn_query`] and the attach handler.
pub struct App {
    /// Scrollable output region.
    pub output: OutputPane,
    /// Single-line input with readline keybindings and history.
    pub input: InputLine,
    /// Token-rate meter for the status bar.
    pub throughput: ThroughputMeter,
    /// Live VRAM / RAM readings.
    pub resources: ResourceState,
    /// Display name of the loaded model.
    pub model_name: String,
    /// `true` while a query stream is open.
    pub generating: bool,
    /// Set to `true` to exit the event loop.
    pub quitting: bool,
    /// Index into [`SPINNER_CHARS`], incremented on each tick.
    pub spinner: usize,
    /// Transient status-bar message with its display timestamp. Cleared by [`App::on_tick`] after [`NOTICE_HOLD`].
    pub notice: Option<(String, Instant)>,
    /// Height of the output area as of the last render, used for scrolling math.
    pub last_output_height: u16,
    /// Last-known daemon presence state; `None` until the first poll response.
    pub presence_state: Option<PresenceState>,
    /// Idle-sleep thresholds copied from config for the local countdown display.
    pub sleep_cfg: SleepConfig,
    /// Local wallclock for the most recent user activity (typing /
    /// submitting / answering a modal). Used to compute the
    /// status-bar countdown without round-tripping to the daemon. The
    /// daemon authoritatively transitions on its own clock; this is a
    /// display approximation that drifts at most a couple seconds.
    last_activity_at: Instant,
    /// Set at chat startup from a daemon `GetCapabilities` probe.
    /// `false` means the loaded model has no mmproj; `/attach`
    /// rejects with the AC `vision not available` error and the
    /// status bar renders `vision: off`.
    pub vision_enabled: bool,
    /// Active confirmation modal, if any. Only one can be open at a time.
    pub modal: Option<ConfirmationModal>,
    /// Push-to-talk capture state. Updated by `Event::VoiceState`
    /// flowing in over the wire. Rendered as an indicator in
    /// `render_status` (recording = red, transcribing = yellow).
    pub listening: VoiceCaptureState,
    /// TTS enabled flag, polled from the daemon. Rendered as the
    /// "voice-output: on/off" chip on the status bar.
    pub voice_output_enabled: bool,
    /// Continuous-listen active flag, polled from the daemon.
    pub listen_active: bool,
    /// Command of the in-flight tool call, captured on `Event::ToolCall`
    /// and consumed on the matching `Event::ToolResult` so the
    /// call+result pair becomes one `ToolBlock`. The agent loop is
    /// strictly serial (only one tool runs at a time), so a single
    /// slot suffices.
    pending_tool_call: Option<(String, String)>,
    /// Images staged by `/attach`, drained into the user's next
    /// submission.
    pub pending_attachments: Vec<PendingAttachment>,
    /// Terminal graphics-protocol picker, probed once at TUI startup.
    picker: Option<Picker>,
    /// Daemon connection factory.
    ipc: Arc<IpcClient>,
    /// Sender into the active query's bidirectional dialog connection.
    /// The query-driver task owns the read half + the write half; this
    /// `mpsc::Sender` lets the modal handler enqueue a
    /// `Request::ConfirmResponse` for the writer task to forward.
    /// `None` between queries.
    active_writer: Option<mpsc::Sender<Request>>,
    /// Tracks an in-flight branch command so [`Self::on_wire_event`]
    /// can route `BranchInfo` / `BranchSwitched` / `HistoryEntry` /
    /// `UndoApplied` events into the right rendering path. Cleared on
    /// terminal `Done` / `Error`.
    in_flight_branch_op: Option<BranchOp>,
    /// Buffer for branch rows accumulated during a `/resume` listing
    /// until the terminal `Done`, at which point they're handed off to
    /// the picker modal in one batch.
    branches_buffer: Vec<BranchListEntry>,
    chat_tx: mpsc::Sender<ChatEvent>,
    /// Highlighted entry in the slash-command suggestion popup.
    /// Reset to 0 whenever the buffer leaves a `/` prefix; clamped to
    /// `len-1` after each keystroke filters the list down.
    slash_selected: usize,
    /// Set by Esc while the slash popup is visible. Cleared once the
    /// buffer no longer starts with `/`, so the next `/<x>` reopens
    /// the popup as the user expects.
    slash_dismissed: bool,
    /// Interactive branch picker shown by `/resume`. Mutually
    /// exclusive with [`Self::modal`] (destructive-command modals);
    /// when both somehow co-exist, the destructive modal wins
    /// because the agent is blocked on it.
    pub picker_modal: Option<BranchPickerModal>,
}

/// Which branch slash-command is currently in flight, if any. Used to
/// distinguish `Event::BranchSwitched` arriving from `/fork` (no chat
/// repaint) from one arriving from `/switch` (clear + replay).
#[derive(Debug, Clone, Copy)]
enum BranchOp {
    Fork,
    Switch,
    Undo,
    /// In-flight `ResumeOrNew` issued at TUI startup. Suppresses the
    /// usual "[branch X is now active]" banner; on resume the
    /// HistoryEntry events repaint the conversation, on fresh-start
    /// the cleared output pane is the user-visible signal.
    Resume,
    /// In-flight `/new`: clear the chat pane silently when the
    /// daemon's `BranchSwitched` lands.
    New,
    /// In-flight `Branches` request triggered by `/resume`. Rows are
    /// buffered into `branches_buffer`; on terminal `Done` the
    /// branch-picker modal opens populated from that buffer.
    ResumePicker,
}

/// One row buffered during a `/resume` branch listing.
#[derive(Debug, Clone)]
pub struct BranchListEntry {
    pub name: String,
    pub parent_branch_name: Option<String>,
    pub fork_point_seq: Option<i64>,
    pub message_count: i64,
    pub is_current_in_session: bool,
    pub is_active_session: bool,
    pub session_short: String,
    /// LLM-generated session title, when available. Picker rows show
    /// this in place of `session_short` whenever it's set so resuming
    /// reads as "resume the chat about cats" rather than "resume
    /// 4f9a1b2c".
    pub session_title: Option<String>,
}

/// Interactive picker shown by `/resume`. Rendered as a modal overlay
/// with arrow-key navigation; Enter dispatches `Request::Switch`
/// against the qualified target, Esc cancels.
pub struct BranchPickerModal {
    pub entries: Vec<BranchListEntry>,
    pub selected: usize,
}

impl BranchPickerModal {
    /// Build the qualified `<session_prefix>/<name>` target the
    /// daemon's `Request::Switch` expects for unambiguous
    /// cross-session resolution.
    pub fn current_target(&self) -> Option<String> {
        self.entries
            .get(self.selected)
            .map(|e| format!("{}/{}", e.session_short, e.name))
    }
}

impl App {
    /// Construct a new `App` with the given IPC handle and configuration.
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
            active_writer: None,
            in_flight_branch_op: None,
            branches_buffer: Vec::new(),
            chat_tx,
            slash_selected: 0,
            slash_dismissed: false,
            picker_modal: None,
        }
    }

    /// Open a confirmation modal from a daemon-issued `Event::ConfirmRequest`.
    /// Only one modal can be active at a time; if a second prompt
    /// arrives while one is open we deny the new one immediately
    /// (which the daemon's gate maps to "cancel").
    pub fn open_confirmation_modal(
        &mut self,
        confirm_id: String,
        tool: String,
        script: String,
        matched_pattern: String,
    ) {
        if self.modal.is_some() {
            self.send_confirm_response(&confirm_id, false);
            return;
        }
        self.modal = Some(ConfirmationModal {
            request: ConfirmationRequest {
                tool,
                script,
                matched_pattern,
            },
            confirm_id,
        });
    }

    fn resolve_modal(&mut self, decision: bool) {
        if let Some(modal) = self.modal.take() {
            self.send_confirm_response(&modal.confirm_id, decision);
        }
    }

    fn send_confirm_response(&self, confirm_id: &str, allow: bool) {
        let Some(writer) = self.active_writer.clone() else {
            tracing::warn!(
                confirm_id,
                "no active query writer to forward ConfirmResponse"
            );
            return;
        };
        let req = Request::ConfirmResponse {
            id: Uuid::new_v4().to_string(),
            confirm_id: confirm_id.to_string(),
            allow,
        };
        tokio::spawn(async move {
            if let Err(e) = writer.send(req).await {
                tracing::warn!("ConfirmResponse send failed: {e}");
            }
        });
    }

    #[cfg(test)]
    pub fn has_modal(&self) -> bool {
        self.modal.is_some()
    }

    /// Returns `true` when the event loop should exit.
    pub fn should_quit(&self) -> bool {
        self.quitting
    }

    /// Current spinner character for the status bar.
    pub fn spinner_char(&self) -> char {
        SPINNER_CHARS[self.spinner % SPINNER_CHARS.len()]
    }

    /// Active notice text, if one is currently being displayed.
    pub fn notice(&self) -> Option<&str> {
        self.notice.as_ref().map(|(s, _)| s.as_str())
    }

    /// Record the output pane's rendered height so scrolling math stays correct.
    pub fn set_output_height(&mut self, h: u16) {
        self.last_output_height = h;
    }

    /// Slash-command entries that prefix-match the current buffer.
    /// Empty when the popup should be hidden (buffer doesn't start
    /// with `/`, contains whitespace, or the user dismissed it with
    /// Esc).
    pub fn slash_suggestions(&self) -> Vec<&'static (&'static str, &'static str)> {
        if self.slash_dismissed {
            return Vec::new();
        }
        let buf = self.input.buffer();
        if !buf.starts_with('/') {
            return Vec::new();
        }
        if buf.chars().any(char::is_whitespace) {
            return Vec::new();
        }
        SLASH_COMMANDS
            .iter()
            .filter(|(cmd, _)| cmd.starts_with(buf) && *cmd != buf)
            .collect()
    }

    /// Index of the currently highlighted suggestion, clamped to the
    /// visible list length. Used by [`super::ui`] to render the
    /// selection highlight.
    pub fn slash_selected(&self) -> usize {
        self.slash_selected
    }

    /// Replace the input buffer with the highlighted suggestion. No
    /// trailing space — the user adds one themselves if the command
    /// takes an argument.
    fn accept_slash_selection(&mut self) {
        let suggestions = self.slash_suggestions();
        if let Some((cmd, _)) = suggestions.get(self.slash_selected).copied() {
            self.input.set_buffer(cmd.to_string());
        }
    }

    /// Recompute slash-popup invariants after the input buffer mutates.
    /// Resets dismissal when the prefix leaves; clamps the selection
    /// to the (possibly smaller) filtered list.
    fn refresh_slash_state(&mut self) {
        let buf = self.input.buffer();
        if !buf.starts_with('/') {
            self.slash_dismissed = false;
            self.slash_selected = 0;
            return;
        }
        let n = self.slash_suggestions().len();
        if n == 0 {
            self.slash_selected = 0;
        } else if self.slash_selected >= n {
            self.slash_selected = n - 1;
        }
    }

    /// Handle a terminal mouse event. Only scroll-wheel ticks are
    /// consumed today; other mouse events (clicks, drags, motion) are
    /// ignored so the alt-screen behaves like a static viewport.
    pub fn on_mouse(&mut self, ev: MouseEvent) {
        match ev.kind {
            MouseEventKind::ScrollUp => {
                self.touch_activity();
                self.output.scroll_lines_up(MOUSE_WHEEL_STEP);
            }
            MouseEventKind::ScrollDown => {
                self.touch_activity();
                self.output.scroll_lines_down(MOUSE_WHEEL_STEP);
            }
            _ => {}
        }
    }

    /// Handle a terminal key event, routing to the modal or the input line.
    pub fn on_key(&mut self, ev: KeyEvent) {
        self.touch_activity();
        if self.modal.is_some() {
            self.handle_modal_key(ev);
            return;
        }
        if self.picker_modal.is_some() {
            self.handle_picker_key(ev);
            return;
        }
        let slash_active = !self.slash_suggestions().is_empty();
        match ev.code {
            KeyCode::PageUp => {
                self.output.scroll_page_up(self.last_output_height);
                return;
            }
            KeyCode::PageDown => {
                self.output.scroll_page_down(self.last_output_height);
                return;
            }
            KeyCode::F(2) => {
                self.on_cycle_key();
                return;
            }
            KeyCode::Tab => {
                if slash_active {
                    self.accept_slash_selection();
                    self.refresh_slash_state();
                    return;
                }
                if self.try_complete_attach_path() {
                    return;
                }
                self.output.toggle_last_tool_block();
                return;
            }
            KeyCode::Up if slash_active => {
                if self.slash_selected > 0 {
                    self.slash_selected -= 1;
                }
                return;
            }
            KeyCode::Down if slash_active => {
                let n = self.slash_suggestions().len();
                if self.slash_selected + 1 < n {
                    self.slash_selected += 1;
                }
                return;
            }
            KeyCode::Esc if slash_active => {
                self.slash_dismissed = true;
                return;
            }
            _ => {}
        }
        let action = self.input.on_key(ev);
        self.refresh_slash_state();
        match action {
            InputAction::None => {}
            InputAction::Submit(text) => {
                self.submit_typed(text);
            }
            InputAction::Quit => {
                self.resolve_modal(false);
                self.quitting = true;
            }
        }
    }

    /// Process keys while the `/resume` branch picker is open.
    /// Up/Down move the highlight, Enter dispatches a `Switch`, Esc
    /// cancels. Other keys are swallowed so the input line stays
    /// untouched.
    fn handle_picker_key(&mut self, ev: KeyEvent) {
        let Some(picker) = self.picker_modal.as_mut() else {
            return;
        };
        let len = picker.entries.len();
        match ev.code {
            KeyCode::Up => {
                if picker.selected > 0 {
                    picker.selected -= 1;
                }
            }
            KeyCode::Down => {
                if picker.selected + 1 < len {
                    picker.selected += 1;
                }
            }
            KeyCode::Home => picker.selected = 0,
            KeyCode::End => picker.selected = len.saturating_sub(1),
            KeyCode::Enter => self.picker_confirm(),
            KeyCode::Esc => {
                self.picker_modal = None;
            }
            _ => {}
        }
    }

    fn handle_modal_key(&mut self, ev: KeyEvent) {
        match ev.code {
            KeyCode::Char('y') | KeyCode::Char('Y') | KeyCode::Enter => {
                self.resolve_modal(true);
            }
            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                self.resolve_modal(false);
            }
            _ => {
                // Swallow every other key while the modal is open
            }
        }
    }

    /// Update live resource (VRAM / RAM) readings from the probe background task.
    pub fn on_resources(&mut self, v: ResourceState) {
        self.resources = v;
    }

    /// Approximate countdown to the next presence transition,
    /// computed locally from `last_activity_at + sleep_cfg`. The
    /// daemon owns the actual clock; this is a display courtesy
    /// that may drift by a couple seconds in either direction.
    /// Returns `None` when no countdown applies (presence unknown,
    /// or already Sleeping with no further transition pending).
    pub fn local_time_until_next_transition(&self) -> Option<Duration> {
        let state = self.presence_state?;
        let elapsed = self.last_activity_at.elapsed();
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
        if elapsed >= threshold {
            Some(Duration::from_secs(0))
        } else {
            Some(threshold - elapsed)
        }
    }

    fn touch_activity(&mut self) {
        self.last_activity_at = Instant::now();
    }

    fn on_cycle_key(&mut self) {
        let target = self
            .presence_state
            .map(|s| s.next())
            .unwrap_or(PresenceState::Active);
        self.set_notice(&format!("cycling → {}", presence_label(target)));
        let ipc = self.ipc.clone();
        let chat_tx = self.chat_tx.clone();
        tokio::spawn(async move {
            let req = Request::Cycle {
                id: Uuid::new_v4().to_string(),
            };
            match ipc.one_shot(req).await {
                Ok(mut stream) => {
                    while let Ok(Some(ev)) = stream.next_event().await {
                        let terminal = ev.is_terminal();
                        let _ = chat_tx.send(ChatEvent::Wire(ev)).await;
                        if terminal {
                            break;
                        }
                    }
                }
                Err(e) => {
                    let _ = chat_tx
                        .send(ChatEvent::WireError(format!("cycle: {e}")))
                        .await;
                }
            }
        });
    }

    /// Reducer entry point for everything the event loop pumps in.
    pub fn on_chat_event(&mut self, ev: ChatEvent) {
        match ev {
            ChatEvent::Wire(event) => self.on_wire_event(event),
            ChatEvent::WireError(msg) => {
                self.output.push_error(&format!("[wire error] {msg}"));
                self.generating = false;
                self.active_writer = None;
            }
            ChatEvent::AttachLoaded(payload) => {
                let AttachLoadedPayload {
                    path: _,
                    name,
                    mime,
                    size,
                    bytes,
                    protocol,
                } = *payload;
                let label = format!("📎 attached: {name} ({mime}, {})", human_size_short(size));
                self.output.push_info(&label);
                self.set_notice(&format!("📎 {name} attached"));
                self.pending_attachments.push(PendingAttachment {
                    name,
                    mime,
                    bytes,
                    protocol,
                });
            }
            ChatEvent::AttachFailed { path, message } => {
                self.output
                    .push_error(&format!("/attach {path}: {message}"));
                self.set_notice(&format!("📎 {path}: {message}"));
            }
        }
    }

    fn on_wire_event(&mut self, ev: Event) {
        let now = Instant::now();
        match ev {
            Event::Delta { text, .. } => {
                self.throughput.on_delta(now);
                self.output.append_assistant(&text);
            }
            Event::ToolCall { id, args, name, .. } => {
                let cmd = args
                    .get("command")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| name.clone());
                self.pending_tool_call = Some((id, cmd));
            }
            Event::ToolResult { id, result, .. } => {
                let body = result
                    .get("output")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                let exit_code = result
                    .get("exit_code")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0) as i32;
                let duration_ms = result
                    .get("duration_ms")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let command = self
                    .pending_tool_call
                    .take()
                    .filter(|(pid, _)| pid == &id)
                    .map(|(_, c)| c)
                    .unwrap_or_else(|| "<?>".to_string());
                self.output
                    .push_tool_block(command, body, exit_code, duration_ms);
            }
            Event::ConfirmRequest {
                confirm_id,
                tool,
                script,
                matched_pattern,
                ..
            } => {
                self.open_confirmation_modal(confirm_id, tool, script, matched_pattern);
            }
            Event::Presence { state, .. } => {
                self.presence_state = Some(state);
            }
            Event::VoiceState { state, .. } => {
                self.listening = state;
            }
            Event::Transcription { text, .. } => {
                if text.trim().is_empty() {
                    self.set_notice("no speech detected");
                } else {
                    self.output.push_user(&text);
                    self.output.reset_scroll();
                    self.output.begin_assistant();
                    self.throughput.reset();
                    self.generating = true;
                }
            }
            Event::ListenState { active, .. } => {
                self.listen_active = active;
            }
            Event::VoiceOutputState { enabled, .. } => {
                self.voice_output_enabled = enabled;
            }
            Event::Capabilities {
                vision, model_name, ..
            } => {
                self.vision_enabled = vision;
                if !model_name.is_empty() {
                    self.model_name = model_name;
                }
            }
            Event::Status {
                severity,
                component,
                event,
                message,
                ..
            } => {
                self.set_notice(&message);
                if event == "restarting" {
                    self.output.finish_assistant();
                    self.output.push_info(&format!("[{component} restarting…]"));
                } else if severity == "error" {
                    self.output.push_info(&format!("[{component}: {message}]"));
                }
            }
            Event::BranchInfo {
                name,
                parent_branch_name,
                fork_point_seq,
                message_count,
                is_current_in_session,
                is_active_session,
                session_id,
                session_title,
                ..
            } => {
                let session_short = session_id.chars().take(8).collect::<String>();
                self.branches_buffer.push(BranchListEntry {
                    name,
                    parent_branch_name,
                    fork_point_seq,
                    message_count,
                    is_current_in_session,
                    is_active_session,
                    session_short,
                    session_title,
                });
            }
            Event::BranchSwitched {
                name,
                parent_branch_name,
                fork_point_seq,
                session_title,
                ..
            } => match self.in_flight_branch_op {
                Some(BranchOp::Switch) => {
                    self.output.clear();
                    let msg = match session_title.as_deref().map(str::trim) {
                        Some(t) if !t.is_empty() => {
                            format!("[switched to conversation '{t}' on branch '{name}']")
                        }
                        _ => format!("[switched to new conversation on branch '{name}']"),
                    };
                    self.output.push_info(&msg);
                }
                Some(BranchOp::Resume) | Some(BranchOp::New) => {
                    // Startup resume / `/new`: silently clear so
                    // HistoryEntry events (if any) repaint into a
                    // clean pane, or the user simply sees a fresh
                    // chat.
                    self.output.clear();
                }
                _ => {
                    let detail = match (parent_branch_name.as_deref(), fork_point_seq) {
                        (Some(p), Some(seq)) => {
                            format!("[forked from '{p}'@seq{seq} into '{name}']")
                        }
                        _ => format!("[branch '{name}' is now active]"),
                    };
                    self.output.push_info(&detail);
                }
            },
            Event::HistoryEntry {
                role,
                content,
                tool_name,
                ..
            } => match role.as_str() {
                "user" => {
                    self.output.push_user(&content);
                }
                "assistant" => {
                    if !content.is_empty() {
                        self.output.begin_assistant();
                        self.output.append_assistant(&content);
                        self.output.finish_assistant();
                    }
                }
                "tool" => {
                    let name = tool_name.unwrap_or_default();
                    self.output.push_tool_block(name, content, 0, 0);
                }
                "system" => {
                    self.output.push_info(&content);
                }
                _ => self.output.push_info(&content),
            },
            Event::UndoApplied {
                removed_messages,
                last_user_text,
                ..
            } => {
                if removed_messages == 0 {
                    self.set_notice("nothing to undo");
                } else {
                    self.output.pop_last_user_exchange();
                    let preview = last_user_text
                        .as_deref()
                        .map(|t| t.chars().take(48).collect::<String>())
                        .unwrap_or_default();
                    if preview.is_empty() {
                        self.set_notice(&format!("undid {removed_messages} message(s)"));
                    } else {
                        self.set_notice(&format!("undid: {preview}"));
                    }
                }
            }
            Event::Done { .. } => {
                self.throughput.on_done(now);
                self.output.finish_assistant();
                self.generating = false;
                self.active_writer = None;
                if let Some(BranchOp::ResumePicker) = self.in_flight_branch_op {
                    self.open_branch_picker();
                }
                self.in_flight_branch_op = None;
            }
            Event::Error { message, .. } => {
                self.throughput.on_done(now);
                self.output.finish_assistant();
                self.output.push_error(&message);
                self.generating = false;
                self.active_writer = None;
                self.in_flight_branch_op = None;
                self.branches_buffer.clear();
            }
            Event::SemanticHit { .. }
            | Event::MemoryValue { .. }
            | Event::MemoryKeys { .. }
            | Event::MemoryRow { .. }
            | Event::MemoryForgetResult { .. }
            | Event::ReindexProgress { .. } => {}
        }
    }

    /// Advance the spinner and expire stale notices.
    pub fn on_tick(&mut self) {
        self.spinner = self.spinner.wrapping_add(1);
        if let Some((_, at)) = &self.notice {
            if at.elapsed() > NOTICE_HOLD {
                self.notice = None;
            }
        }
    }

    fn set_notice(&mut self, text: &str) {
        self.notice = Some((text.to_string(), Instant::now()));
    }

    fn try_complete_attach_path(&mut self) -> bool {
        let buffer = self.input.buffer();
        let partial = if let Some(rest) = buffer.strip_prefix("/attach ") {
            rest
        } else if let Some(rest) = buffer.strip_prefix("/attach_image ") {
            rest
        } else {
            return false;
        };
        let (dir, file_prefix) = match partial.rsplit_once('/') {
            Some((d, f)) => (
                if d.is_empty() {
                    PathBuf::from("/")
                } else {
                    expand_tilde(d)
                },
                f,
            ),
            None => (PathBuf::from("."), partial),
        };
        let entries: Vec<(String, bool)> = match std::fs::read_dir(&dir) {
            Ok(rd) => rd
                .filter_map(|e| e.ok())
                .filter_map(|e| {
                    let name = e.file_name().to_string_lossy().to_string();
                    if name.starts_with(file_prefix) {
                        let is_dir = e.file_type().map(|t| t.is_dir()).unwrap_or(false);
                        Some((name, is_dir))
                    } else {
                        None
                    }
                })
                .collect(),
            Err(_) => return true,
        };
        if entries.is_empty() {
            return true;
        }
        let names: Vec<&str> = entries.iter().map(|(n, _)| n.as_str()).collect();
        let lcp = longest_common_prefix(&names);
        let completed = if entries.len() == 1 {
            let (name, is_dir) = &entries[0];
            if *is_dir {
                format!("{name}/")
            } else {
                name.clone()
            }
        } else if lcp.len() > file_prefix.len() {
            lcp.to_string()
        } else {
            return true;
        };
        let cmd_prefix = if buffer.starts_with("/attach_image ") {
            "/attach_image "
        } else {
            "/attach "
        };
        let dir_part = match partial.rsplit_once('/') {
            Some((d, _)) => format!("{d}/"),
            None => String::new(),
        };
        self.input
            .set_buffer(format!("{cmd_prefix}{dir_part}{completed}"));
        true
    }

    fn submit_typed(&mut self, text: String) {
        let is_attach = text.starts_with("/attach ") || text.trim() == "/attach";
        if is_attach && !self.vision_enabled {
            self.output.push_error(
                "[error] attach_image: vision not available: model does not support \
                 images. Use: a model with mmproj loaded",
            );
            self.set_notice("vision not available");
            return;
        }
        if let Some(rest) = text.strip_prefix("/attach ") {
            self.handle_attach(rest.trim());
            return;
        }
        if text.trim() == "/attach" {
            self.output
                .push_error("/attach: expected a path. Usage: /attach <path>");
            self.set_notice("/attach: missing path");
            return;
        }
        if let Some(name) = parse_slash_arg(&text, "/fork") {
            self.handle_fork_cmd(name.to_string());
            return;
        }
        if let Some(target) = parse_slash_arg(&text, "/switch") {
            self.handle_switch_cmd(target.to_string());
            return;
        }
        if text.trim() == "/undo" {
            self.handle_undo_cmd();
            return;
        }
        if text.trim() == "/new" {
            self.handle_new_cmd();
            return;
        }
        if text.trim() == "/resume" {
            self.handle_resume_cmd();
            return;
        }
        if self.generating {
            self.set_notice("still generating, please wait");
            return;
        }
        let pending = std::mem::take(&mut self.pending_attachments);
        let attachment_names: Vec<String> = pending.iter().map(|a| a.name.clone()).collect();
        let mut attachments: Vec<Attachment> = Vec::with_capacity(pending.len());
        let mut thumbnails: Vec<(String, StatefulProtocol)> = Vec::new();
        for p in pending {
            let (att, proto, name) = p.into_parts();
            attachments.push(att);
            if let Some(pr) = proto {
                thumbnails.push((name, pr));
            }
        }
        self.begin_submit(&text, &attachment_names);
        for (name, protocol) in thumbnails {
            self.output.push_thumbnail(name, protocol);
        }
        self.spawn_query(text, attachments);
    }

    fn begin_submit(&mut self, text: &str, attachment_names: &[String]) {
        if attachment_names.is_empty() {
            self.output.push_user(text);
        } else {
            self.output
                .push_user_with_attachments(text, attachment_names);
        }
        self.output.reset_scroll();
        self.output.begin_assistant();
        self.throughput.reset();
        self.generating = true;
    }

    fn handle_attach(&mut self, raw: &str) {
        let args = match shlex::split(raw) {
            Some(v) => v,
            None => {
                self.output
                    .push_error("/attach: unterminated quote in path");
                self.set_notice("/attach: bad quoting");
                return;
            }
        };
        if args.len() != 1 {
            self.output.push_error(&format!(
                "/attach: expected exactly one path, got {}",
                args.len()
            ));
            self.set_notice("/attach: need one path");
            return;
        }
        let path = args.into_iter().next().expect("len == 1 checked above");
        let name = PathBuf::from(&path)
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.clone());
        self.set_notice(&format!("📎 reading {name}…"));
        let tx = self.chat_tx.clone();
        let picker = self.picker.clone();
        let path_for_load = path.clone();
        tokio::spawn(async move {
            match load_image_attachment(std::path::Path::new(&path_for_load)).await {
                Ok((Attachment::Image { mime, bytes }, size)) => {
                    let protocol = picker.and_then(|p| match image::load_from_memory(&bytes) {
                        Ok(img) => Some(p.new_resize_protocol(img)),
                        Err(e) => {
                            tracing::warn!(
                                "/attach: thumbnail decode failed for {path_for_load}: {e}"
                            );
                            None
                        }
                    });
                    let _ = tx
                        .send(ChatEvent::AttachLoaded(Box::new(AttachLoadedPayload {
                            path: path_for_load,
                            name,
                            mime,
                            size,
                            bytes,
                            protocol,
                        })))
                        .await;
                }
                Err(e) => {
                    let _ = tx
                        .send(ChatEvent::AttachFailed {
                            path: path_for_load,
                            message: e.user_message(),
                        })
                        .await;
                }
            }
        });
    }

    fn spawn_branch_command(&mut self, op: BranchOp, req: Request) {
        if self.in_flight_branch_op.is_some() {
            self.set_notice("branch command in flight, please wait");
            return;
        }
        if self.generating {
            self.set_notice("still generating, please wait");
            return;
        }
        self.in_flight_branch_op = Some(op);
        self.branches_buffer.clear();
        let ipc = self.ipc.clone();
        let chat_tx = self.chat_tx.clone();
        tokio::spawn(async move {
            let mut stream = match ipc.one_shot(req).await {
                Ok(s) => s,
                Err(e) => {
                    let _ = chat_tx
                        .send(ChatEvent::WireError(format!("branch connect: {e}")))
                        .await;
                    return;
                }
            };
            loop {
                match stream.next_event().await {
                    Ok(Some(ev)) => {
                        let terminal = ev.is_terminal();
                        let _ = chat_tx.send(ChatEvent::Wire(ev)).await;
                        if terminal {
                            return;
                        }
                    }
                    Ok(None) => {
                        let _ = chat_tx
                            .send(ChatEvent::WireError(
                                "daemon closed branch stream mid-flight".into(),
                            ))
                            .await;
                        return;
                    }
                    Err(e) => {
                        let _ = chat_tx
                            .send(ChatEvent::WireError(format!("branch read: {e}")))
                            .await;
                        return;
                    }
                }
            }
        });
    }

    /// Issue `Request::ResumeOrNew` at TUI startup so the daemon
    /// decides between resuming the current branch (when its latest
    /// message landed within `recency_secs`) and starting a fresh
    /// session. Wire events flow into the existing
    /// `BranchSwitched` / `HistoryEntry` handlers via `chat_tx`.
    pub fn spawn_resume_or_new(&mut self, recency_secs: u64) {
        let req = Request::ResumeOrNew {
            id: Uuid::new_v4().to_string(),
            recency_secs,
        };
        self.spawn_branch_command(BranchOp::Resume, req);
    }

    fn handle_fork_cmd(&mut self, name: String) {
        if name.trim().is_empty() {
            self.output
                .push_error("/fork: expected a name. Usage: /fork <name>");
            self.set_notice("/fork: missing name");
            return;
        }
        let req = Request::Fork {
            id: Uuid::new_v4().to_string(),
            name,
        };
        self.spawn_branch_command(BranchOp::Fork, req);
    }

    fn handle_switch_cmd(&mut self, target: String) {
        if target.trim().is_empty() {
            self.output
                .push_error("/switch: expected a target. Usage: /switch <name>");
            self.set_notice("/switch: missing target");
            return;
        }
        let req = Request::Switch {
            id: Uuid::new_v4().to_string(),
            target,
        };
        self.spawn_branch_command(BranchOp::Switch, req);
    }

    fn handle_undo_cmd(&mut self) {
        let req = Request::Undo {
            id: Uuid::new_v4().to_string(),
        };
        self.spawn_branch_command(BranchOp::Undo, req);
    }

    fn handle_new_cmd(&mut self) {
        let req = Request::NewSession {
            id: Uuid::new_v4().to_string(),
        };
        self.spawn_branch_command(BranchOp::New, req);
    }

    fn handle_resume_cmd(&mut self) {
        let req = Request::Branches {
            id: Uuid::new_v4().to_string(),
        };
        self.spawn_branch_command(BranchOp::ResumePicker, req);
    }

    /// Drain the branch buffer that just landed in response to
    /// `/resume` and open the interactive picker. The current branch
    /// (in the active session) starts highlighted so Enter on it is a
    /// no-op switch.
    fn open_branch_picker(&mut self) {
        let entries = std::mem::take(&mut self.branches_buffer);
        if entries.is_empty() {
            self.set_notice("no branches to resume");
            return;
        }
        let selected = entries
            .iter()
            .position(|e| e.is_active_session && e.is_current_in_session)
            .unwrap_or(0);
        self.picker_modal = Some(BranchPickerModal { entries, selected });
    }

    /// Dispatch a `Request::Switch` for the currently-highlighted
    /// branch in the picker, then close the modal. Used by the
    /// picker's Enter handler.
    fn picker_confirm(&mut self) {
        let target = match self.picker_modal.as_ref().and_then(|m| m.current_target()) {
            Some(t) => t,
            None => {
                self.picker_modal = None;
                return;
            }
        };
        self.picker_modal = None;
        self.handle_switch_cmd(target);
    }

    /// Open a daemon dialog connection for a Query and pump its
    /// streaming events onto `chat_tx`. The dialog connection stays
    /// open while the query is generating so the modal can write a
    /// `Request::ConfirmResponse` back on the same connection if the
    /// model triggers a destructive bash command.
    fn spawn_query(&mut self, text: String, attachments: Vec<Attachment>) {
        let ipc = self.ipc.clone();
        let chat_tx = self.chat_tx.clone();

        let (writer_tx, mut writer_rx) = mpsc::channel::<Request>(8);
        self.active_writer = Some(writer_tx);

        let req_id = Uuid::new_v4().to_string();
        let req = if attachments.is_empty() {
            Request::Query {
                id: req_id,
                text,
                attachments: Vec::new(),
                version: Some(assistd_ipc::PROTOCOL_VERSION),
            }
        } else {
            let wire_attachments: Vec<assistd_ipc::ImageAttachment> = attachments
                .into_iter()
                .map(|a| match a {
                    Attachment::Image { mime, bytes } => {
                        assistd_ipc::ImageAttachment::from_bytes(mime, &bytes)
                    }
                })
                .collect();
            Request::Query {
                id: req_id,
                text,
                attachments: wire_attachments,
                version: Some(assistd_ipc::PROTOCOL_VERSION),
            }
        };

        tokio::spawn(async move {
            let mut conn = match ipc.open_dialog(req).await {
                Ok(c) => c,
                Err(e) => {
                    let _ = chat_tx
                        .send(ChatEvent::WireError(format!("query connect: {e}")))
                        .await;
                    return;
                }
            };

            loop {
                tokio::select! {
                    maybe = conn.next_event() => {
                        match maybe {
                            Ok(Some(ev)) => {
                                let terminal = ev.is_terminal();
                                let _ = chat_tx.send(ChatEvent::Wire(ev)).await;
                                if terminal {
                                    return;
                                }
                            }
                            Ok(None) => {
                                let _ = chat_tx
                                    .send(ChatEvent::WireError(
                                        "daemon closed connection mid-stream".into(),
                                    ))
                                    .await;
                                return;
                            }
                            Err(e) => {
                                let _ = chat_tx
                                    .send(ChatEvent::WireError(format!("read: {e}")))
                                    .await;
                                return;
                            }
                        }
                    }
                    maybe_out = writer_rx.recv() => {
                        match maybe_out {
                            Some(req) => {
                                if let Err(e) = conn.send(req).await {
                                    let _ = chat_tx
                                        .send(ChatEvent::WireError(format!("write: {e}")))
                                        .await;
                                    return;
                                }
                            }
                            None => {
                                std::future::pending::<()>().await;
                            }
                        }
                    }
                }
            }
        });
    }
}

fn parse_slash_arg<'a>(text: &'a str, verb: &str) -> Option<&'a str> {
    let trimmed = text.trim_end();
    if trimmed == verb {
        return Some("");
    }
    let prefix = format!("{verb} ");
    trimmed.strip_prefix(&prefix).map(|rest| rest.trim())
}

fn expand_tilde(p: &str) -> PathBuf {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    if p == "~" {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home);
        }
    }
    PathBuf::from(p)
}

fn longest_common_prefix<'a>(xs: &[&'a str]) -> &'a str {
    let Some(first) = xs.first() else { return "" };
    let mut end = first.len();
    for s in &xs[1..] {
        end = end.min(s.len());
        while end > 0 && first.as_bytes()[..end] != s.as_bytes()[..end] {
            end -= 1;
        }
        if end == 0 {
            return "";
        }
    }
    &first[..end]
}

fn presence_label(s: PresenceState) -> &'static str {
    match s {
        PresenceState::Active => "active",
        PresenceState::Drowsy => "drowsy",
        PresenceState::Sleeping => "sleeping",
    }
}

fn human_size_short(n: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;
    if n >= GB {
        format!("{:.1}GB", n as f64 / GB as f64)
    } else if n >= MB {
        format!("{:.1}MB", n as f64 / MB as f64)
    } else if n >= KB {
        format!("{}KB", n / KB)
    } else {
        format!("{n}B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEvent, MouseEventKind};

    fn test_sleep_cfg() -> SleepConfig {
        let mut cfg = assistd_core::Config::default().sleep;
        cfg.idle_to_drowsy_mins = 0;
        cfg.idle_to_sleep_mins = 0;
        cfg
    }

    fn test_app() -> (App, mpsc::Receiver<ChatEvent>) {
        test_app_with(true)
    }

    fn test_app_with(vision_enabled: bool) -> (App, mpsc::Receiver<ChatEvent>) {
        let (tx, rx) = mpsc::channel::<ChatEvent>(16);
        // Bogus socket path; these tests never open a real connection.
        let ipc = Arc::new(IpcClient::with_path(std::path::PathBuf::from(
            "/tmp/assistd-test-nonexistent.sock",
        )));
        let app = App::new(
            ipc,
            tx,
            "test-model".into(),
            test_sleep_cfg(),
            vision_enabled,
            None,
        );
        (app, rx)
    }

    fn typed(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE)
    }

    fn delta(text: &str) -> Event {
        Event::Delta {
            id: "r".into(),
            text: text.into(),
        }
    }

    fn done() -> Event {
        Event::Done { id: "r".into() }
    }

    #[test]
    fn delta_keeps_generating_true() {
        let (mut app, _rx) = test_app();
        app.generating = true;
        app.on_chat_event(ChatEvent::Wire(delta("hi")));
        assert!(app.generating);
    }

    #[test]
    fn done_clears_generating() {
        let (mut app, _rx) = test_app();
        app.generating = true;
        app.on_chat_event(ChatEvent::Wire(delta("hi")));
        app.on_chat_event(ChatEvent::Wire(done()));
        assert!(!app.generating);
    }

    #[test]
    fn wire_error_clears_generating() {
        let (mut app, _rx) = test_app();
        app.generating = true;
        app.on_chat_event(ChatEvent::WireError("boom".into()));
        assert!(!app.generating);
    }

    #[test]
    fn page_up_increments_scroll() {
        let (mut app, _rx) = test_app();
        app.last_output_height = 10;
        app.on_key(KeyEvent::new(KeyCode::PageUp, KeyModifiers::NONE));
        assert!(app.output.scroll_offset() > 0);
    }

    fn wheel(kind: MouseEventKind) -> MouseEvent {
        MouseEvent {
            kind,
            column: 0,
            row: 0,
            modifiers: KeyModifiers::NONE,
        }
    }

    #[test]
    fn mouse_wheel_up_scrolls_history_up() {
        let (mut app, _rx) = test_app();
        let before = app.output.scroll_offset();
        app.on_mouse(wheel(MouseEventKind::ScrollUp));
        let after = app.output.scroll_offset();
        assert_eq!(after - before, MOUSE_WHEEL_STEP);
    }

    #[test]
    fn mouse_wheel_down_undoes_wheel_up() {
        let (mut app, _rx) = test_app();
        app.on_mouse(wheel(MouseEventKind::ScrollUp));
        app.on_mouse(wheel(MouseEventKind::ScrollUp));
        app.on_mouse(wheel(MouseEventKind::ScrollDown));
        assert_eq!(app.output.scroll_offset(), MOUSE_WHEEL_STEP);
    }

    #[test]
    fn mouse_non_wheel_events_are_ignored() {
        let (mut app, _rx) = test_app();
        app.on_mouse(wheel(MouseEventKind::Moved));
        assert_eq!(app.output.scroll_offset(), 0);
    }

    #[test]
    fn ctrl_c_on_empty_input_sets_quitting() {
        let (mut app, _rx) = test_app();
        app.on_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL));
        assert!(app.should_quit());
    }

    #[test]
    fn enter_while_generating_sets_notice() {
        let (mut app, _rx) = test_app();
        app.generating = true;
        app.on_key(typed('h'));
        app.on_key(typed('i'));
        app.on_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert!(app.notice().is_some());
    }

    #[test]
    fn on_tick_clears_stale_notice() {
        let (mut app, _rx) = test_app();
        app.notice = Some(("old".into(), Instant::now() - Duration::from_secs(10)));
        app.on_tick();
        assert!(app.notice().is_none());
    }

    #[test]
    fn spinner_char_cycles() {
        let (mut app, _rx) = test_app();
        let c0 = app.spinner_char();
        app.on_tick();
        let c1 = app.spinner_char();
        assert_ne!(c0, c1);
    }

    #[test]
    fn presence_event_updates_state() {
        let (mut app, _rx) = test_app();
        assert_eq!(app.presence_state, None);
        app.on_chat_event(ChatEvent::Wire(Event::Presence {
            id: "p".into(),
            state: PresenceState::Drowsy,
        }));
        assert_eq!(app.presence_state, Some(PresenceState::Drowsy));
        app.on_chat_event(ChatEvent::Wire(Event::Presence {
            id: "p".into(),
            state: PresenceState::Active,
        }));
        assert_eq!(app.presence_state, Some(PresenceState::Active));
    }

    #[tokio::test]
    async fn modal_approve_on_y() {
        let (mut app, _rx) = test_app();
        app.open_confirmation_modal(
            "c1".into(),
            "bash".into(),
            "rm -rf /tmp/junk".into(),
            "rm -rf".into(),
        );
        assert!(app.has_modal());
        app.on_key(typed('y'));
        assert!(!app.has_modal(), "modal should close on approve");
    }

    #[tokio::test]
    async fn modal_deny_on_n() {
        let (mut app, _rx) = test_app();
        app.open_confirmation_modal(
            "c1".into(),
            "bash".into(),
            "rm -rf /tmp/junk".into(),
            "rm -rf".into(),
        );
        app.on_key(typed('n'));
        assert!(!app.has_modal());
    }

    #[tokio::test]
    async fn modal_swallows_unrelated_keys() {
        let (mut app, _rx) = test_app();
        app.open_confirmation_modal(
            "c1".into(),
            "bash".into(),
            "rm -rf /tmp/junk".into(),
            "rm -rf".into(),
        );
        app.on_key(typed('x'));
        app.on_key(typed('z'));
        assert!(app.has_modal());
        assert!(app.input.buffer().is_empty());
    }

    #[test]
    fn tool_call_then_result_creates_one_block_with_command() {
        let (mut app, _rx) = test_app();
        app.on_chat_event(ChatEvent::Wire(Event::ToolCall {
            id: "c1".into(),
            name: "run".into(),
            args: serde_json::json!({"command": "ls /tmp"}),
        }));
        assert!(app.pending_tool_call.is_some());
        app.on_chat_event(ChatEvent::Wire(Event::ToolResult {
            id: "c1".into(),
            name: "run".into(),
            result: serde_json::json!({
                "output": "a\nb\n[exit:0 | 5ms]",
                "exit_code": 0,
                "truncated": false,
                "duration_ms": 5,
            }),
        }));
        assert!(app.pending_tool_call.is_none());
        let (lines, _) = app.output.render_view(80, 50);
        let rendered: Vec<String> = lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();
        assert!(rendered.iter().any(|l| l.contains("$ ls /tmp")));
        assert!(rendered.iter().any(|l| l.contains("[exit:0 | 5ms]")));
    }

    #[test]
    fn confirm_request_event_opens_modal() {
        let (mut app, _rx) = test_app();
        app.on_chat_event(ChatEvent::Wire(Event::ConfirmRequest {
            id: "r".into(),
            confirm_id: "c-xyz".into(),
            tool: "bash".into(),
            script: "rm -rf /tmp/foo".into(),
            matched_pattern: "rm -rf".into(),
        }));
        assert!(app.has_modal());
        let modal = app.modal.as_ref().unwrap();
        assert_eq!(modal.confirm_id, "c-xyz");
        assert_eq!(modal.request.script, "rm -rf /tmp/foo");
    }

    #[test]
    fn capabilities_event_updates_vision_and_model_name() {
        let (mut app, _rx) = test_app_with(false);
        assert!(!app.vision_enabled);
        app.on_chat_event(ChatEvent::Wire(Event::Capabilities {
            id: "c".into(),
            vision: true,
            model_name: "Qwen".into(),
        }));
        assert!(app.vision_enabled);
        assert_eq!(app.model_name, "Qwen");
    }

    fn type_str(app: &mut App, s: &str) {
        for c in s.chars() {
            app.on_key(typed(c));
        }
    }

    #[test]
    fn slash_popup_shows_after_slash() {
        let (mut app, _rx) = test_app();
        assert!(app.slash_suggestions().is_empty());
        type_str(&mut app, "/");
        let s = app.slash_suggestions();
        assert_eq!(s.len(), SLASH_COMMANDS.len());
    }

    #[test]
    fn slash_popup_filters_by_prefix() {
        let (mut app, _rx) = test_app();
        type_str(&mut app, "/fo");
        let s = app.slash_suggestions();
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].0, "/fork");
    }

    #[test]
    fn slash_popup_hides_after_whitespace() {
        let (mut app, _rx) = test_app();
        type_str(&mut app, "/attach ");
        assert!(app.slash_suggestions().is_empty());
    }

    #[test]
    fn tab_accepts_selection_and_fills_buffer() {
        let (mut app, _rx) = test_app();
        type_str(&mut app, "/f");
        app.on_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(app.input.buffer(), "/fork");
        assert!(app.slash_suggestions().is_empty());
    }

    #[test]
    fn down_moves_selection_when_popup_active() {
        let (mut app, _rx) = test_app();
        type_str(&mut app, "/");
        assert_eq!(app.slash_selected(), 0);
        app.on_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        assert_eq!(app.slash_selected(), 1);
    }

    #[test]
    fn esc_dismisses_popup_without_clearing_buffer() {
        let (mut app, _rx) = test_app();
        type_str(&mut app, "/at");
        assert!(!app.slash_suggestions().is_empty());
        app.on_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert!(app.slash_suggestions().is_empty());
        assert_eq!(app.input.buffer(), "/at");
    }

    #[test]
    fn slash_registry_contains_new_and_resume() {
        let names: Vec<&str> = SLASH_COMMANDS.iter().map(|(c, _)| *c).collect();
        assert!(names.contains(&"/new"));
        assert!(names.contains(&"/resume"));
    }

    fn picker_entry(
        name: &str,
        session: &str,
        current: bool,
        active_sess: bool,
    ) -> BranchListEntry {
        BranchListEntry {
            name: name.into(),
            parent_branch_name: None,
            fork_point_seq: None,
            message_count: 0,
            is_current_in_session: current,
            is_active_session: active_sess,
            session_short: session.into(),
            session_title: None,
        }
    }

    #[test]
    fn open_branch_picker_highlights_active_current() {
        let (mut app, _rx) = test_app();
        app.branches_buffer = vec![
            picker_entry("main", "aaaaaaaa", false, false),
            picker_entry("main", "bbbbbbbb", true, true),
            picker_entry("feat", "bbbbbbbb", false, true),
        ];
        app.open_branch_picker();
        let picker = app.picker_modal.as_ref().expect("picker opened");
        assert_eq!(picker.selected, 1);
        assert_eq!(picker.entries.len(), 3);
    }

    #[test]
    fn open_branch_picker_with_no_entries_sets_notice() {
        let (mut app, _rx) = test_app();
        app.branches_buffer.clear();
        app.open_branch_picker();
        assert!(app.picker_modal.is_none());
        assert!(app.notice.is_some());
    }

    #[test]
    fn picker_current_target_is_session_qualified() {
        let modal = BranchPickerModal {
            entries: vec![picker_entry("feature-x", "deadbeef", false, false)],
            selected: 0,
        };
        assert_eq!(
            modal.current_target().as_deref(),
            Some("deadbeef/feature-x")
        );
    }

    #[test]
    fn picker_arrow_keys_move_selection() {
        let (mut app, _rx) = test_app();
        app.picker_modal = Some(BranchPickerModal {
            entries: vec![
                picker_entry("a", "11111111", false, false),
                picker_entry("b", "11111111", false, false),
                picker_entry("c", "11111111", false, false),
            ],
            selected: 0,
        });
        app.on_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        app.on_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        assert_eq!(app.picker_modal.as_ref().unwrap().selected, 2);
        app.on_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(app.picker_modal.as_ref().unwrap().selected, 1);
    }

    #[test]
    fn picker_esc_cancels_without_dispatching() {
        let (mut app, _rx) = test_app();
        app.picker_modal = Some(BranchPickerModal {
            entries: vec![picker_entry("a", "11111111", false, false)],
            selected: 0,
        });
        app.on_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert!(app.picker_modal.is_none());
    }

    #[test]
    fn dismissal_resets_after_buffer_clears() {
        let (mut app, _rx) = test_app();
        type_str(&mut app, "/at");
        app.on_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        // Wipe the buffer; popup should re-arm on the next `/`.
        for _ in 0..3 {
            app.on_key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE));
        }
        type_str(&mut app, "/");
        assert!(!app.slash_suggestions().is_empty());
    }
}
