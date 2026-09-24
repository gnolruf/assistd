//! Chat application state and reducer. The `on_*` methods mutate state
//! only; I/O lives in the `spawn_*` methods and the attach handler.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use assistd_core::{PresenceState, SleepConfig};
use assistd_ipc::{Event, IpcClient, Request, Role, StatusKind, StatusSeverity, VoiceCaptureState};
use assistd_tools::{Attachment, ConfirmationRequest, load_image_attachment};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEvent, MouseEventKind};
use ratatui_image::picker::Picker;
use ratatui_image::protocol::StatefulProtocol;
use tokio::sync::mpsc;
use tokio::task::JoinSet;
use uuid::Uuid;

use super::input::{InputAction, InputLine};
use super::output::OutputPane;
use super::throughput::ThroughputMeter;
use super::vram::ResourceState;

const SPINNER_CHARS: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
const NOTICE_HOLD: Duration = Duration::from_secs(3);
const MOUSE_WHEEL_STEP: u16 = 3;

/// `(command, usage_hint)` pairs for the autocomplete popup.
pub const SLASH_COMMANDS: &[(&str, &str)] = &[
    ("/attach", "<path>"),
    ("/fork", "<name>"),
    ("/new", ""),
    ("/resume", ""),
    ("/switch", "<target>"),
    ("/undo", ""),
];

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

/// Which concurrent daemon connection an event arrived on. Several run
/// at once and all feed one reducer, so the tag says which slice of
/// `App` state a stream's terminal event may retire.
#[derive(Debug, Clone, Copy)]
pub enum WireStream {
    /// A query dialog or a push-to-talk turn. Owns the assistant
    /// message, [`App::generating`] and the query writer.
    Reply,
    /// A branch command. Owns the in-flight branch op and its rows.
    Branch,
    /// Polls and the F2 cycle. Indicator state only.
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

/// Keys that approve are ignored for this long after the modal opens,
/// so a keystroke aimed at the input line cannot approve a command the
/// user has not yet seen.
const CONFIRM_ARM_DELAY: Duration = Duration::from_millis(750);

/// Destructive-command prompt shown as an overlay while the daemon's
/// agent loop blocks on the answer.
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
    /// Local approximation of the daemon's idle clock, for the
    /// status-bar countdown.
    last_activity_at: Instant,
    pub vision_enabled: bool,
    pub modal: Option<ConfirmationModal>,
    pub listening: VoiceCaptureState,
    pub voice_output_enabled: bool,
    pub listen_active: bool,
    /// `(call id, command)` of the tool call awaiting its result. One
    /// slot suffices because the agent loop runs tools serially.
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
    /// Connection and attachment-loading tasks; aborted when the app is
    /// dropped.
    tasks: JoinSet<()>,
    slash_selected: usize,
    /// Set by Esc on the slash popup; cleared when the buffer leaves
    /// its `/` prefix.
    slash_dismissed: bool,
    /// Mutually exclusive with `modal`; when both exist the destructive
    /// modal wins because the agent is blocked on it.
    pub picker_modal: Option<BranchPickerModal>,
    /// Throttles rewraps for a live thinking block's timer to 1 Hz.
    last_thinking_seconds: Option<u64>,
    pub session_title: Option<String>,
    /// Ctrl+O. Force-expands every thinking and tool block without
    /// touching their per-item `expanded` flags.
    pub verbose: bool,
}

/// The reply turn that owns the output pane. A push-to-talk turn can
/// start while a typed query is still streaming; the daemon serialises
/// turns, so the pane is handed from one to the next rather than shared.
struct ActiveReply {
    id: String,
    /// Answers `ConfirmRequest`. `None` for a push-to-talk turn, whose
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

    /// Show a destructive-command prompt. A second prompt arriving while
    /// one is open is denied immediately.
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
            opened_at: Instant::now(),
        });
    }

    fn resolve_modal(&mut self, decision: bool) {
        if let Some(modal) = self.modal.take() {
            self.send_confirm_response(&modal.confirm_id, decision);
        }
    }

    fn send_confirm_response(&mut self, confirm_id: &str, allow: bool) {
        let Some(writer) = self
            .active_reply
            .as_ref()
            .and_then(|reply| reply.writer.clone())
        else {
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
        self.tasks.spawn(async move {
            if let Err(e) = writer.send(req).await {
                tracing::warn!("ConfirmResponse send failed: {e}");
            }
        });
    }

    #[cfg(test)]
    pub fn has_modal(&self) -> bool {
        self.modal.is_some()
    }

    #[cfg(test)]
    fn arm_modal(&mut self) {
        if let Some(modal) = self.modal.as_mut() {
            modal.opened_at = Instant::now() - CONFIRM_ARM_DELAY;
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

    /// Empty when the popup should be hidden.
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

    pub fn slash_selected(&self) -> usize {
        self.slash_selected
    }

    fn accept_slash_selection(&mut self) {
        let suggestions = self.slash_suggestions();
        if let Some((cmd, _)) = suggestions.get(self.slash_selected).copied() {
            self.input.set_buffer(cmd.to_string());
        }
    }

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
                self.output.toggle_last_expandable();
                return;
            }
            KeyCode::Char('o') if ev.modifiers.contains(KeyModifiers::CONTROL) => {
                self.verbose = !self.verbose;
                self.output.set_verbose(self.verbose);
                self.set_notice(if self.verbose {
                    "verbose mode: on"
                } else {
                    "verbose mode: off"
                });
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

    fn handle_picker_key(&mut self, ev: KeyEvent) {
        let Some(picker) = self.picker_modal.as_mut() else {
            return;
        };
        let len = picker.entries.len();
        match ev.code {
            KeyCode::Up if picker.selected > 0 => {
                picker.selected -= 1;
            }
            KeyCode::Down if picker.selected + 1 < len => {
                picker.selected += 1;
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
        let armed = self.modal.as_ref().is_some_and(ConfirmationModal::armed);
        match ev.code {
            KeyCode::Char('y') | KeyCode::Char('Y') if armed => {
                self.resolve_modal(true);
            }
            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                self.resolve_modal(false);
            }
            _ => {}
        }
    }

    pub fn on_resources(&mut self, v: ResourceState) {
        self.resources = v;
    }

    /// Local approximation of the daemon's countdown to its next idle
    /// transition. `None` when no transition is pending.
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
        let req = Request::Cycle {
            id: Uuid::new_v4().to_string(),
        };
        self.spawn_one_shot(req, WireStream::Status, "cycle");
    }

    /// Send `req` on a fresh connection and pump its events into the
    /// reducer tagged `stream`, reporting connection or read failures as
    /// a `WireError` on the same stream.
    fn spawn_one_shot(&mut self, req: Request, stream: WireStream, label: &'static str) {
        let ipc = self.ipc.clone();
        let chat_tx = self.chat_tx.clone();
        self.tasks.spawn(async move {
            let wire_error = |message: String| ChatEvent::WireError { stream, message };
            let mut events = match ipc.one_shot(req).await {
                Ok(s) => s,
                Err(e) => {
                    let _ = chat_tx
                        .send(wire_error(format!("{label} connect: {e}")))
                        .await;
                    return;
                }
            };
            loop {
                let outcome = match events.next_event().await {
                    Ok(Some(event)) => {
                        let terminal = event.is_terminal();
                        let _ = chat_tx.send(ChatEvent::Wire { stream, event }).await;
                        if terminal {
                            return;
                        }
                        continue;
                    }
                    Ok(None) => wire_error(format!("{label}: daemon closed stream mid-flight")),
                    Err(e) => wire_error(format!("{label} read: {e}")),
                };
                let _ = chat_tx.send(outcome).await;
                return;
            }
        });
    }

    pub fn on_chat_event(&mut self, ev: ChatEvent) {
        match ev {
            ChatEvent::Wire { stream, event } => self.on_wire_event(stream, event),
            ChatEvent::WireError { stream, message } => {
                self.output.push_error(&format!("[wire error] {message}"));
                match stream {
                    WireStream::Reply => self.finish_reply(Instant::now()),
                    WireStream::Branch => self.fail_branch_op(),
                    WireStream::Status => {}
                }
            }
            ChatEvent::AttachLoaded(payload) => {
                let AttachLoadedPayload {
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

    fn on_wire_event(&mut self, stream: WireStream, ev: Event) {
        if matches!(stream, WireStream::Reply) && !self.accept_reply_event(&ev) {
            return;
        }
        let now = Instant::now();
        match ev {
            Event::Delta { text, .. } => {
                self.output.finish_thinking();
                self.throughput.on_delta(now);
                self.output.append_assistant(&text);
            }
            Event::ReasoningDelta { text, .. } => {
                self.throughput.on_delta(now);
                self.output.append_thinking(&text);
            }
            Event::ToolCall { id, args, name, .. } => {
                self.output.finish_thinking();
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
                    self.begin_voice_turn(&text);
                }
            }
            Event::ListenState { active, .. } => {
                self.listen_active = active;
            }
            Event::VoiceOutputState { enabled, .. } => {
                self.voice_output_enabled = enabled;
            }
            Event::SpeakingState { .. } => {}
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
                if event == StatusKind::Restarting {
                    self.output.finish_thinking();
                    self.output.finish_assistant();
                    self.output.push_info(&format!("[{component} restarting…]"));
                } else if severity == StatusSeverity::Error {
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
            } => self.on_branch_switched(name, parent_branch_name, fork_point_seq, session_title),
            Event::SessionTitle { title, .. } => {
                self.session_title = Some(title);
            }
            Event::HistoryEntry {
                role,
                content,
                tool_name,
                ..
            } => self.on_history_entry(role, content, tool_name),
            Event::UndoApplied {
                removed_messages,
                last_user_text,
                ..
            } => self.on_undo_applied(removed_messages, last_user_text),
            Event::Done { .. } => match stream {
                WireStream::Reply => self.finish_reply(now),
                WireStream::Branch => self.finish_branch_op(),
                WireStream::Status => {}
            },
            Event::Error { message, .. } => {
                self.output.push_error(&message);
                match stream {
                    WireStream::Reply => self.finish_reply(now),
                    WireStream::Branch => self.fail_branch_op(),
                    WireStream::Status => {}
                }
            }
            Event::SemanticHit { .. }
            | Event::MemoryValue { .. }
            | Event::MemoryKeys { .. }
            | Event::MemoryRow { .. }
            | Event::MemoryForgetResult { .. }
            | Event::ReindexProgress { .. }
            | Event::LastDelta { .. } => {}
        }
    }

    /// `BranchSwitched` always carries the now-active session's title, so
    /// a missing one clears it.
    fn on_branch_switched(
        &mut self,
        name: String,
        parent_branch_name: Option<String>,
        fork_point_seq: Option<i64>,
        session_title: Option<String>,
    ) {
        let title = session_title
            .as_deref()
            .map(str::trim)
            .filter(|t| !t.is_empty());
        self.session_title = title.map(str::to_string);
        match self.in_flight_branch_op {
            Some(BranchOp::Switch) => {
                self.output.clear();
                let msg = match title {
                    Some(t) => format!("[switched to conversation '{t}' on branch '{name}']"),
                    None => format!("[switched to new conversation on branch '{name}']"),
                };
                self.output.push_info(&msg);
            }
            Some(BranchOp::Resume) | Some(BranchOp::New) => {
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
        }
    }

    fn on_history_entry(&mut self, role: Role, content: String, tool_name: Option<String>) {
        match role {
            Role::User => self.output.push_user(&content),
            Role::Assistant => {
                if !content.is_empty() {
                    self.output.begin_assistant();
                    self.output.append_assistant(&content);
                    self.output.finish_assistant();
                }
            }
            Role::Tool => {
                self.output
                    .push_tool_block(tool_name.unwrap_or_default(), content, 0, 0);
            }
            Role::System => self.output.push_info(&content),
        }
    }

    fn on_undo_applied(&mut self, removed_messages: u32, last_user_text: Option<String>) {
        if removed_messages == 0 {
            self.set_notice("nothing to undo");
            return;
        }
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

    fn accept_reply_event(&mut self, ev: &Event) -> bool {
        let Some(id) = turn_scoped_id(ev) else {
            return true;
        };
        match self.active_reply.as_ref().map(|reply| reply.id == id) {
            Some(true) => true,
            Some(false) => {
                self.defer_reply_event(id, ev);
                false
            }
            None => {
                self.active_reply = Some(ActiveReply {
                    id: id.to_string(),
                    writer: None,
                });
                if let Some(queued) = self.queued_transcription.take()
                    && queued.id == id
                {
                    self.begin_voice_turn(&queued.text);
                }
                true
            }
        }
    }

    fn defer_reply_event(&mut self, id: &str, ev: &Event) {
        match ev {
            Event::Transcription { text, .. } => {
                self.queued_transcription = Some(QueuedTranscription {
                    id: id.to_string(),
                    text: text.clone(),
                });
            }
            Event::Error { message, .. } => {
                self.set_notice(message);
                self.drop_queued_transcription(id);
            }
            Event::Done { .. } => self.drop_queued_transcription(id),
            _ => {}
        }
    }

    fn drop_queued_transcription(&mut self, id: &str) {
        if self
            .queued_transcription
            .as_ref()
            .is_some_and(|q| q.id == id)
        {
            self.queued_transcription = None;
        }
    }

    fn begin_voice_turn(&mut self, text: &str) {
        self.output.push_user(text);
        self.output.reset_scroll();
        self.output.begin_assistant();
        self.throughput.reset();
        self.generating = true;
    }

    fn finish_reply(&mut self, now: Instant) {
        self.throughput.on_done(now);
        self.output.finish_thinking();
        self.output.finish_assistant();
        self.generating = false;
        self.active_reply = None;
        self.modal = None;
        self.last_thinking_seconds = None;
    }

    fn finish_branch_op(&mut self) {
        if let Some(BranchOp::ResumePicker) = self.in_flight_branch_op.take() {
            self.open_branch_picker();
        }
    }

    fn fail_branch_op(&mut self) {
        self.in_flight_branch_op = None;
        self.branches_buffer.clear();
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

    fn set_notice(&mut self, text: &str) {
        self.notice = Some((text.to_string(), Instant::now()));
    }

    fn try_complete_attach_path(&mut self) -> bool {
        let buffer = self.input.buffer();
        let Some(partial) = buffer.strip_prefix("/attach ") else {
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
        let dir_part = match partial.rsplit_once('/') {
            Some((d, _)) => format!("{d}/"),
            None => String::new(),
        };
        self.input
            .set_buffer(format!("/attach {dir_part}{completed}"));
        true
    }

    fn submit_typed(&mut self, text: String) {
        match SlashCommand::parse(&text) {
            Some(SlashCommand::Attach(_)) if !self.vision_enabled => {
                self.output.push_error(
                    "[error] attach: vision not available: model does not support \
                     images. Use: a model with mmproj loaded",
                );
                self.set_notice("vision not available");
                return;
            }
            Some(SlashCommand::Attach(path)) => {
                if path.is_empty() {
                    self.output
                        .push_error("/attach: expected a path. Usage: /attach <path>");
                    self.set_notice("/attach: missing path");
                } else {
                    self.handle_attach(&path);
                }
                return;
            }
            Some(SlashCommand::Fork(name)) => return self.handle_fork_cmd(name),
            Some(SlashCommand::Switch(target)) => return self.handle_switch_cmd(target),
            Some(SlashCommand::Undo) => return self.handle_undo_cmd(),
            Some(SlashCommand::New) => return self.handle_new_cmd(),
            Some(SlashCommand::Resume) => return self.handle_resume_cmd(),
            None => {}
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
        self.tasks.spawn(async move {
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
        self.spawn_one_shot(req, WireStream::Branch, "branch");
    }

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

    fn spawn_query(&mut self, text: String, attachments: Vec<Attachment>) {
        let ipc = self.ipc.clone();
        let chat_tx = self.chat_tx.clone();

        let (writer_tx, mut writer_rx) = mpsc::channel::<Request>(8);
        let req_id = Uuid::new_v4().to_string();
        self.active_reply = Some(ActiveReply {
            id: req_id.clone(),
            writer: Some(writer_tx),
        });

        let req = if attachments.is_empty() {
            Request::query(req_id, text)
        } else {
            let wire_attachments: Vec<assistd_ipc::ImageAttachment> = attachments
                .into_iter()
                .map(|a| match a {
                    Attachment::Image { mime, bytes } => {
                        assistd_ipc::ImageAttachment::from_bytes(mime, &bytes)
                    }
                })
                .collect();
            Request::query_with_attachments(req_id, text, wire_attachments)
        };

        self.tasks.spawn(async move {
            let mut conn = match ipc.open_dialog(req).await {
                Ok(c) => c,
                Err(e) => {
                    let _ = chat_tx
                        .send(ChatEvent::WireError {
                            stream: WireStream::Reply,
                            message: format!("query connect: {e}"),
                        })
                        .await;
                    return;
                }
            };

            let mut writer_open = true;
            loop {
                tokio::select! {
                    maybe = conn.next_event() => {
                        match maybe {
                            Ok(Some(ev)) => {
                                let terminal = ev.is_terminal();
                                let _ = chat_tx
                                    .send(ChatEvent::Wire {
                                        stream: WireStream::Reply,
                                        event: ev,
                                    })
                                    .await;
                                if terminal {
                                    return;
                                }
                            }
                            Ok(None) => {
                                let _ = chat_tx
                                    .send(ChatEvent::WireError {
                                        stream: WireStream::Reply,
                                        message: "daemon closed connection mid-stream".into(),
                                    })
                                    .await;
                                return;
                            }
                            Err(e) => {
                                let _ = chat_tx
                                    .send(ChatEvent::WireError {
                                        stream: WireStream::Reply,
                                        message: format!("read: {e}"),
                                    })
                                    .await;
                                return;
                            }
                        }
                    }
                    maybe_out = writer_rx.recv(), if writer_open => {
                        match maybe_out {
                            Some(req) => {
                                if let Err(e) = conn.send(req).await {
                                    let _ = chat_tx
                                        .send(ChatEvent::WireError {
                                            stream: WireStream::Reply,
                                            message: format!("write: {e}"),
                                        })
                                        .await;
                                    return;
                                }
                            }
                            None => writer_open = false,
                        }
                    }
                }
            }
        });
    }
}

fn turn_scoped_id(ev: &Event) -> Option<&str> {
    match ev {
        Event::Delta { id, .. }
        | Event::ReasoningDelta { id, .. }
        | Event::ToolCall { id, .. }
        | Event::ToolResult { id, .. }
        | Event::ConfirmRequest { id, .. }
        | Event::Done { id, .. }
        | Event::Error { id, .. } => Some(id),
        Event::Transcription { id, text } => (!text.trim().is_empty()).then_some(id.as_str()),
        _ => None,
    }
}

/// A typed slash command with its trimmed argument.
enum SlashCommand {
    Attach(String),
    Fork(String),
    Switch(String),
    Undo,
    New,
    Resume,
}

impl SlashCommand {
    /// `None` when `text` is not a slash command; unknown commands are
    /// sent to the model as ordinary text.
    fn parse(text: &str) -> Option<Self> {
        let trimmed = text.trim();
        let (verb, arg) = match trimmed.split_once(char::is_whitespace) {
            Some((verb, rest)) => (verb, rest.trim()),
            None => (trimmed, ""),
        };
        match verb {
            "/attach" => Some(Self::Attach(arg.to_string())),
            "/fork" => Some(Self::Fork(arg.to_string())),
            "/switch" => Some(Self::Switch(arg.to_string())),
            "/undo" => Some(Self::Undo),
            "/new" => Some(Self::New),
            "/resume" => Some(Self::Resume),
            _ => None,
        }
    }
}

fn expand_tilde(p: &str) -> PathBuf {
    if let Some(rest) = p.strip_prefix("~/")
        && let Ok(home) = std::env::var("HOME")
    {
        return PathBuf::from(home).join(rest);
    }
    if p == "~"
        && let Ok(home) = std::env::var("HOME")
    {
        return PathBuf::from(home);
    }
    PathBuf::from(p)
}

fn longest_common_prefix<'a>(xs: &[&'a str]) -> &'a str {
    let Some((first, rest)) = xs.split_first() else {
        return "";
    };
    let end = rest.iter().fold(first.len(), |end, s| {
        first[..end]
            .char_indices()
            .zip(s.chars())
            .find(|((_, a), b)| a != b)
            .map_or(end.min(s.len()), |((i, _), _)| i)
    });
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
mod tests;
