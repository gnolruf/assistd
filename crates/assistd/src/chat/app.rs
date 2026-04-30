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
use crossterm::event::{KeyCode, KeyEvent};
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

/// One image staged by `/attach`, waiting to ride along with the user's
/// next text submission.
pub struct PendingAttachment {
    /// Display name (file basename) shown in the `📎×N` indicator and
    /// the user-prompt tag.
    pub name: String,
    pub mime: String,
    /// Reserved for future use (token-budget hints) — held alongside the
    /// bytes so we don't have to rescan on display.
    #[allow(dead_code)]
    pub size: usize,
    pub bytes: Vec<u8>,
    /// Pre-built terminal-graphics protocol for inline thumbnail
    /// rendering. `None` on terminals without graphics support — the
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

#[allow(clippy::large_enum_variant)]
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
    AttachLoaded {
        #[allow(dead_code)]
        path: String,
        name: String,
        mime: String,
        size: usize,
        bytes: Vec<u8>,
        protocol: Option<StatefulProtocol>,
    },
    /// `/attach <path>` failed (missing file, unsupported format, etc.).
    AttachFailed { path: String, message: String },
}

impl std::fmt::Debug for ChatEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatEvent::Wire(ev) => f.debug_tuple("Wire").field(ev).finish(),
            ChatEvent::WireError(msg) => f.debug_tuple("WireError").field(msg).finish(),
            ChatEvent::AttachLoaded {
                path,
                name,
                mime,
                size,
                protocol,
                ..
            } => f
                .debug_struct("AttachLoaded")
                .field("path", path)
                .field("name", name)
                .field("mime", mime)
                .field("size", size)
                .field("has_thumbnail", &protocol.is_some())
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
/// is ever active at a time — the agent loop on the daemon side blocks
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

pub struct App {
    pub output: OutputPane,
    pub input: InputLine,
    pub throughput: ThroughputMeter,
    pub resources: ResourceState,
    pub model_name: String,
    pub generating: bool,
    pub quitting: bool,
    pub spinner: usize,
    pub notice: Option<(String, Instant)>,
    pub last_output_height: u16,
    pub presence_state: Option<PresenceState>,
    pub sleep_cfg: SleepConfig,
    /// Local wallclock for the most recent user activity (typing /
    /// submitting / answering a modal). Used to compute the
    /// status-bar countdown without round-tripping to the daemon. The
    /// daemon authoritatively transitions on its own clock; this is a
    /// display approximation that drifts at most a couple seconds.
    last_activity_at: Instant,
    /// Set at chat startup from a daemon `GetCapabilities` probe.
    /// `false` means the loaded model has no mmproj — `/attach`
    /// rejects with the AC `vision not available` error and the
    /// status bar renders `vision: off`.
    pub vision_enabled: bool,
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
    /// strictly serial — only one tool runs at a time — so a single
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
    chat_tx: mpsc::Sender<ChatEvent>,
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
            active_writer: None,
            chat_tx,
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
            // Already prompting; auto-deny the second prompt so the
            // agent loop doesn't hang.
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

    /// Consume the modal (if any) and send the user's Y/N decision
    /// back to the daemon as a `Request::ConfirmResponse`. Used by
    /// both the Y/N key handlers and the quit path.
    fn resolve_modal(&mut self, decision: bool) {
        if let Some(modal) = self.modal.take() {
            self.send_confirm_response(&modal.confirm_id, decision);
        }
    }

    /// Enqueue a `ConfirmResponse` onto the active query's writer
    /// channel. No-op if no query is in flight (shouldn't happen — a
    /// modal can only have come from an active query — but handle
    /// defensively rather than panic).
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

    pub fn on_key(&mut self, ev: KeyEvent) {
        // Any key counts as user activity — even a Page Up scroll
        // means the user is at the keyboard. The daemon's idle
        // monitor does its own tracking based on Query traffic; this
        // local clock just keeps the status-bar countdown live.
        self.touch_activity();
        // Modal takes precedence over every other keybinding: when the
        // agent is blocked on a confirmation prompt, the only meaningful
        // input is Y / N / Esc. Scrolling and input-line edits stay
        // inert until the user decides.
        if self.modal.is_some() {
            self.handle_modal_key(ev);
            return;
        }
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
                if self.try_complete_attach_path() {
                    return;
                }
                self.output.toggle_last_tool_block();
                return;
            }
            _ => {}
        }
        match self.input.on_key(ev) {
            InputAction::None => {}
            InputAction::Submit(text) => {
                self.submit_typed(text);
            }
            InputAction::Quit => {
                // Closing during a pending modal isn't currently
                // reachable — modal intercepts input — but guard anyway
                // so the daemon's gate always gets a decision.
                self.resolve_modal(false);
                self.quitting = true;
            }
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
                // Swallow every other key while the modal is open — the
                // agent is blocked, so keystrokes meant for the input
                // line would be confusing.
            }
        }
    }

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
            ChatEvent::AttachLoaded {
                name,
                mime,
                size,
                bytes,
                protocol,
                ..
            } => {
                let label = format!("📎 attached: {name} ({mime}, {})", human_size_short(size));
                self.output.push_info(&label);
                self.set_notice(&format!("📎 {name} attached"));
                self.pending_attachments.push(PendingAttachment {
                    name,
                    mime,
                    size,
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
                    // The daemon auto-dispatches the transcription as
                    // a Query on the same connection, so the next
                    // events on this stream will be Deltas. We still
                    // render the user-prompt locally so the
                    // conversation reads naturally.
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
            Event::Done { .. } => {
                self.throughput.on_done(now);
                self.output.finish_assistant();
                self.generating = false;
                self.active_writer = None;
            }
            Event::Error { message, .. } => {
                self.throughput.on_done(now);
                self.output.finish_assistant();
                self.output.push_error(&message);
                self.generating = false;
                self.active_writer = None;
            }
            // Memory* events shouldn't appear on a query/PTT stream —
            // the daemon only emits them on `Request::Memory*`.
            Event::SemanticHit { .. }
            | Event::MemoryValue { .. }
            | Event::MemoryKeys { .. }
            | Event::MemoryRow { .. }
            | Event::MemoryForgetResult { .. }
            | Event::ReindexProgress { .. } => {}
        }
    }

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

    /// Tab completion for `/attach <partial-path>`. Returns `true` if
    /// completion fired (so the caller skips its fallback action), or
    /// `false` when the input isn't an attach line and the caller's
    /// default Tab behavior should run.
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

    /// Submit text typed at the input line. Slash-command parsing
    /// only happens here — a voice transcription that contains
    /// "/attach foo" is dispatched as plain text by the daemon's
    /// auto-Query path, which never lands here.
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
        if self.generating {
            self.set_notice("still generating — please wait");
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
                        .send(ChatEvent::AttachLoaded {
                            path: path_for_load,
                            name,
                            mime,
                            size,
                            bytes,
                            protocol,
                        })
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

    /// Open a daemon dialog connection for a Query and pump its
    /// streaming events onto `chat_tx`. The dialog connection stays
    /// open while the query is generating so the modal can write a
    /// `Request::ConfirmResponse` back on the same connection if the
    /// model triggers a destructive bash command.
    fn spawn_query(&mut self, text: String, attachments: Vec<Attachment>) {
        let ipc = self.ipc.clone();
        let chat_tx = self.chat_tx.clone();

        // Channel for outgoing requests from the main thread (modal Y/N
        // → ConfirmResponse). Capacity 8 is generous; one in-flight
        // confirm at a time is the only realistic shape.
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

            // Drive the connection: read events from the daemon AND
            // forward outgoing requests from the modal. Both halves
            // share the same `DialogConnection`, so we tokio::select.
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
                                // Sender dropped. Don't close the read
                                // half — the daemon may still emit
                                // events. Replace the branch with
                                // `pending` so it never fires again.
                                std::future::pending::<()>().await;
                            }
                        }
                    }
                }
            }
        });
    }
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
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

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
        // Bogus socket path — these tests never open a real connection.
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
}
