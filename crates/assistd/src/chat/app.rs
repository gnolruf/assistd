//! Top-level chat application state and reducer.
//!
//! `App` owns the output pane, input line, throughput meter, VRAM state,
//! and a handle to an `LlmBackend` for spawning generation tasks. The
//! `on_*` methods are pure reducers; I/O is confined to `spawn_generation`.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use assistd_core::{PresenceManager, PresenceState, SleepConfig, ToolRegistry, VoiceCaptureState};
use assistd_llm::{LlmBackend, LlmEvent};
use assistd_tools::{Attachment, ConfirmationRequest, load_image_attachment};
use crossterm::event::{KeyCode, KeyEvent};
use ratatui_image::picker::Picker;
use ratatui_image::protocol::StatefulProtocol;
use tokio::sync::{mpsc, oneshot};

use super::gate::PendingConfirmation;
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

/// Where a `submit` was triggered from. Slash-command parsing only fires
/// for `Typed` submissions so a voice transcription that happens to
/// contain "/attach foo" goes to the model as plain text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitOrigin {
    Typed,
    Voice,
}

// `AttachLoaded`'s `bytes: Vec<u8>` and `protocol` push its size well
// past the other variants — but it's a one-shot transient event flowing
// through a bounded mpsc, so the size asymmetry is fine. Boxing the
// payload would just trade one heap alloc per attach for two.
#[allow(clippy::large_enum_variant)]
pub enum ChatEvent {
    Llm(LlmEvent),
    LlmError(String),
    /// `/attach <path>` finished reading + validating the file. Carries
    /// everything the App needs to update the UI.
    AttachLoaded {
        /// Original path the user typed — kept for diagnostics; the
        /// reducer only uses `name` for the visible label.
        #[allow(dead_code)]
        path: String,
        name: String,
        mime: String,
        size: usize,
        bytes: Vec<u8>,
        /// Pre-built graphics-protocol state for the inline thumbnail.
        /// `None` when the terminal doesn't support a graphics protocol
        /// or image decoding failed (the bytes still get sent to the
        /// LLM — only the local rendering fallback differs).
        protocol: Option<StatefulProtocol>,
    },
    /// `/attach <path>` failed (missing file, unsupported format, etc.).
    AttachFailed {
        path: String,
        message: String,
    },
}

// Manual Debug because `StatefulProtocol` (inside `AttachLoaded.protocol`)
// does not implement Debug. Hides the protocol blob behind a placeholder
// so the rest of the variant fields stay debuggable.
impl std::fmt::Debug for ChatEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatEvent::Llm(ev) => f.debug_tuple("Llm").field(ev).finish(),
            ChatEvent::LlmError(msg) => f.debug_tuple("LlmError").field(msg).finish(),
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

/// Pending destructive-command prompt displayed as an overlay. Only one is
/// ever active at a time — the agent loop is blocked on the gate's
/// `oneshot` while we wait for the user's Y/N decision.
pub struct ConfirmationModal {
    pub request: ConfirmationRequest,
    responder: oneshot::Sender<bool>,
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
    pub presence: Option<Arc<PresenceManager>>,
    pub modal: Option<ConfirmationModal>,
    /// Push-to-talk capture state, driven by the mic-voice listener
    /// spawned alongside the event loop. Rendered as an indicator in
    /// `render_status` (recording = red, transcribing = yellow).
    pub listening: VoiceCaptureState,
    /// Command of the in-flight tool call, captured on `LlmEvent::ToolCall`
    /// and consumed on the matching `LlmEvent::ToolResult` so the
    /// call+result pair becomes one `ToolBlock`. The agent loop is strictly
    /// serial — only one tool runs at a time — so a single slot suffices.
    pending_tool_call: Option<(String, String)>,
    /// Images staged by `/attach`, drained into the user's next
    /// submission. Persists across multiple `/attach` invocations so a
    /// user can attach several images before sending a query.
    pub pending_attachments: Vec<PendingAttachment>,
    /// Terminal graphics-protocol picker, probed once at TUI startup.
    /// `None` on terminals without Kitty / Sixel / iTerm2 support — in
    /// that case `/attach` falls back to filename-only display.
    /// Cloned into the spawned attach-load task so the worker can build
    /// `StatefulProtocol`s without holding a lock back to the UI loop.
    picker: Option<Picker>,
    client: Arc<dyn LlmBackend>,
    tools: Arc<ToolRegistry>,
    max_iterations: u32,
    chat_tx: mpsc::Sender<ChatEvent>,
}

impl App {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        client: Arc<dyn LlmBackend>,
        tools: Arc<ToolRegistry>,
        max_iterations: u32,
        chat_tx: mpsc::Sender<ChatEvent>,
        model_name: String,
        sleep_cfg: SleepConfig,
        presence: Option<Arc<PresenceManager>>,
        picker: Option<Picker>,
    ) -> Self {
        let presence_state = presence.as_ref().map(|p| p.state());
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
            presence_state,
            sleep_cfg,
            presence,
            modal: None,
            listening: VoiceCaptureState::Idle,
            pending_tool_call: None,
            pending_attachments: Vec::new(),
            picker,
            client,
            tools,
            max_iterations,
            chat_tx,
        }
    }

    /// Called by the TUI event loop when a destructive-command request
    /// arrives from the gate. Only one modal is supported at a time; if a
    /// second request arrives while one is pending we reject the newer one
    /// immediately (its oneshot is dropped, which the gate interprets as
    /// a denial). This is a defensive guard — it should not happen because
    /// the agent loop blocks on the first prompt before issuing a second
    /// bash invocation.
    pub fn open_confirmation_modal(&mut self, pending: PendingConfirmation) {
        if self.modal.is_some() {
            // Drop the new request's responder → gate returns false.
            drop(pending);
            return;
        }
        self.modal = Some(ConfirmationModal {
            request: pending.request,
            responder: pending.responder,
        });
    }

    /// Consume the modal (if any) and send `decision` through the stored
    /// oneshot. Used by both the Y/N key handlers and the quit path.
    fn resolve_modal(&mut self, decision: bool) {
        if let Some(modal) = self.modal.take() {
            let _ = modal.responder.send(decision);
        }
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
                self.output.toggle_last_tool_block();
                return;
            }
            _ => {}
        }
        match self.input.on_key(ev) {
            InputAction::None => {}
            InputAction::Submit(text) => {
                self.submit_with_origin(text, SubmitOrigin::Typed);
            }
            InputAction::Quit => {
                // Closing during a pending modal isn't currently
                // reachable — modal intercepts input — but guard anyway
                // so the gate always gets a decision.
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

    pub fn on_presence(&mut self, s: PresenceState) {
        self.presence_state = Some(s);
    }

    /// Reducer for `VoiceCaptureState` transitions from the mic
    /// listener. Called on every watch-channel change; sets the
    /// rendered indicator.
    pub fn on_voice_state(&mut self, s: VoiceCaptureState) {
        self.listening = s;
    }

    /// Handle a completed transcription from the PTT pipeline. Empty
    /// strings (VAD trimmed everything) surface as a brief notice.
    /// Non-empty text is auto-submitted to the LLM via the same path
    /// as typing into the input line, but with `Voice` origin so a
    /// transcription containing "/attach" is treated as plain text
    /// rather than parsed as a slash command.
    pub fn on_transcription(&mut self, text: String) {
        if text.trim().is_empty() {
            self.set_notice("no speech detected");
            return;
        }
        if self.generating {
            self.set_notice("still generating — transcription dropped");
            return;
        }
        self.submit_with_origin(text, SubmitOrigin::Voice);
    }

    /// Surface a non-fatal voice pipeline error as a TUI notice. The
    /// listening indicator is reset so stale "recording…" doesn't
    /// linger on the status bar.
    pub fn on_voice_error(&mut self, msg: String) {
        self.listening = VoiceCaptureState::Idle;
        self.set_notice(&format!("voice: {msg}"));
    }

    fn on_cycle_key(&mut self) {
        let Some(p) = self.presence.clone() else {
            self.set_notice("presence unavailable");
            return;
        };
        let target = self.presence_state.unwrap_or_else(|| p.state()).next();
        self.set_notice(&format!("cycling → {}", presence_label(target)));
        tokio::spawn(async move {
            if let Err(e) = p.cycle().await {
                tracing::warn!("F2 cycle failed: {e:#}");
            }
        });
    }

    pub fn on_llm_event(&mut self, ev: ChatEvent) {
        let now = Instant::now();
        match ev {
            ChatEvent::Llm(LlmEvent::Delta { text }) => {
                self.throughput.on_delta(now);
                self.output.append_assistant(&text);
            }
            ChatEvent::Llm(LlmEvent::ToolCall { id, arguments, .. }) => {
                let cmd = arguments
                    .get("command")
                    .and_then(|v| v.as_str())
                    .unwrap_or("<?>")
                    .to_string();
                self.pending_tool_call = Some((id, cmd));
            }
            ChatEvent::Llm(LlmEvent::ToolResult { id, result, .. }) => {
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
            ChatEvent::Llm(LlmEvent::Done) => {
                self.throughput.on_done(now);
                self.output.finish_assistant();
                self.generating = false;
            }
            ChatEvent::LlmError(msg) => {
                self.throughput.on_done(now);
                self.output.finish_assistant();
                self.output.push_error(&msg);
                self.generating = false;
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

    pub fn on_tick(&mut self) {
        self.spinner = self.spinner.wrapping_add(1);
        if let Some((_, at)) = &self.notice {
            if at.elapsed() > NOTICE_HOLD {
                self.notice = None;
            }
        }
    }

    pub fn on_resources(&mut self, v: ResourceState) {
        self.resources = v;
    }

    fn set_notice(&mut self, text: &str) {
        self.notice = Some((text.to_string(), Instant::now()));
    }

    fn submit_with_origin(&mut self, text: String, origin: SubmitOrigin) {
        if origin == SubmitOrigin::Typed {
            if let Some(rest) = text.strip_prefix("/attach ") {
                self.handle_attach(rest.trim());
                return;
            }
            // Reject a bare `/attach` (no path) early so we don't ship the
            // literal string to the LLM and waste a turn.
            if text.trim() == "/attach" {
                self.output
                    .push_error("/attach: expected a path. Usage: /attach <path>");
                self.set_notice("/attach: missing path");
                return;
            }
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
        self.spawn_generation(text, attachments);
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
                    // Build the graphics-protocol thumbnail when the
                    // terminal supports it. `image::load_from_memory`
                    // can fail on rare valid-but-unusual encodings (e.g.
                    // 16-bit PNGs the `image` crate doesn't enable by
                    // default) — fall back to filename-only display in
                    // that case rather than failing the whole attach.
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

    fn spawn_generation(&self, text: String, attachments: Vec<Attachment>) {
        let client = self.client.clone();
        let tools = self.tools.clone();
        let max_iterations = self.max_iterations;
        let tx = self.chat_tx.clone();
        let presence = self.presence.clone();
        tokio::spawn(async move {
            // Auto-wake and take an in-flight guard so a user hitting
            // Enter from Drowsy/Sleeping sees the response stream once
            // the model is ready, and so a concurrent sleep waits for
            // this generation to finish before tearing down
            // llama-server. The guard is held until the end of this
            // task.
            let _request_guard = if let Some(p) = presence.as_ref() {
                match p.acquire_request_guard().await {
                    Ok(g) => Some(g),
                    Err(e) => {
                        let _ = tx
                            .send(ChatEvent::LlmError(format!("wake failed: {e:#}")))
                            .await;
                        return;
                    }
                }
            } else {
                None
            };
            let (llm_tx, mut llm_rx) = mpsc::channel::<LlmEvent>(64);
            let tx_fwd = tx.clone();
            let forwarder = tokio::spawn(async move {
                while let Some(ev) = llm_rx.recv().await {
                    if tx_fwd.send(ChatEvent::Llm(ev)).await.is_err() {
                        return;
                    }
                }
            });
            let result = assistd_core::run_agent_turn(
                client,
                tools,
                max_iterations,
                text,
                attachments,
                llm_tx,
            )
            .await;
            let _ = forwarder.await;
            if let Err(e) = result {
                let _ = tx.send(ChatEvent::LlmError(e.to_string())).await;
            }
        });
    }
}

fn presence_label(s: PresenceState) -> &'static str {
    match s {
        PresenceState::Active => "active",
        PresenceState::Drowsy => "drowsy",
        PresenceState::Sleeping => "sleeping",
    }
}

/// Compact bytes-to-human formatter for the `📎 attached: ...` line.
/// Mirrors `assistd_tools::commands::cat::human_size` but is duplicated
/// here so the chat TUI doesn't depend on a `pub(crate)` helper.
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
    use assistd_llm::{EchoBackend, FailedBackend};
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    fn test_sleep_cfg() -> SleepConfig {
        let mut cfg = assistd_core::Config::default().sleep;
        cfg.idle_to_drowsy_mins = 0;
        cfg.idle_to_sleep_mins = 0;
        cfg
    }

    fn test_app() -> (App, mpsc::Receiver<ChatEvent>) {
        let (tx, rx) = mpsc::channel::<ChatEvent>(16);
        let client: Arc<dyn LlmBackend> = Arc::new(EchoBackend::new());
        let tools = Arc::new(ToolRegistry::default());
        let app = App::new(
            client,
            tools,
            20,
            tx,
            "test-model".into(),
            test_sleep_cfg(),
            None,
            None,
        );
        (app, rx)
    }

    fn typed(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE)
    }

    #[test]
    fn delta_keeps_generating_true() {
        let (mut app, _rx) = test_app();
        app.generating = true;
        app.on_llm_event(ChatEvent::Llm(LlmEvent::Delta { text: "hi".into() }));
        assert!(app.generating);
    }

    #[test]
    fn done_clears_generating() {
        let (mut app, _rx) = test_app();
        app.generating = true;
        app.on_llm_event(ChatEvent::Llm(LlmEvent::Delta { text: "hi".into() }));
        app.on_llm_event(ChatEvent::Llm(LlmEvent::Done));
        assert!(!app.generating);
    }

    #[test]
    fn llm_error_clears_generating() {
        let (mut app, _rx) = test_app();
        app.generating = true;
        app.on_llm_event(ChatEvent::LlmError("boom".into()));
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
    fn on_presence_updates_state() {
        let (mut app, _rx) = test_app();
        assert_eq!(app.presence_state, None);
        app.on_presence(PresenceState::Drowsy);
        assert_eq!(app.presence_state, Some(PresenceState::Drowsy));
        app.on_presence(PresenceState::Active);
        assert_eq!(app.presence_state, Some(PresenceState::Active));
    }

    #[test]
    fn f2_without_presence_sets_notice_not_panic() {
        let (mut app, _rx) = test_app();
        app.on_key(KeyEvent::new(KeyCode::F(2), KeyModifiers::NONE));
        assert_eq!(
            app.notice().map(|s| s.to_string()),
            Some("presence unavailable".to_string())
        );
    }

    fn pending_confirmation() -> (PendingConfirmation, oneshot::Receiver<bool>) {
        let (tx, rx) = oneshot::channel();
        let pending = PendingConfirmation {
            request: ConfirmationRequest {
                tool: "bash".into(),
                script: "rm -rf /tmp/junk".into(),
                matched_pattern: "rm -rf".into(),
            },
            responder: tx,
        };
        (pending, rx)
    }

    #[tokio::test]
    async fn modal_approve_on_y() {
        let (mut app, _rx) = test_app();
        let (pending, responder_rx) = pending_confirmation();
        app.open_confirmation_modal(pending);
        assert!(app.has_modal());
        app.on_key(typed('y'));
        assert!(!app.has_modal(), "modal should close on approve");
        let decision = responder_rx.await.expect("responder not dropped");
        assert!(decision);
    }

    #[tokio::test]
    async fn modal_approve_on_uppercase_y() {
        let (mut app, _rx) = test_app();
        let (pending, responder_rx) = pending_confirmation();
        app.open_confirmation_modal(pending);
        app.on_key(typed('Y'));
        assert!(responder_rx.await.unwrap());
    }

    #[tokio::test]
    async fn modal_approve_on_enter() {
        let (mut app, _rx) = test_app();
        let (pending, responder_rx) = pending_confirmation();
        app.open_confirmation_modal(pending);
        app.on_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert!(responder_rx.await.unwrap());
    }

    #[tokio::test]
    async fn modal_deny_on_n() {
        let (mut app, _rx) = test_app();
        let (pending, responder_rx) = pending_confirmation();
        app.open_confirmation_modal(pending);
        app.on_key(typed('n'));
        assert!(!app.has_modal());
        assert!(!responder_rx.await.unwrap());
    }

    #[tokio::test]
    async fn modal_deny_on_esc() {
        let (mut app, _rx) = test_app();
        let (pending, responder_rx) = pending_confirmation();
        app.open_confirmation_modal(pending);
        app.on_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert!(!responder_rx.await.unwrap());
    }

    #[tokio::test]
    async fn modal_swallows_unrelated_keys() {
        let (mut app, _rx) = test_app();
        let (pending, _responder_rx) = pending_confirmation();
        app.open_confirmation_modal(pending);
        // Typing into the input line must NOT land in the buffer while
        // the modal is open — it would be confusing.
        app.on_key(typed('x'));
        app.on_key(typed('z'));
        assert!(app.has_modal());
        assert!(app.input.buffer().is_empty());
    }

    #[tokio::test]
    async fn modal_second_request_while_open_is_denied() {
        let (mut app, _rx) = test_app();
        let (first, first_rx) = pending_confirmation();
        let (second, second_rx) = pending_confirmation();
        app.open_confirmation_modal(first);
        // Second request arrives while the first is pending — the gate
        // must not hang; instead, the second request's responder is
        // dropped, which the gate interprets as denial.
        app.open_confirmation_modal(second);
        // First is still the active modal.
        assert!(app.has_modal());
        assert!(
            second_rx.await.is_err(),
            "second responder should be dropped"
        );
        // Resolving the first still works.
        app.on_key(typed('y'));
        assert!(first_rx.await.unwrap());
    }

    #[tokio::test]
    async fn echo_backend_spawn_yields_delta_then_done() {
        let (app, mut rx) = test_app();
        app.spawn_generation("hello".into(), Vec::new());
        let first = rx.recv().await.expect("delta");
        let second = rx.recv().await.expect("done");
        match first {
            ChatEvent::Llm(LlmEvent::Delta { text }) => assert_eq!(text, "hello"),
            other => panic!("expected Delta, got {other:?}"),
        }
        assert!(matches!(second, ChatEvent::Llm(LlmEvent::Done)));
    }

    #[tokio::test]
    async fn failed_backend_spawn_yields_llm_error() {
        let (tx, mut rx) = mpsc::channel::<ChatEvent>(16);
        let client: Arc<dyn LlmBackend> = Arc::new(FailedBackend::new("server down".into()));
        let tools = Arc::new(ToolRegistry::default());
        let app = App::new(
            client,
            tools,
            20,
            tx,
            "test-model".into(),
            test_sleep_cfg(),
            None,
            None,
        );
        app.spawn_generation("hello".into(), Vec::new());
        let ev = rx.recv().await.expect("error event");
        match ev {
            ChatEvent::LlmError(msg) => assert!(msg.contains("server down")),
            other => panic!("expected LlmError, got {other:?}"),
        }
    }

    fn rendered_text(app: &mut App) -> Vec<String> {
        let (lines, _) = app.output.render_view(80, 50);
        lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect()
    }

    #[test]
    fn tool_call_then_result_creates_one_block_with_command() {
        let (mut app, _rx) = test_app();
        app.on_llm_event(ChatEvent::Llm(LlmEvent::ToolCall {
            id: "c1".into(),
            name: "run".into(),
            arguments: serde_json::json!({"command": "ls /tmp"}),
        }));
        assert!(app.pending_tool_call.is_some());
        app.on_llm_event(ChatEvent::Llm(LlmEvent::ToolResult {
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
        let rendered = rendered_text(&mut app);
        assert!(rendered.iter().any(|l| l.contains("$ ls /tmp")));
        assert!(rendered.iter().any(|l| l.contains("[exit:0 | 5ms]")));
    }

    #[test]
    fn tab_toggles_last_tool_block() {
        let (mut app, _rx) = test_app();
        let body: String =
            (0..30).map(|i| format!("l{i}\n")).collect::<String>() + "[exit:0 | 1ms]";
        app.on_llm_event(ChatEvent::Llm(LlmEvent::ToolCall {
            id: "c1".into(),
            name: "run".into(),
            arguments: serde_json::json!({"command": "seq 30"}),
        }));
        app.on_llm_event(ChatEvent::Llm(LlmEvent::ToolResult {
            id: "c1".into(),
            name: "run".into(),
            result: serde_json::json!({
                "output": body,
                "exit_code": 0,
                "truncated": false,
                "duration_ms": 1,
            }),
        }));
        let collapsed_marker_count = |app: &mut App| {
            rendered_text(app)
                .iter()
                .filter(|l| l.contains("more lines"))
                .count()
        };
        assert_eq!(collapsed_marker_count(&mut app), 1);
        app.on_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(collapsed_marker_count(&mut app), 0);
        app.on_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(collapsed_marker_count(&mut app), 1);
    }

    #[test]
    fn tool_result_without_matching_call_renders_with_placeholder() {
        let (mut app, _rx) = test_app();
        app.on_llm_event(ChatEvent::Llm(LlmEvent::ToolResult {
            id: "lost".into(),
            name: "run".into(),
            result: serde_json::json!({
                "output": "[exit:0 | 1ms]",
                "exit_code": 0,
                "truncated": false,
                "duration_ms": 1,
            }),
        }));
        let rendered = rendered_text(&mut app);
        assert!(rendered.iter().any(|l| l.contains("$ <?>")));
    }

    #[test]
    fn tab_with_no_blocks_is_safe_noop_and_not_typed() {
        let (mut app, _rx) = test_app();
        app.on_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert!(app.input.buffer().is_empty());
    }

    // --- /attach slash command -------------------------------------------

    /// Minimal valid 1×1 PNG that `infer` recognizes as `image/png`.
    /// Same bytes used by `assistd-tools::commands::see` tests so we
    /// know the loader accepts them.
    const PNG_BYTES: &[u8] = &[
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44,
        0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1F,
        0x15, 0xC4, 0x89, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0x00,
        0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00, 0x00, 0x00, 0x00, 0x49,
        0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
    ];

    /// Drive an Enter on `text` then drain pending `AttachLoaded` /
    /// `AttachFailed` events so the App applies them. Mirrors what the
    /// real event loop does between `events.next()` and `chat_rx.recv`.
    async fn attach_and_drain(app: &mut App, rx: &mut mpsc::Receiver<ChatEvent>, text: &str) {
        for c in text.chars() {
            app.on_key(typed(c));
        }
        app.on_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        // First event: either AttachLoaded or AttachFailed.
        if let Some(ev) = rx.recv().await {
            app.on_llm_event(ev);
        }
    }

    #[tokio::test]
    async fn attach_png_stages_attachment_and_renders_info_line() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shot.png");
        tokio::fs::write(&path, PNG_BYTES).await.unwrap();

        let (mut app, mut rx) = test_app();
        attach_and_drain(&mut app, &mut rx, &format!("/attach {}", path.display())).await;

        assert_eq!(
            app.pending_attachments.len(),
            1,
            "one image staged after /attach"
        );
        let staged = &app.pending_attachments[0];
        assert_eq!(staged.name, "shot.png");
        assert_eq!(staged.mime, "image/png");
        let rendered = rendered_text(&mut app);
        assert!(
            rendered
                .iter()
                .any(|l| l.contains("📎 attached: shot.png") && l.contains("image/png")),
            "info line missing in: {rendered:?}"
        );
    }

    #[tokio::test]
    async fn attach_missing_path_pushes_error_and_stages_nothing() {
        let (mut app, mut rx) = test_app();
        attach_and_drain(&mut app, &mut rx, "/attach /nonexistent/missing.png").await;

        assert!(
            app.pending_attachments.is_empty(),
            "no attachment staged on error"
        );
        let rendered = rendered_text(&mut app);
        assert!(
            rendered
                .iter()
                .any(|l| l.contains("/attach") && l.contains("file not found")),
            "expected error line, got: {rendered:?}"
        );
    }

    #[tokio::test]
    async fn attach_text_file_is_rejected_as_unrecognized() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("notes.txt");
        tokio::fs::write(&path, b"not an image").await.unwrap();

        let (mut app, mut rx) = test_app();
        attach_and_drain(&mut app, &mut rx, &format!("/attach {}", path.display())).await;

        assert!(app.pending_attachments.is_empty());
        let rendered = rendered_text(&mut app);
        assert!(
            rendered
                .iter()
                .any(|l| l.contains("not a recognized image")),
            "expected unrecognized-image error: {rendered:?}"
        );
    }

    #[tokio::test]
    async fn attach_gif_is_rejected_by_format_allowlist() {
        // GIF89a header — `infer` flags as image/gif; allowlist rejects.
        const GIF_BYTES: &[u8] = &[
            0x47, 0x49, 0x46, 0x38, 0x39, 0x61, 0x01, 0x00, 0x01, 0x00, 0x80, 0x00, 0x00, 0xFF,
            0xFF, 0xFF, 0x00, 0x00, 0x00, 0x21, 0xF9, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2C,
            0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x02, 0x02, 0x44, 0x01, 0x00,
            0x3B,
        ];
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("anim.gif");
        tokio::fs::write(&path, GIF_BYTES).await.unwrap();

        let (mut app, mut rx) = test_app();
        attach_and_drain(&mut app, &mut rx, &format!("/attach {}", path.display())).await;

        assert!(app.pending_attachments.is_empty());
        // Long error messages may wrap across multiple visual lines —
        // join before searching so the test isn't sensitive to width.
        let joined = rendered_text(&mut app).join(" ");
        assert!(
            joined.contains("unsupported image format") && joined.contains("image/gif"),
            "expected unsupported-format error: {joined}"
        );
    }

    #[tokio::test]
    async fn attach_then_submit_drains_pending_and_passes_attachments() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.png");
        tokio::fs::write(&path, PNG_BYTES).await.unwrap();

        let (mut app, mut rx) = test_app();
        attach_and_drain(&mut app, &mut rx, &format!("/attach {}", path.display())).await;
        assert_eq!(app.pending_attachments.len(), 1);

        // Submitting plain text now should drain the attachment.
        for c in "hi".chars() {
            app.on_key(typed(c));
        }
        app.on_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert!(
            app.pending_attachments.is_empty(),
            "submit must drain pending_attachments"
        );
        let rendered = rendered_text(&mut app);
        assert!(
            rendered
                .iter()
                .any(|l| l.contains("> hi") && l.contains("📎 a.png")),
            "user prompt should include 📎 tag: {rendered:?}"
        );
    }

    #[tokio::test]
    async fn attach_twice_stages_both_then_drains_both() {
        let dir = tempfile::tempdir().unwrap();
        let p1 = dir.path().join("first.png");
        let p2 = dir.path().join("second.png");
        tokio::fs::write(&p1, PNG_BYTES).await.unwrap();
        tokio::fs::write(&p2, PNG_BYTES).await.unwrap();

        let (mut app, mut rx) = test_app();
        attach_and_drain(&mut app, &mut rx, &format!("/attach {}", p1.display())).await;
        attach_and_drain(&mut app, &mut rx, &format!("/attach {}", p2.display())).await;
        assert_eq!(app.pending_attachments.len(), 2);

        for c in "go".chars() {
            app.on_key(typed(c));
        }
        app.on_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert!(app.pending_attachments.is_empty());
        let rendered = rendered_text(&mut app);
        // Multi-attachment tag mentions both filenames + count.
        assert!(
            rendered
                .iter()
                .any(|l| l.contains("first.png") && l.contains("second.png")),
            "multi-attachment user tag missing names: {rendered:?}"
        );
    }

    #[tokio::test]
    async fn voice_transcription_with_slash_attach_is_treated_as_text() {
        // A transcription that happens to start with "/attach" must
        // NOT trigger file I/O — it's just spoken text. After
        // submission we expect a regular user output line and no
        // staged attachment.
        let (mut app, _rx) = test_app();
        app.on_transcription("/attach foo.png".into());
        assert!(app.pending_attachments.is_empty());
        let rendered = rendered_text(&mut app);
        assert!(
            rendered.iter().any(|l| l.contains("> /attach foo.png")),
            "voice transcription starting with /attach must render as plain user text: {rendered:?}"
        );
    }

    #[test]
    fn attach_with_no_path_pushes_error() {
        let (mut app, _rx) = test_app();
        for c in "/attach".chars() {
            app.on_key(typed(c));
        }
        app.on_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert!(app.pending_attachments.is_empty());
        let rendered = rendered_text(&mut app);
        assert!(
            rendered.iter().any(|l| l.contains("expected a path")),
            "missing usage error: {rendered:?}"
        );
    }
}
