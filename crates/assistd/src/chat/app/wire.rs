//! Reducer for daemon events. Several streams feed it at once, so reply
//! events are gated on the turn that owns the output pane.

use std::time::Instant;

use assistd_ipc::{ComponentReadiness, Event, PresenceState, Role, StatusKind, StatusSeverity};
use assistd_tools::ConfirmationRequest;
use serde_json::Value;

use super::{
    ActiveReply, App, BranchListEntry, BranchOp, ChatEvent, OWN_TURN_MEMORY, QueuedTranscription,
    WireStream,
};

impl App {
    pub(crate) fn on_chat_event(&mut self, ev: ChatEvent) {
        match ev {
            ChatEvent::Wire { stream, event } => self.on_wire_event(stream, event),
            ChatEvent::WireError { stream, message } => {
                self.output.push_error(&format!("[wire error] {message}"));
                self.fail_stream(stream, Instant::now());
            }
            ChatEvent::Bus(event) => self.on_bus_event(event),
            ChatEvent::AttachLoaded(payload) => self.on_attach_loaded(*payload),
            ChatEvent::AttachFailed { path, message } => self.on_attach_failed(&path, &message),
        }
    }

    /// Record a query id so the bus copy of its events is ignored.
    pub(super) fn remember_own_turn(&mut self, id: &str) {
        if self.own_turn_ids.len() == OWN_TURN_MEMORY {
            self.own_turn_ids.pop_front();
        }
        self.own_turn_ids.push_back(id.to_string());
    }

    /// Route a bus event: titles go to the status stream, and turns this
    /// chat did not start are drawn like its own replies. A `Done` or
    /// `Error` counts only when it closes the turn on screen, since every
    /// client's one-shot requests end with one.
    fn on_bus_event(&mut self, ev: Event) {
        if matches!(ev, Event::SessionTitle { .. }) {
            self.on_wire_event(WireStream::Status, ev);
            return;
        }
        let id = ev.id();
        if self.own_turn_ids.iter().any(|own| own == id) {
            return;
        }
        let closes_shown_turn = self.active_reply.as_ref().is_some_and(|r| r.id == id)
            || self
                .queued_transcription
                .as_ref()
                .is_some_and(|q| q.id == id);
        if matches!(ev, Event::Done { .. } | Event::Error { .. }) && !closes_shown_turn {
            return;
        }
        self.on_wire_event(WireStream::Reply, ev);
    }

    fn on_wire_event(&mut self, stream: WireStream, ev: Event) {
        if matches!(stream, WireStream::Reply) && !self.accept_reply_event(&ev) {
            return;
        }
        let now = Instant::now();
        match ev {
            Event::Done { .. } => self.finish_stream(stream, now),
            Event::Error { message, .. } => {
                self.output.push_error(&message);
                self.fail_stream(stream, now);
            }
            turn @ (Event::Delta { .. }
            | Event::ReasoningDelta { .. }
            | Event::ToolCall { .. }
            | Event::ToolResult { .. }
            | Event::ConfirmRequest { .. }
            | Event::Transcription { .. }) => self.on_turn_event(turn, now),
            state @ (Event::Presence { .. }
            | Event::VoiceState { .. }
            | Event::ListenState { .. }
            | Event::VoiceOutputState { .. }
            | Event::VoiceReadiness { .. }
            | Event::Capabilities { .. }
            | Event::Status { .. }) => self.on_state_event(state),
            session @ (Event::BranchInfo { .. }
            | Event::BranchSwitched { .. }
            | Event::SessionTitle { .. }
            | Event::HistoryEntry { .. }
            | Event::UndoApplied { .. }) => self.on_session_event(session),
            Event::SpeakingState { .. }
            | Event::ChatFocus { .. }
            | Event::SemanticHit { .. }
            | Event::MemoryValue { .. }
            | Event::MemoryKeys { .. }
            | Event::MemoryRow { .. }
            | Event::MemoryForgetResult { .. }
            | Event::ReindexProgress { .. }
            | Event::LastDelta { .. } => {}
        }
    }

    /// Events that build the current reply in the output pane.
    fn on_turn_event(&mut self, ev: Event, now: Instant) {
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
            Event::ToolCall { args, name, .. } => self.on_tool_call(name, &args),
            Event::ToolResult { result, .. } => self.on_tool_result(&result),
            Event::ConfirmRequest {
                confirm_id,
                tool,
                script,
                matched_pattern,
                always_allow,
                ..
            } => self.open_confirmation_modal(
                confirm_id,
                ConfirmationRequest {
                    tool,
                    script,
                    matched_pattern,
                    always_allow,
                },
            ),
            Event::Transcription { text, .. } if text.trim().is_empty() => {
                self.set_notice("no speech detected");
            }
            Event::Transcription { text, .. } => self.begin_turn(&text, &[]),
            _ => {}
        }
    }

    fn on_tool_call(&mut self, name: String, args: &Value) {
        self.output.finish_thinking();
        let command = args
            .get("command")
            .and_then(Value::as_str)
            .map_or(name, str::to_string);
        self.output.begin_tool_block(command);
    }

    fn on_tool_result(&mut self, result: &Value) {
        let output = result
            .get("output")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let exit_code = result
            .get("exit_code")
            .and_then(Value::as_i64)
            .and_then(|code| i32::try_from(code).ok())
            .unwrap_or(0);
        let duration_ms = result
            .get("duration_ms")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        self.output
            .finish_tool_block(output, exit_code, duration_ms);
    }

    /// Daemon and model state shown in the status bar.
    fn on_state_event(&mut self, ev: Event) {
        match ev {
            Event::Presence { state, .. } => self.on_presence(state),
            Event::VoiceState { state, .. } => self.listening = state,
            Event::ListenState { active, .. } => self.listen_active = active,
            Event::VoiceOutputState { enabled, .. } => self.voice_output_enabled = enabled,
            Event::VoiceReadiness {
                capture, speech, ..
            } => {
                self.voice_starting = [capture, speech].contains(&ComponentReadiness::Starting);
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
                match event {
                    StatusKind::Restarting => {
                        self.close_discarded_step(&format!("[{component} restarting…]"));
                    }
                    StatusKind::CompactingHistory => {
                        self.output.push_info("[compacting history…]");
                    }
                    StatusKind::OutputTruncated => {
                        self.close_discarded_step(&format!(
                            "[{component} response hit the token limit; retrying…]"
                        ));
                    }
                    _ if severity == StatusSeverity::Error => {
                        self.output.push_info(&format!("[{component}: {message}]"));
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    /// Re-read capabilities when the model comes up, since vision support
    /// is only known once it has loaded.
    fn on_presence(&mut self, state: PresenceState) {
        let came_up = state == PresenceState::Active
            && self
                .presence_state
                .is_some_and(|prior| prior != PresenceState::Active);
        self.presence_state = Some(state);
        if came_up {
            self.spawn_capabilities_refresh();
        }
    }

    /// Close the streamed blocks of a step the daemon discarded, then say why.
    fn close_discarded_step(&mut self, info: &str) {
        self.output.finish_thinking();
        self.output.finish_assistant();
        self.output.push_info(info);
    }

    /// Branch and session responses.
    fn on_session_event(&mut self, ev: Event) {
        match ev {
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
            } => self.branches_buffer.push(BranchListEntry {
                name,
                parent_branch_name,
                fork_point_seq,
                message_count,
                is_current_in_session,
                is_active_session,
                session_short: session_id.chars().take(8).collect(),
                session_title,
            }),
            Event::BranchSwitched {
                name,
                parent_branch_name,
                fork_point_seq,
                session_title,
                ..
            } => self.on_branch_switched(
                &name,
                parent_branch_name.as_deref(),
                fork_point_seq,
                session_title.as_deref(),
            ),
            Event::SessionTitle { title, .. } => self.session_title = Some(title),
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
            } => self.on_undo_applied(removed_messages, last_user_text.as_deref()),
            _ => {}
        }
    }

    /// `BranchSwitched` always carries the now-active session's title, so
    /// a missing one clears it.
    fn on_branch_switched(
        &mut self,
        name: &str,
        parent_branch_name: Option<&str>,
        fork_point_seq: Option<i64>,
        session_title: Option<&str>,
    ) {
        let title = session_title.map(str::trim).filter(|t| !t.is_empty());
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
            Some(BranchOp::Resume | BranchOp::New) => {
                self.output.clear();
            }
            _ => {
                let detail = match (parent_branch_name, fork_point_seq) {
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

    fn on_undo_applied(&mut self, removed_messages: u32, last_user_text: Option<&str>) {
        if removed_messages == 0 {
            self.set_notice("nothing to undo");
            return;
        }
        self.output.pop_last_user_exchange();
        let preview = last_user_text
            .map(|t| t.chars().take(48).collect::<String>())
            .unwrap_or_default();
        if preview.is_empty() {
            self.set_notice(&format!("undid {removed_messages} message(s)"));
        } else {
            self.set_notice(&format!("undid: {preview}"));
        }
    }

    /// Whether a reply-stream event belongs to the turn owning the pane.
    /// An idle pane is claimed by the first turn-scoped event.
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
                    self.begin_turn(&queued.text, &[]);
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

    /// Echo the prompt and open an assistant block for its reply.
    pub(super) fn begin_turn(&mut self, text: &str, attachment_names: &[String]) {
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

    fn finish_reply(&mut self, now: Instant) {
        self.throughput.on_done(now);
        self.output.finish_thinking();
        self.output.abandon_running_tool();
        self.output.finish_assistant();
        self.generating = false;
        self.active_reply = None;
        self.modal = None;
        self.last_live_seconds = None;
    }

    fn finish_stream(&mut self, stream: WireStream, now: Instant) {
        match stream {
            WireStream::Reply => self.finish_reply(now),
            WireStream::Branch => {
                if let Some(BranchOp::ResumePicker) = self.in_flight_branch_op.take() {
                    self.open_branch_picker();
                }
            }
            WireStream::Status => {}
        }
    }

    fn fail_stream(&mut self, stream: WireStream, now: Instant) {
        match stream {
            WireStream::Reply => self.finish_reply(now),
            WireStream::Branch => {
                self.in_flight_branch_op = None;
                self.branches_buffer.clear();
            }
            WireStream::Status => {}
        }
    }
}

/// The turn id of an event that belongs to one reply turn; `None` for
/// events any stream may carry.
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
