//! Reducer for daemon events. Several streams feed it at once, so reply
//! events are gated on the turn that owns the output pane.

use std::time::{Duration, Instant};

use assistd_ipc::{
    Event, EventBody, ListenPhase, PresenceState, RelayEvent, RelayStepRecord, Role,
    StartupComponent, StatusKind, StatusSeverity, ToolReport, TurnStats, VoiceCaptureState,
};
use assistd_tools::{ConfirmationRequest, Risk, Unattended};
use assistd_utils::time::saturating_millis;
use serde_json::Value;

use super::super::output::tool_block_label;
use super::super::ui::turn_summary;
use super::{
    ActiveReply, AnswerTo, App, BranchListEntry, BranchOp, ChatEvent, ChatMode, OWN_TURN_MEMORY,
    QueuedTranscription, WakeWordStream, WireStream,
};

impl App {
    pub(crate) fn on_chat_event(&mut self, ev: ChatEvent) {
        match ev {
            ChatEvent::Wire { stream, event } => self.on_wire_event(stream, event),
            ChatEvent::WireError {
                stream: WireStream::Tasks,
                message,
            } => self.fail_tasks_stream(&message),
            ChatEvent::WireError { stream, message } => {
                self.output.push_error(&format!("[wire error] {message}"));
                self.fail_stream(stream);
            }
            ChatEvent::Bus(event) => self.on_bus_event(event),
            ChatEvent::BusLost => self.on_bus_lost(),
            ChatEvent::BusRestored => self.on_bus_restored(),
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

    /// End the turn shown from the bus, whose `Done` can no longer arrive; a reply
    /// this chat's own query stream carries is left to that stream.
    fn on_bus_lost(&mut self) {
        self.on_daemon_unreachable();
        self.queued_transcription = None;
        if self
            .active_reply
            .as_ref()
            .is_some_and(|reply| reply.writer.is_none())
        {
            self.finish_reply();
        }
    }

    /// Replay the daemon's transcript to cover what was missed while the bus was down.
    fn on_bus_restored(&mut self) {
        self.on_daemon_reachable();
        if !self.generating() && self.in_flight_branch_op.is_none() {
            self.spawn_resume_or_new();
        }
    }

    /// Re-read the capabilities on reconnect, since the daemon may have restarted
    /// with another model or none was known at startup.
    fn on_daemon_reachable(&mut self) {
        if !std::mem::replace(&mut self.daemon_reachable, true) {
            self.spawn_capabilities_refresh();
        }
    }

    /// Drop the indicator state the daemon last reported, which may no longer hold.
    fn on_daemon_unreachable(&mut self) {
        self.daemon_reachable = false;
        self.presence_state = None;
        self.listening = VoiceCaptureState::Idle;
        self.wake_word_heard = false;
    }

    /// Route a bus event, drawing turns this chat did not start like its own replies.
    /// A `Done` or `Error` counts only when it closes the turn on screen, since every
    /// client's one-shot requests end with one.
    fn on_bus_event(&mut self, ev: Event) {
        if matches!(ev.body, EventBody::Transcription { .. }) {
            self.wake_word_heard = false;
        }
        if matches!(ev.body, EventBody::SessionTitle { .. }) {
            self.on_wire_event(WireStream::Status, ev);
            return;
        }
        if let EventBody::TaskQueue { snapshot } = ev.body {
            self.on_task_queue(snapshot);
            return;
        }
        if let EventBody::Readiness {
            component: StartupComponent::Mcp { server },
            state,
        } = ev.body
        {
            self.on_mcp_readiness(server, state);
            return;
        }
        let id = ev.id.as_str();
        if self.own_turn_ids.iter().any(|own| own == id) {
            return;
        }
        let closes_shown_turn = self.active_reply.as_ref().is_some_and(|r| r.id == id)
            || self
                .queued_transcription
                .as_ref()
                .is_some_and(|q| q.id == id);
        if matches!(ev.body, EventBody::Done | EventBody::Error { .. }) && !closes_shown_turn {
            return;
        }
        self.on_wire_event(WireStream::Reply, ev);
    }

    fn on_wire_event(&mut self, stream: WireStream, ev: Event) {
        match stream {
            WireStream::WakeWord(wake_word_stream) => {
                self.on_wake_word_event(wake_word_stream, ev);
                return;
            }
            WireStream::Tasks => {
                self.on_tasks_event(ev);
                return;
            }
            WireStream::Reply | WireStream::Branch | WireStream::Status => {}
        }
        if matches!(stream, WireStream::Reply) && !self.accept_reply_event(&ev) {
            return;
        }
        let now = Instant::now();
        match ev.body {
            EventBody::Done => self.finish_stream(stream, now),
            EventBody::Error { message } => {
                self.output.push_error(&message);
                self.fail_stream(stream);
            }
            turn @ (EventBody::Delta { .. }
            | EventBody::ReasoningDelta { .. }
            | EventBody::ToolCall { .. }
            | EventBody::ToolResult { .. }
            | EventBody::Relay { .. }
            | EventBody::ConfirmRequest { .. }
            | EventBody::PlanReview { .. }
            | EventBody::PlanModeEntered
            | EventBody::Transcription { .. }) => self.on_turn_event(turn),
            state @ (EventBody::Presence { .. }
            | EventBody::VoiceState { .. }
            | EventBody::ListenState { .. }
            | EventBody::WakeWordDetected
            | EventBody::VoiceOutputState { .. }
            | EventBody::Capabilities { .. }
            | EventBody::Status { .. }) => self.on_state_event(state),
            EventBody::ContextSize { tokens, total } => {
                self.tokens.on_context_size(tokens, total);
            }
            session @ (EventBody::BranchInfo { .. }
            | EventBody::BranchSwitched { .. }
            | EventBody::SessionTitle { .. }
            | EventBody::HistoryEntry { .. }
            | EventBody::UndoApplied { .. }) => self.on_session_event(session),
            EventBody::SpeakingState { .. }
            | EventBody::Readiness { .. }
            | EventBody::ChatFocus { .. }
            | EventBody::SemanticHit { .. }
            | EventBody::MemoryValue { .. }
            | EventBody::MemoryRow { .. }
            | EventBody::MemoryForgetResult { .. }
            | EventBody::ReindexProgress { .. }
            | EventBody::WakeWordStatus { .. }
            | EventBody::WakeWordTest { .. }
            | EventBody::WakeWordSpeech { .. }
            | EventBody::Task { .. }
            | EventBody::TaskRunRecord { .. }
            | EventBody::TaskRunQueued { .. }
            | EventBody::TaskRun { .. }
            | EventBody::TaskQueue { .. }
            | EventBody::TaskReply { .. }
            | EventBody::Queued { .. }
            | EventBody::TaskRefined { .. }
            | EventBody::TaskPlan { .. }
            | EventBody::TriggerFires { .. }
            | EventBody::McpAuthorizeUrl { .. }
            | EventBody::McpServer { .. }
            | EventBody::McpLoggedOut { .. }
            | EventBody::LastDelta { .. } => {}
        }
    }

    /// Events that build the current reply in the output pane.
    fn on_turn_event(&mut self, body: EventBody) {
        match body {
            EventBody::Delta { text } => {
                self.output.finish_thinking();
                self.tokens.on_delta();
                self.output.append_assistant(&text);
            }
            EventBody::ReasoningDelta { text } => {
                self.tokens.on_delta();
                self.output.append_thinking(&text);
            }
            EventBody::ToolCall { args, name } => self.on_tool_call(name, &args),
            EventBody::ToolResult { result, .. } => self.on_tool_result(result),
            EventBody::Relay { event } => self.on_relay_event(event),
            EventBody::ConfirmRequest {
                confirm_id,
                tool,
                script,
                matched_pattern,
                always_allow,
                source,
            } => self.open_confirmation(
                confirm_id,
                ConfirmationRequest {
                    tool,
                    script,
                    matched_pattern,
                    always_allow,
                    risk: Risk::Elevated,
                    unattended: Unattended::Never,
                    source,
                },
            ),
            EventBody::PlanReview { review_id, plan } => self.open_plan_review(review_id, &plan),
            EventBody::PlanModeEntered => self.mode = ChatMode::Plan,
            EventBody::Transcription { text } if text.trim().is_empty() => {
                self.set_notice("no speech detected");
            }
            EventBody::Transcription { text } => self.begin_turn(&text, &[]),
            _ => {}
        }
    }

    fn on_tool_call(&mut self, name: String, args: &Value) {
        self.output.finish_thinking();
        self.output
            .begin_tool_block(tool_block_label(name, Some(args)));
    }

    fn on_tool_result(&mut self, result: ToolReport) {
        self.output
            .finish_tool_block(result.output, result.exit_code, result.duration_ms);
    }

    /// Draw a relay's progress inside the running `relay` tool block.
    fn on_relay_event(&mut self, event: RelayEvent) {
        match event {
            RelayEvent::Planned { steps } => self.output.begin_relay(steps),
            event => self.output.apply_relay(event),
        }
    }

    /// Daemon and model state the chat tracks between turns.
    fn on_state_event(&mut self, body: EventBody) {
        match body {
            EventBody::Presence { state } => self.on_presence(state),
            EventBody::VoiceState { state } => self.listening = state,
            EventBody::ListenState { phase } => {
                self.wake_word_heard &= phase == ListenPhase::Capturing;
            }
            EventBody::WakeWordDetected => self.wake_word_heard = true,
            EventBody::VoiceOutputState { enabled } => self.voice_output_enabled = enabled,
            EventBody::Capabilities {
                vision,
                model_name,
                auto_mode,
            } => {
                self.vision_enabled = vision;
                self.offer_auto_mode(auto_mode);
                if !model_name.is_empty() {
                    self.output.set_banner_model(&model_name);
                    self.model_name = model_name;
                }
            }
            EventBody::Status {
                event: StatusKind::CompactingSoon,
                ..
            } => self.compacting_soon = true,
            EventBody::Status {
                event: StatusKind::HistoryCompacted,
                ..
            } => self.output.finish_compaction(),
            EventBody::Status {
                severity,
                component,
                event,
                message,
            } => {
                self.set_notice(&message);
                match event {
                    StatusKind::Restarting => {
                        self.close_discarded_step(&format!("[{component} restarting…]"));
                    }
                    StatusKind::CompactingHistory => {
                        self.compacting_soon = false;
                        self.output.begin_compaction();
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
    fn on_session_event(&mut self, body: EventBody) {
        match body {
            EventBody::BranchInfo {
                name,
                parent_branch_name,
                fork_point_seq,
                message_count,
                is_current_in_session,
                is_active_session,
                session_id,
                session_title,
                last_active_at,
                task_run,
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
                last_active_at: last_active_at.parse().ok(),
                task_run,
            }),
            EventBody::BranchSwitched {
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
            EventBody::SessionTitle { title, .. } => self.session_title = Some(title),
            EventBody::HistoryEntry {
                role,
                content,
                reasoning,
                reasoning_ms,
                tool_name,
                tool_args,
                exit_code,
                duration_ms,
                compaction_ms,
                turn_stats,
                relay,
                ..
            } => {
                if let Some(ms) = reasoning_ms
                    && !reasoning.is_empty()
                {
                    self.output
                        .push_replayed_thinking(reasoning, Duration::from_millis(ms));
                }
                let outcome = (exit_code, duration_ms);
                let tool = (tool_name, tool_args.as_ref());
                self.on_history_entry(role, content, tool, outcome, relay);
                if let Some(ms) = compaction_ms {
                    self.output
                        .push_replayed_compaction(Duration::from_millis(ms));
                }
                if let Some(stats) = turn_stats {
                    self.output.push_turn_summary(turn_summary(&stats));
                }
            }
            EventBody::UndoApplied {
                removed_messages,
                last_user_text,
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
            Some(BranchOp::Switch | BranchOp::Resume | BranchOp::New) => self.reset_transcript(),
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

    /// `tool` is a tool row's name and arguments, `outcome` its exit code and
    /// duration, and `relay` the sub-agent steps behind a `relay` row.
    fn on_history_entry(
        &mut self,
        role: Role,
        content: String,
        (tool_name, tool_args): (Option<String>, Option<&Value>),
        outcome: (Option<i64>, Option<u64>),
        relay: Vec<RelayStepRecord>,
    ) {
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
                let label = tool_block_label(tool_name.unwrap_or_default(), tool_args);
                let exit_code = outcome
                    .0
                    .and_then(|code| i32::try_from(code).ok())
                    .unwrap_or(0);
                let duration_ms = outcome.1.unwrap_or(0);
                if relay.is_empty() {
                    self.output
                        .push_tool_block(label, content, exit_code, duration_ms);
                } else {
                    self.output
                        .push_replayed_relay(label, content, exit_code, duration_ms, relay);
                }
            }
            Role::System => self.output.push_info(&content),
        }
    }

    fn on_undo_applied(&mut self, removed_messages: u32, last_user_text: Option<&str>) {
        if removed_messages == 0 {
            self.set_notice("nothing to undo");
            return;
        }
        self.close_agent_panel();
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
        match &ev.body {
            EventBody::Transcription { text } => {
                self.queued_transcription = Some(QueuedTranscription {
                    id: id.to_string(),
                    text: text.clone(),
                });
            }
            EventBody::Error { message } => {
                self.set_notice(message);
                self.drop_queued_transcription(id);
            }
            EventBody::Done => self.drop_queued_transcription(id),
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
        self.tokens.reset();
        self.reply_started_at = Some(Instant::now());
    }

    fn finish_reply(&mut self) {
        self.output.finish_thinking();
        self.output.finish_compaction();
        self.output.abandon_running_tool();
        self.output.finish_assistant();
        self.reply_started_at = None;
        self.compacting_soon = false;
        self.active_reply = None;
        self.plan_review = None;
        self.confirmation
            .take_if(|prompt| prompt.answer_to == AnswerTo::Reply);
        self.last_live_seconds = None;
    }

    fn finish_reply_with_summary(&mut self, now: Instant) {
        let started_at = self.reply_started_at;
        self.finish_reply();
        if let Some(started_at) = started_at {
            let elapsed = now.duration_since(started_at);
            let stats = TurnStats {
                duration_ms: saturating_millis(elapsed),
                tokens: self.tokens.tokens(),
            };
            self.output.push_turn_summary(turn_summary(&stats));
        }
    }

    fn finish_stream(&mut self, stream: WireStream, now: Instant) {
        match stream {
            WireStream::Reply => self.finish_reply_with_summary(now),
            WireStream::Branch => {
                if let Some(BranchOp::ResumePicker) = self.in_flight_branch_op.take() {
                    self.open_resume_picker();
                }
            }
            WireStream::Status | WireStream::WakeWord(_) | WireStream::Tasks => {}
        }
    }

    fn fail_stream(&mut self, stream: WireStream) {
        match stream {
            WireStream::Reply => self.finish_reply(),
            WireStream::Branch => {
                self.in_flight_branch_op = None;
                self.branches_buffer.clear();
            }
            WireStream::WakeWord(WakeWordStream::Session) => self.fail_wake_word_session(),
            WireStream::Status | WireStream::WakeWord(_) | WireStream::Tasks => {}
        }
    }
}

/// The turn id of an event that belongs to one reply turn; `None` for
/// events any stream may carry.
fn turn_scoped_id(ev: &Event) -> Option<&str> {
    match ev {
        Event {
            id,
            body:
                EventBody::Delta { .. }
                | EventBody::ReasoningDelta { .. }
                | EventBody::ToolCall { .. }
                | EventBody::ToolResult { .. }
                | EventBody::Relay { .. }
                | EventBody::ConfirmRequest { .. }
                | EventBody::PlanReview { .. }
                | EventBody::PlanModeEntered
                | EventBody::Done
                | EventBody::Error { .. },
        } => Some(id),
        Event {
            id,
            body: EventBody::Transcription { text },
        } => (!text.trim().is_empty()).then_some(id.as_str()),
        _ => None,
    }
}
