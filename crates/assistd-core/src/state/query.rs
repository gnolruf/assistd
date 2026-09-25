//! `handle_query`: per-turn agent loop driver.

use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::Value;
use tokio::sync::{broadcast, mpsc};
use tokio::task::{JoinError, JoinHandle};
use tokio_util::sync::CancellationToken;
use tokio_util::task::AbortOnDropHandle;
use tracing::{Instrument, debug, warn};

use assistd_ipc::{Event, EventKind, ImageAttachment, StatusKind};
use assistd_llm::{LlmError, LlmEvent, LlmHealthProbe, ToolCall};
use assistd_memory::{PersistedMessage, SessionId, TurnId};
use assistd_tools::{Attachment, inherit_confirm_router};
use assistd_voice::{SentenceBuffer, SpeakDecision, VoiceOutputController};

use super::context::combine_context_blocks;
use super::wire::decode_wire_attachments;
use super::{AppState, DispatchError, RuntimeState, send_error};
use crate::Agent;
use crate::presence::{LlmStreamGuard, PresenceLlmHealthProbe, RequestGuard};
use crate::recovery::{Component, spawn_supervised};

const LAST_DELTA_DEBOUNCE: Duration = Duration::from_millis(100);
const SENTENCE_PREVIEW_CHARS: usize = 60;

struct QueryGuards {
    _request: RequestGuard,
    _stream: LlmStreamGuard,
}

/// Sentence splitter and the channel to the speech worker.
struct SpeechPipeline {
    tx: mpsc::Sender<String>,
    sentences: SentenceBuffer,
    partial_flush: Option<Duration>,
    first_sentence_emitted: bool,
}

impl SpeechPipeline {
    async fn flush_idle(&mut self) {
        if let Some(partial) = self.sentences.flush_idle()
            && self.tx.send(partial).await.is_err()
        {
            debug!(
                target: "assistd::voice",
                "speech worker channel closed; dropping idle flush"
            );
        }
    }

    async fn speak(&mut self, sentences: Vec<String>) {
        for sentence in sentences {
            if !self.first_sentence_emitted {
                debug!(
                    target: "assistd::voice::latency",
                    stage = "first_sentence_emitted",
                    "voice latency stage"
                );
                self.first_sentence_emitted = true;
            }
            if self.tx.send(sentence).await.is_err() {
                debug!(
                    target: "assistd::voice",
                    "speech worker channel closed; dropping sentence"
                );
                break;
            }
        }
    }
}

/// Translates one turn's agent events into wire events, persisting the
/// transcript as it goes.
struct TurnTranslator {
    id: String,
    turn_id: Option<TurnId>,
    assistant_accum: String,
    awaiting_tool_result: bool,
    /// Backdated at start so the first `Delta` publishes `LastDelta` at once.
    last_emit_at: Instant,
    done_emitted: bool,
}

impl TurnTranslator {
    fn new(id: String, turn_id: Option<TurnId>) -> Self {
        Self {
            id,
            turn_id,
            assistant_accum: String::new(),
            awaiting_tool_result: false,
            last_emit_at: Instant::now()
                .checked_sub(LAST_DELTA_DEBOUNCE)
                .unwrap_or_else(Instant::now),
            done_emitted: false,
        }
    }

    /// The wire event and sentences to speak for `event`, or `None` when
    /// it has no wire form.
    fn translate(
        &mut self,
        state: &AppState,
        event: LlmEvent,
        sentence_buf: &mut SentenceBuffer,
    ) -> Option<(Event, Vec<String>)> {
        let id = self.id.clone();
        let translated = match event {
            LlmEvent::Delta { text } => {
                let sentences = sentence_buf.push(&text);
                self.accumulate_delta(&state.runtime, &text);
                (Event::Delta { id, text }, sentences)
            }
            LlmEvent::ReasoningDelta { text } => (Event::ReasoningDelta { id, text }, Vec::new()),
            LlmEvent::ToolCallsRequested { calls } => {
                self.persist_tool_calls(state, &calls);
                return None;
            }
            LlmEvent::ToolCall {
                name, arguments, ..
            } => {
                self.awaiting_tool_result = true;
                (
                    Event::ToolCall {
                        id,
                        name,
                        args: arguments,
                    },
                    Vec::new(),
                )
            }
            LlmEvent::ToolResult {
                id: call_id,
                name,
                result,
            } => {
                self.awaiting_tool_result = false;
                state.persist_message_fire_and_forget(
                    self.turn_id,
                    PersistedMessage::tool_result(tool_result_body(&result), call_id, name.clone()),
                );
                (Event::ToolResult { id, name, result }, Vec::new())
            }
            LlmEvent::Status {
                severity,
                component,
                event,
                message,
            } => {
                if matches!(event, StatusKind::Restarting) {
                    self.assistant_accum.clear();
                    let _ = sentence_buf.finish();
                }
                (
                    Event::Status {
                        id,
                        severity,
                        component,
                        event,
                        message,
                    },
                    Vec::new(),
                )
            }
            LlmEvent::Done => {
                let sentences = sentence_buf.finish().into_iter().collect();
                self.persist_final_text(state);
                self.done_emitted = true;
                (Event::Done { id }, sentences)
            }
        };
        Some(translated)
    }

    fn accumulate_delta(&mut self, runtime: &RuntimeState, text: &str) {
        self.assistant_accum.push_str(text);
        if self.last_emit_at.elapsed() >= LAST_DELTA_DEBOUNCE {
            publish_last_delta(runtime, &self.id, &self.assistant_accum);
            self.last_emit_at = Instant::now();
        }
    }

    fn persist_tool_calls(&mut self, state: &AppState, calls: &[ToolCall]) {
        let narration = std::mem::take(&mut self.assistant_accum);
        if !narration.is_empty() {
            publish_last_delta(&state.runtime, &self.id, &narration);
        }
        state.persist_message_fire_and_forget(self.turn_id, tool_calls_message(narration, calls));
    }

    fn persist_final_text(&mut self, state: &AppState) {
        if self.assistant_accum.is_empty() {
            return;
        }
        let final_text = std::mem::take(&mut self.assistant_accum);
        publish_last_delta(&state.runtime, &self.id, &final_text);
        state.persist_message_fire_and_forget(
            self.turn_id,
            PersistedMessage::assistant_text(final_text),
        );
    }
}

enum NextLlmEvent {
    Event(LlmEvent),
    Idle,
    Closed,
}

impl AppState {
    /// Handle a single user query: wake the daemon if needed, run the
    /// agent loop, stream events through `tx`, and persist the turn.
    pub async fn handle_query(
        self: Arc<Self>,
        id: String,
        text: String,
        wire_attachments: Vec<ImageAttachment>,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        let attachments = self
            .prepare_attachments(&id, &wire_attachments, &tx)
            .await?;
        let _session_guards = self.acquire_query_guards(&id, &tx).await?;
        let _agent_guard = self.runtime.agent_turn_lock.clone().lock_owned().await;
        if let Some(revalidator) = &self.subsystems.vision_revalidator {
            revalidator
                .revalidate_if_stale(&self.subsystems.presence)
                .await;
        }

        let (current_session, turn_id) = self.open_persistence_turn(&text).await;
        self.assemble_transient_context(&text).await;

        let cancel = CancellationToken::new();
        let _cancel_on_return = cancel.clone().drop_guard();
        *self.runtime.current_cancel.lock().await = Some(cancel.clone());

        let title_user_text = text.clone();
        let (llm_tx, llm_rx) = mpsc::channel::<LlmEvent>(32);
        let agent_task = self.spawn_agent_task(text, attachments, llm_tx, cancel.clone());

        let (speech, speech_rx) = self.speech_pipeline();
        self.subsystems.voice_output.skip().await;
        let start_epoch = self.subsystems.voice_output.current_epoch();
        let speech_handle = self.spawn_speech_worker(id.clone(), start_epoch, speech_rx);

        let done_emitted = self
            .drive_event_loop(id.clone(), llm_rx, &tx, speech, turn_id)
            .await;

        let agent_result = agent_task.await;
        *self.runtime.current_cancel.lock().await = None;
        drop(_agent_guard);

        if done_emitted && matches!(&agent_result, Ok(Ok(()))) {
            self.clone().spawn_session_title_generation(
                id.clone(),
                current_session,
                title_user_text,
            );
        }

        self.finalize_turn(id, turn_id, agent_result, speech_handle, &tx, done_emitted)
            .await
    }

    async fn prepare_attachments(
        &self,
        id: &str,
        wire: &[ImageAttachment],
        tx: &mpsc::Sender<Event>,
    ) -> Result<Vec<Attachment>, DispatchError> {
        let decoded = decode_wire_attachments(wire);
        if let Err(e) = &decoded {
            send_error(tx, id.to_string(), e.to_string()).await;
        }
        decoded
    }

    async fn acquire_query_guards(
        &self,
        id: &str,
        tx: &mpsc::Sender<Event>,
    ) -> Result<QueryGuards, DispatchError> {
        let request = match self
            .subsystems
            .presence
            .acquire_request_guard_with_progress(id.to_string(), tx.clone())
            .await
        {
            Ok(guard) => guard,
            Err(e) => {
                send_error(tx, id.to_string(), format!("wake failed: {e}")).await;
                return Err(e.into());
            }
        };
        let stream = self.subsystems.presence.acquire_stream_guard();
        Ok(QueryGuards {
            _request: request,
            _stream: stream,
        })
    }

    async fn open_persistence_turn(&self, text: &str) -> (Arc<SessionId>, Option<TurnId>) {
        let (current_session, _current_branch) = self.runtime.conversation_ctx.current().await;
        let turn_id: Option<TurnId> = match self
            .memory
            .conversations
            .begin_turn(&current_session, text)
            .await
        {
            Ok(turn) if turn.0 != 0 => Some(turn),
            Ok(_) => None,
            Err(e) => {
                warn!(
                    target: "assistd::memory",
                    error = %e,
                    "begin_turn failed; turn will not be persisted"
                );
                None
            }
        };
        self.persist_message_fire_and_forget(turn_id, PersistedMessage::user(text.to_string()));
        (current_session, turn_id)
    }

    async fn assemble_transient_context(&self, text: &str) {
        let semantic = if self.memory.embedding_cfg.enabled && self.memory.embedding_cfg.auto_inject
        {
            match self.build_semantic_context(text).await {
                Ok(block) => block,
                Err(e) => {
                    debug!(
                        target: "assistd::embed",
                        error = %e,
                        "semantic context injection failed; continuing without it",
                    );
                    None
                }
            }
        } else {
            None
        };
        let window = self.build_window_context().await;
        if let Some(block) = combine_context_blocks(semantic, window)
            && let Err(e) = self.subsystems.llm.set_transient_context(block).await
        {
            debug!(
                target: "assistd::context",
                error = %e,
                "set_transient_context failed; continuing without it",
            );
        }
    }

    fn speech_pipeline(&self) -> (SpeechPipeline, mpsc::Receiver<String>) {
        let synthesis = &self.config.voice.synthesis;
        let sentences = SentenceBuffer::new_with_mode(
            synthesis.max_sentence_chars.get() as usize,
            synthesis.code_block_mode,
        );
        let partial_flush = (synthesis.partial_flush_ms > 0)
            .then(|| Duration::from_millis(synthesis.partial_flush_ms as u64));
        let (tx, rx) = mpsc::channel::<String>(32);
        (
            SpeechPipeline {
                tx,
                sentences,
                partial_flush,
                first_sentence_emitted: false,
            },
            rx,
        )
    }

    fn spawn_agent_task(
        &self,
        text: String,
        attachments: Vec<Attachment>,
        llm_tx: mpsc::Sender<LlmEvent>,
        cancel: CancellationToken,
    ) -> AbortOnDropHandle<Result<(), LlmError>> {
        let llm = self.subsystems.llm.clone();
        let tools = self.subsystems.tools.clone();
        let health: Option<Arc<dyn LlmHealthProbe>> = Some(Arc::new(PresenceLlmHealthProbe::new(
            self.subsystems.presence.clone(),
        )));
        let agent = Agent::new(
            llm,
            tools,
            health,
            Duration::from_secs(self.config.timeouts.tool_call_secs),
        );
        AbortOnDropHandle::new(tokio::spawn(
            inherit_confirm_router(async move {
                agent.run_turn(text, attachments, llm_tx, cancel).await
            })
            .in_current_span(),
        ))
    }

    fn spawn_speech_worker(
        &self,
        id: String,
        start_epoch: u64,
        speech_rx: mpsc::Receiver<String>,
    ) -> JoinHandle<()> {
        spawn_supervised(
            "speech_worker",
            Component::Voice,
            run_speech_worker(
                self.subsystems.voice_output.clone(),
                self.runtime.events_bus().clone(),
                id,
                start_epoch,
                speech_rx,
            )
            .in_current_span(),
        )
    }

    /// Forward the turn's events to `tx` and the speech worker until the
    /// agent finishes or the client leaves. Returns whether `Done` was sent.
    async fn drive_event_loop(
        &self,
        id: String,
        mut llm_rx: mpsc::Receiver<LlmEvent>,
        tx: &mpsc::Sender<Event>,
        mut speech: SpeechPipeline,
        turn_id: Option<TurnId>,
    ) -> bool {
        let mut translator = TurnTranslator::new(id, turn_id);
        loop {
            let idle_timeout = speech
                .partial_flush
                .filter(|_| !translator.awaiting_tool_result);
            let llm_event = match next_llm_event(&mut llm_rx, idle_timeout).await {
                NextLlmEvent::Event(event) => event,
                NextLlmEvent::Closed => break,
                NextLlmEvent::Idle => {
                    speech.flush_idle().await;
                    continue;
                }
            };
            let Some((wire_event, sentences)) =
                translator.translate(self, llm_event, &mut speech.sentences)
            else {
                continue;
            };
            let client_alive = tx.send(wire_event).await.is_ok();
            speech.speak(sentences).await;
            if !client_alive {
                break;
            }
        }
        translator.done_emitted
    }

    /// End the persisted turn, wait for speech, and report the agent's
    /// outcome. Sends `Done` if the stream did not, as a cancelled turn
    /// ends without one.
    async fn finalize_turn(
        &self,
        id: String,
        turn_id: Option<TurnId>,
        agent_result: Result<Result<(), LlmError>, JoinError>,
        speech_handle: JoinHandle<()>,
        tx: &mpsc::Sender<Event>,
        done_emitted: bool,
    ) -> Result<(), DispatchError> {
        if let Some(turn) = turn_id {
            let conversations = self.memory.conversations.clone();
            self.runtime.persistence_tracker.spawn(async move {
                if let Err(e) = conversations.end_turn(turn).await {
                    warn!(
                        target: "assistd::memory",
                        error = %e,
                        "end_turn failed (continuing)"
                    );
                }
            });
        }

        let _ = speech_handle.await;

        match agent_result {
            Ok(Ok(())) => {
                if !done_emitted {
                    let _ = tx.send(Event::Done { id }).await;
                }
                Ok(())
            }
            Ok(Err(e)) => {
                send_error(tx, id, format!("llm backend error: {e}")).await;
                Err(e.into())
            }
            Err(join_err) => {
                let e = DispatchError::AgentPanicked(join_err);
                send_error(tx, id, e.to_string()).await;
                Err(e)
            }
        }
    }
}

async fn next_llm_event(
    llm_rx: &mut mpsc::Receiver<LlmEvent>,
    idle_timeout: Option<Duration>,
) -> NextLlmEvent {
    let received = match idle_timeout {
        Some(timeout) => match tokio::time::timeout(timeout, llm_rx.recv()).await {
            Ok(received) => received,
            Err(_) => return NextLlmEvent::Idle,
        },
        None => llm_rx.recv().await,
    };
    received.map_or(NextLlmEvent::Closed, NextLlmEvent::Event)
}

async fn run_speech_worker(
    voice_output: Arc<VoiceOutputController>,
    events_bus: broadcast::Sender<Event>,
    id: String,
    start_epoch: u64,
    mut speech_rx: mpsc::Receiver<String>,
) {
    let mut emitted_start = false;
    while let Some(sentence) = speech_rx.recv().await {
        match voice_output.should_speak(start_epoch) {
            SpeakDecision::Speak => {
                if !emitted_start {
                    let _ = events_bus.send(Event::SpeakingState {
                        id: id.clone(),
                        speaking: true,
                    });
                    emitted_start = true;
                }
                speak_sentence(&voice_output, sentence, start_epoch).await;
            }
            SpeakDecision::DropSilent | SpeakDecision::DropForSkip => {}
        }
    }
    if let Err(e) = voice_output.inner().wait_idle().await {
        debug!(
            target: "assistd::voice",
            error = %e,
            "voice_output.wait_idle failed (non-fatal)"
        );
    }
    if emitted_start {
        let _ = events_bus.send(Event::SpeakingState {
            id,
            speaking: false,
        });
    }
}

/// Speak one sentence, then cancel playback if a skip landed meanwhile,
/// since `speak()` may append audio after the skip cleared the queue.
async fn speak_sentence(voice_output: &VoiceOutputController, sentence: String, start_epoch: u64) {
    let preview: String = sentence.chars().take(SENTENCE_PREVIEW_CHARS).collect();
    if let Err(e) = voice_output.inner().speak(sentence).await {
        warn!(
            target: "assistd::voice",
            error = %e,
            sentence_preview = %preview,
            "voice_output.speak failed; sentence dropped"
        );
    }
    if matches!(
        voice_output.should_speak(start_epoch),
        SpeakDecision::DropForSkip
    ) {
        voice_output.inner().cancel().await;
    }
}

fn tool_calls_message(narration: String, calls: &[ToolCall]) -> PersistedMessage {
    let content = if narration.trim().is_empty() {
        String::new()
    } else {
        narration
    };
    let calls_json = calls
        .iter()
        .map(|call| serde_json::json!({"id": call.id, "name": call.name, "arguments": call.arguments}))
        .collect();
    PersistedMessage::assistant_tool_calls(content, calls_json)
}

fn tool_result_body(result: &Value) -> String {
    result
        .get("output")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| result.to_string())
}

/// `LastDelta` carries the whole reply so far, so building one costs a
/// copy of it; skip that when no subscriber would receive it.
fn publish_last_delta(runtime: &RuntimeState, id: &str, text: &str) {
    if runtime.bus_wants(EventKind::LastDelta) {
        let _ = runtime.events_bus().send(Event::LastDelta {
            id: id.to_string(),
            text: text.to_string(),
        });
    }
}
