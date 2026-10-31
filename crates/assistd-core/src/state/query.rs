//! `handle_query`: per-turn agent loop driver.

use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::mpsc::error::SendError;
use tokio::sync::{OwnedMutexGuard, broadcast, mpsc};
use tokio::task::{JoinError, JoinHandle};
use tokio_util::sync::CancellationToken;
use tokio_util::task::AbortOnDropHandle;
use tracing::{Instrument, debug, warn};

use assistd_ipc::{Event, EventKind, ImageAttachment, RelayEvent, StatusKind};
use assistd_llm::{CompactionCut, LlmError, LlmEvent, LlmHealthProbe, ToolCall};
use assistd_memory::{
    PersistedMessage, PersistedToolCall, Reasoning, SessionId, SummaryCut, ToolOutcome, TurnStats,
};
use assistd_tools::{Attachment, ToolOutput, inherit_confirm_router};
use assistd_utils::token_counter::TokenCounter;
use assistd_voice::{
    SentenceBuffer, SpeakDecision, SpeakingGuard, VoiceOutput, VoiceOutputController,
};

use super::context::combine_context_blocks;
use super::persistence::PersistTarget;
use super::relay_trace::RelayTraceBuilder;
use super::wire::decode_wire_attachments;
use super::{AppState, DispatchError, RuntimeState, send_error};
use crate::Agent;
use crate::presence::{PresenceLlmHealthProbe, RequestGuard};
use crate::recovery::{Component, spawn_supervised};

const LAST_DELTA_DEBOUNCE: Duration = Duration::from_millis(100);
const SENTENCE_PREVIEW_CHARS: usize = 60;

/// How a turn's prompt reached the daemon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnOrigin {
    /// Text from a client: the chat, `assistd query`, or another tool.
    Typed,
    /// A push-to-talk or continuous-listening transcription.
    Voice,
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

/// Reasoning streamed so far for the current step.
struct ReasoningTrace {
    text: String,
    started_at: Instant,
    last_delta_at: Instant,
}

impl ReasoningTrace {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            text: String::new(),
            started_at: now,
            last_delta_at: now,
        }
    }

    fn push(&mut self, delta: &str) {
        self.text.push_str(delta);
        self.last_delta_at = Instant::now();
    }

    fn into_reasoning(self) -> Reasoning {
        let elapsed = self.last_delta_at.duration_since(self.started_at);
        Reasoning {
            text: self.text,
            duration_ms: u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX),
        }
    }
}

/// Translates one turn's agent events into wire events, persisting the
/// transcript as it goes.
struct TurnTranslator {
    id: String,
    /// `None` when the session could not be saved, so nothing is persisted.
    persist: Option<PersistTarget>,
    assistant_accum: String,
    reasoning: Option<ReasoningTrace>,
    awaiting_tool_result: bool,
    /// What the relay in the running tool call has done, once it announced its plan.
    relay: Option<RelayTraceBuilder>,
    /// Backdated at start so the first `Delta` publishes `LastDelta` at once.
    last_emit_at: Instant,
    done_emitted: bool,
    started_at: Instant,
    tokens: TokenCounter,
}

impl TurnTranslator {
    fn new(id: String, persist: Option<PersistTarget>) -> Self {
        Self {
            id,
            persist,
            assistant_accum: String::new(),
            reasoning: None,
            awaiting_tool_result: false,
            relay: None,
            last_emit_at: Instant::now()
                .checked_sub(LAST_DELTA_DEBOUNCE)
                .unwrap_or_else(Instant::now),
            done_emitted: false,
            started_at: Instant::now(),
            tokens: TokenCounter::new(),
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
                self.tokens.on_delta();
                let sentences = sentence_buf.push(&text);
                self.accumulate_delta(&state.runtime, &text);
                (Event::Delta { id, text }, sentences)
            }
            LlmEvent::ReasoningDelta { text } => {
                self.tokens.on_delta();
                self.reasoning
                    .get_or_insert_with(ReasoningTrace::new)
                    .push(&text);
                (Event::ReasoningDelta { id, text }, Vec::new())
            }
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
                self.persist_tool_result(state, call_id, &name, &result);
                let result = result.to_json();
                (Event::ToolResult { id, name, result }, Vec::new())
            }
            LlmEvent::Relay(event) => {
                self.record_relay(&event);
                (Event::Relay { id, event }, Vec::new())
            }
            LlmEvent::ContextSize { tokens, total } => {
                self.tokens.on_context_size(tokens, total);
                (Event::ContextSize { id, tokens, total }, Vec::new())
            }
            LlmEvent::Status {
                severity,
                component,
                event,
                message,
            } => {
                if matches!(event, StatusKind::Restarting | StatusKind::OutputTruncated) {
                    self.discard_partial_response(sentence_buf);
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
            LlmEvent::HistoryCompacted {
                summary,
                cut,
                elapsed,
            } => {
                self.persist_summary(state, summary, &cut, elapsed);
                return None;
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
        let reasoning = self.take_reasoning();
        self.persist_message(state, tool_calls_message(narration, calls, reasoning));
    }

    fn persist_tool_result(
        &mut self,
        state: &AppState,
        call_id: String,
        name: &str,
        result: &ToolOutput,
    ) {
        self.awaiting_tool_result = false;
        let outcome = ToolOutcome {
            exit_code: result.exit_code.into(),
            duration_ms: result.duration_ms(),
        };
        let msg =
            PersistedMessage::tool_result(result.output.clone(), call_id, name.to_owned(), outcome);
        match (self.relay.take(), &self.persist) {
            (Some(trace), Some(target)) => {
                state.persist_relay_result_fire_and_forget(target.clone(), msg, trace.finish());
            }
            _ => self.persist_message(state, msg),
        }
    }

    /// Record `event` into the running relay's trace, starting one at its plan.
    fn record_relay(&mut self, event: &RelayEvent) {
        match (&mut self.relay, event) {
            (Some(trace), event) => trace.record(event),
            (None, RelayEvent::Planned { steps }) => {
                self.relay = Some(RelayTraceBuilder::planned(steps));
            }
            (None, _) => {}
        }
    }

    fn discard_partial_response(&mut self, sentence_buf: &mut SentenceBuffer) {
        self.assistant_accum.clear();
        self.reasoning = None;
        let _ = sentence_buf.finish();
    }

    fn persist_summary(
        &self,
        state: &AppState,
        summary: String,
        cut: &CompactionCut,
        elapsed: Duration,
    ) {
        if let Some(target) = &self.persist {
            let cut = SummaryCut {
                kept_turns: cut.kept_turns,
                kept_steps: cut.kept_steps,
            };
            state.persist_summary_fire_and_forget(target.branch, summary, cut, elapsed);
        }
    }

    fn persist_final_text(&mut self, state: &AppState) {
        let reasoning = self.take_reasoning();
        if self.assistant_accum.is_empty() {
            return;
        }
        let final_text = std::mem::take(&mut self.assistant_accum);
        publish_last_delta(&state.runtime, &self.id, &final_text);
        self.persist_message(
            state,
            PersistedMessage::assistant_text(final_text, reasoning),
        );
    }

    /// Record how long the turn ran and the tokens it produced.
    fn persist_turn_stats(&self, state: &AppState) {
        let Some(target) = &self.persist else {
            return;
        };
        let elapsed = self.started_at.elapsed();
        let stats = TurnStats {
            duration_ms: u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX),
            generated_tokens: self.tokens.tokens(),
        };
        state.persist_turn_stats_fire_and_forget(target.clone(), stats);
    }

    fn take_reasoning(&mut self) -> Option<Reasoning> {
        self.reasoning.take().map(ReasoningTrace::into_reasoning)
    }

    fn persist_message(&self, state: &AppState, msg: PersistedMessage) {
        if let Some(target) = &self.persist {
            state.persist_message_fire_and_forget(target.clone(), msg);
        }
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
        origin: TurnOrigin,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        let cancel = self.runtime.turn_cancellation();
        self.run_query(id, text, wire_attachments, origin, tx, cancel)
            .await
    }

    /// [`Self::handle_query`] for a turn whose cancellation token was
    /// taken earlier; the prompt is dropped if `cancel` has already fired.
    pub(super) async fn run_query(
        self: Arc<Self>,
        id: String,
        text: String,
        wire_attachments: Vec<ImageAttachment>,
        origin: TurnOrigin,
        tx: mpsc::Sender<Event>,
        cancel: CancellationToken,
    ) -> Result<(), DispatchError> {
        let _cancel_on_return = cancel.clone().drop_guard();
        let attachments = self.prepare_attachments(&id, wire_attachments, &tx).await?;
        let Some((_request_guard, agent_guard)) = self.admit_turn(&id, &tx, &cancel).await? else {
            return finish_interrupted_before_start(&tx, id).await;
        };
        if !self.revalidate_vision(&cancel).await {
            return finish_interrupted_before_start(&tx, id).await;
        }
        self.require_vision_for(&id, &attachments, &tx).await?;

        let (current_session, persist) = self.open_persistence_turn(&text).await;
        let chat_focused = self.refresh_chat_focus(&id);
        self.assemble_transient_context(&text, origin, chat_focused)
            .await;

        let title_user_text = text.clone();
        let (llm_tx, llm_rx) = mpsc::channel::<LlmEvent>(32);
        let agent_task = self.spawn_agent_task(text, attachments, llm_tx, cancel.clone());

        let (speech, speech_rx) = self.speech_pipeline();
        let voice_output = self.subsystems.voice.speech();
        voice_output.skip().await;
        let start_epoch = voice_output.current_epoch();
        let speech_handle = self.spawn_speech_worker(id.clone(), start_epoch, speech_rx);

        let done_emitted = self
            .drive_event_loop(id.clone(), llm_rx, &tx, speech, persist, &cancel)
            .await;

        let agent_result = agent_task.await;
        drop(agent_guard);

        if done_emitted && matches!(&agent_result, Ok(Ok(()))) {
            self.clone().spawn_session_title_broadcast(
                id.clone(),
                current_session,
                &title_user_text,
            );
        }

        self.finalize_turn(id, agent_result, speech_handle, &tx, done_emitted)
            .await
    }

    async fn prepare_attachments(
        &self,
        id: &str,
        wire: Vec<ImageAttachment>,
        tx: &mpsc::Sender<Event>,
    ) -> Result<Vec<Attachment>, DispatchError> {
        let decoded = decode_wire_attachments(wire).map_err(DispatchError::from);
        if let Err(e) = &decoded {
            send_error(tx, id.to_string(), e.to_string()).await;
        }
        decoded
    }

    async fn require_vision_for(
        &self,
        id: &str,
        attachments: &[Attachment],
        tx: &mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        let supported = self
            .subsystems
            .vision_revalidator
            .as_ref()
            .is_some_and(|revalidator| revalidator.gate().supported());
        if attachments.is_empty() || supported {
            return Ok(());
        }
        let error = DispatchError::VisionUnsupported;
        send_error(tx, id.to_string(), error.to_string()).await;
        Err(error)
    }

    async fn admit_turn(
        &self,
        id: &str,
        tx: &mpsc::Sender<Event>,
        cancel: &CancellationToken,
    ) -> Result<Option<(RequestGuard, OwnedMutexGuard<()>)>, DispatchError> {
        let admission = async {
            let agent_guard = self.runtime.agent_turn_lock.clone().lock_owned().await;
            let request_guard = self.acquire_request_guard(id, tx).await?;
            Ok((request_guard, agent_guard))
        };
        tokio::select! {
            biased;
            () = cancel.cancelled() => Ok(None),
            admitted = admission => admitted.map(Some),
        }
    }

    /// Re-probe vision support if stale; `false` when `cancel` fired first.
    async fn revalidate_vision(&self, cancel: &CancellationToken) -> bool {
        let Some(revalidator) = &self.subsystems.vision_revalidator else {
            return !cancel.is_cancelled();
        };
        tokio::select! {
            biased;
            () = cancel.cancelled() => false,
            () = revalidator.revalidate_if_stale(&self.subsystems.presence) => true,
        }
    }

    async fn acquire_request_guard(
        &self,
        id: &str,
        tx: &mpsc::Sender<Event>,
    ) -> Result<RequestGuard, DispatchError> {
        match self
            .subsystems
            .presence
            .acquire_request_guard_with_progress(id.to_string(), tx.clone())
            .await
        {
            Ok(guard) => Ok(guard),
            Err(e) => {
                send_error(tx, id.to_string(), format!("wake failed: {e}")).await;
                Err(e.into())
            }
        }
    }

    /// Save the session if needed and queue the write opening the turn with
    /// the user's prompt; the target is `None` when the session could not be saved.
    async fn open_persistence_turn(&self, text: &str) -> (SessionId, Option<PersistTarget>) {
        match self.saved_session().await {
            Ok((session, branch)) => {
                let target = self.open_turn_fire_and_forget(branch, PersistedMessage::user(text));
                (session, Some(target))
            }
            Err(e) => {
                warn!(
                    target: "assistd::memory",
                    error = %e,
                    "saving the session failed; turn will not be persisted"
                );
                let (session, _branch) = self.runtime.session_ctx.current().await;
                (session, None)
            }
        }
    }

    async fn assemble_transient_context(&self, text: &str, origin: TurnOrigin, chat_focused: bool) {
        let semantic = if self.recalls_past_context() {
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
        let brief = self.brief_reply_note(origin, chat_focused);
        if let Some(block) = combine_context_blocks([semantic, window, brief])
            && let Err(e) = self.subsystems.llm.set_transient_context(block).await
        {
            debug!(
                target: "assistd::context",
                error = %e,
                "set_transient_context failed; continuing without it",
            );
        }
    }

    fn recalls_past_context(&self) -> bool {
        let embedding = &self.memory.embedding_cfg;
        self.config.memory.enabled && embedding.enabled && embedding.auto_inject
    }

    fn speech_pipeline(&self) -> (SpeechPipeline, mpsc::Receiver<String>) {
        let synthesis = &self.config.voice.synthesis;
        let sentences = SentenceBuffer::new(synthesis.max_sentence_chars.get() as usize);
        let partial_flush = (synthesis.partial_flush_ms > 0)
            .then(|| Duration::from_millis(u64::from(synthesis.partial_flush_ms)));
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
        let tools = self.subsystems.tools.snapshot();
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
                self.subsystems.voice.speech().clone(),
                self.runtime.events_bus().clone(),
                id,
                start_epoch,
                speech_rx,
            )
            .in_current_span(),
        )
    }

    /// Forward the turn's events to `tx` and the speech worker until the agent finishes; a
    /// client that leaves cancels the turn, whose wind-down events are still persisted and
    /// published. Returns whether `Done` was emitted.
    async fn drive_event_loop(
        &self,
        id: String,
        mut llm_rx: mpsc::Receiver<LlmEvent>,
        tx: &mpsc::Sender<Event>,
        mut speech: SpeechPipeline,
        persist: Option<PersistTarget>,
        cancel: &CancellationToken,
    ) -> bool {
        let mut translator = TurnTranslator::new(id, persist);
        let mut client_connected = true;
        loop {
            let idle_timeout = speech
                .partial_flush
                .filter(|_| !translator.awaiting_tool_result);
            let next = tokio::select! {
                biased;
                () = tx.closed(), if client_connected => {
                    client_connected = false;
                    cancel_for_departed_client(cancel, &translator.id);
                    continue;
                }
                next = next_llm_event(&mut llm_rx, idle_timeout) => next,
            };
            let llm_event = match next {
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
            if !self.send_or_publish(tx, wire_event).await && client_connected {
                client_connected = false;
                cancel_for_departed_client(cancel, &translator.id);
            }
            speech.speak(sentences).await;
        }
        translator.persist_turn_stats(self);
        translator.done_emitted
    }

    /// Send `event` to the requester, or publish it straight to the bus
    /// once the requester has gone. Returns whether the requester got it.
    async fn send_or_publish(&self, tx: &mpsc::Sender<Event>, event: Event) -> bool {
        match tx.send(event).await {
            Ok(()) => true,
            Err(SendError(event)) => {
                self.runtime.publish(&event);
                false
            }
        }
    }

    /// Wait for speech and report the agent's outcome. Sends `Done` if the
    /// stream did not, as a cancelled turn ends without one.
    async fn finalize_turn(
        &self,
        id: String,
        agent_result: Result<Result<(), LlmError>, JoinError>,
        speech_handle: JoinHandle<()>,
        tx: &mpsc::Sender<Event>,
        done_emitted: bool,
    ) -> Result<(), DispatchError> {
        let _ = speech_handle.await;

        match agent_result {
            Ok(Ok(())) => {
                if !done_emitted {
                    self.send_or_publish(tx, Event::Done { id }).await;
                }
                Ok(())
            }
            Ok(Err(e)) => {
                let message = format!("llm backend error: {e}");
                self.send_or_publish(tx, Event::Error { id, message }).await;
                Err(e.into())
            }
            Err(join_err) => {
                let e = DispatchError::AgentPanicked(join_err);
                let message = e.to_string();
                self.send_or_publish(tx, Event::Error { id, message }).await;
                Err(e)
            }
        }
    }
}

pub(super) async fn finish_interrupted_before_start(
    tx: &mpsc::Sender<Event>,
    id: String,
) -> Result<(), DispatchError> {
    debug!(
        target: "assistd::state",
        id = %id,
        "turn interrupted before reaching the model; dropping the prompt"
    );
    let _ = tx.send(Event::Done { id }).await;
    Ok(())
}

fn cancel_for_departed_client(cancel: &CancellationToken, id: &str) {
    debug!(
        target: "assistd::state",
        id = %id,
        "client left mid-turn; cancelling the agent turn"
    );
    cancel.cancel();
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
    let mut speaking: Option<SpeakingGuard> = None;
    while let Some(sentence) = speech_rx.recv().await {
        match voice_output.should_speak(start_epoch) {
            SpeakDecision::Speak => {
                let Ok(output) = voice_output.output() else {
                    continue;
                };
                if speaking.is_none() {
                    let _ = events_bus.send(Event::SpeakingState {
                        id: id.clone(),
                        speaking: true,
                    });
                    speaking = Some(voice_output.begin_speaking());
                }
                voice_output.remember_spoken(&sentence);
                speak_sentence(&voice_output, output.as_ref(), sentence, start_epoch).await;
            }
            SpeakDecision::DropSilent | SpeakDecision::DropForSkip => {}
        }
    }
    if let Ok(output) = voice_output.output()
        && let Err(e) = output.wait_idle().await
    {
        debug!(
            target: "assistd::voice",
            error = %e,
            "voice_output.wait_idle failed (non-fatal)"
        );
    }
    if speaking.take().is_some() {
        let _ = events_bus.send(Event::SpeakingState {
            id,
            speaking: false,
        });
    }
}

/// Speak one sentence, then cancel playback if a skip landed meanwhile,
/// since `speak()` may append audio after the skip cleared the queue.
async fn speak_sentence(
    voice_output: &VoiceOutputController,
    output: &dyn VoiceOutput,
    sentence: String,
    start_epoch: u64,
) {
    let preview: String = sentence.chars().take(SENTENCE_PREVIEW_CHARS).collect();
    if let Err(e) = output.speak(sentence).await {
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
        output.cancel().await;
    }
}

fn tool_calls_message(
    narration: String,
    calls: &[ToolCall],
    reasoning: Option<Reasoning>,
) -> PersistedMessage {
    let content = if narration.trim().is_empty() {
        String::new()
    } else {
        narration
    };
    let persisted = calls
        .iter()
        .map(|call| PersistedToolCall {
            id: call.id.clone(),
            name: call.name.clone(),
            arguments: call.arguments.clone(),
        })
        .collect();
    PersistedMessage::assistant_tool_calls(content, persisted, reasoning)
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
