//! `TurnDriver`: carries one turn's agent events to its sink, the transcript and the
//! speech worker, for user turns and task runs alike; and the speech worker itself.

use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{broadcast, mpsc};
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

use assistd_config::SynthesisConfig;
use assistd_ipc::{Access, AgentId, Event, EventBody, EventKind, ToolReport};
use assistd_llm::{CompactionCut, LlmEvent, ToolCall};
use assistd_memory::{PersistedMessage, ToolTrace, TurnStats};
use assistd_utils::time::saturating_millis;
use assistd_utils::token_counter::TokenCounter;
use assistd_voice::{
    SentenceBuffer, SpeakDecision, SpeakingGuard, VoiceOutput, VoiceOutputController,
};

use crate::bus::EventSink;
use crate::transcript::agent_runs::AgentRunsRecorder;
use crate::transcript::messages::{StepDraft, tool_result_message};
use crate::transcript::{PersistTarget, TranscriptLog};

const LAST_DELTA_DEBOUNCE: Duration = Duration::from_millis(100);

const SENTENCE_PREVIEW_CHARS: usize = 60;

/// Splits a spoken turn's reply into sentences for the speech worker.
pub(crate) struct SpeechPipeline {
    tx: mpsc::Sender<String>,
    sentences: SentenceBuffer,
    partial_flush: Option<Duration>,
    first_sentence_emitted: bool,
}

impl SpeechPipeline {
    /// A pipeline shaped by `synthesis`, and the sentences it will produce.
    pub(crate) fn new(synthesis: &SynthesisConfig) -> (Self, mpsc::Receiver<String>) {
        let (tx, rx) = mpsc::channel::<String>(32);
        let pipeline = Self {
            tx,
            sentences: SentenceBuffer::new(synthesis.max_sentence_chars.get() as usize),
            partial_flush: (synthesis.partial_flush_ms > 0)
                .then(|| Duration::from_millis(u64::from(synthesis.partial_flush_ms))),
            first_sentence_emitted: false,
        };
        (pipeline, rx)
    }

    /// The sentences `event` completes, the unfinished one included when the step ends in
    /// tool calls or the turn is done; a discarded step's unspoken text is dropped.
    fn sentences_of(&mut self, event: &LlmEvent) -> Vec<String> {
        match event {
            LlmEvent::Delta { text } => self.sentences.push(text),
            LlmEvent::ToolCallsRequested { .. } | LlmEvent::Done => {
                self.sentences.finish().into_iter().collect()
            }
            LlmEvent::StepDiscarded => {
                let _ = self.sentences.finish();
                Vec::new()
            }
            _ => Vec::new(),
        }
    }

    fn flush_idle(&mut self) {
        if let Some(partial) = self.sentences.flush_idle() {
            self.queue_sentence(partial);
        }
    }

    fn speak(&mut self, sentences: Vec<String>) {
        for sentence in sentences {
            if !self.first_sentence_emitted {
                debug!(
                    target: "assistd::voice::latency",
                    stage = "first_sentence_emitted",
                    "voice latency stage"
                );
                self.first_sentence_emitted = true;
            }
            self.queue_sentence(sentence);
        }
    }

    /// Hand `sentence` to the speech worker, dropping it when the worker is gone or has
    /// fallen a full queue behind, so a stalled voice never holds up the turn's events.
    fn queue_sentence(&self, sentence: String) {
        if let Err(error) = self.tx.try_send(sentence) {
            debug!(
                target: "assistd::voice",
                %error,
                "speech worker is not taking sentences; dropping one"
            );
        }
    }
}

/// Translates one turn's agent events into wire events, persisting the
/// transcript as it goes.
struct TurnTranslator {
    id: String,
    transcript: Arc<TranscriptLog>,
    /// `None` when the session could not be saved, so nothing is persisted.
    persist: Option<PersistTarget>,
    /// What the turn's own agent has streamed of the step in flight.
    step: StepDraft,
    awaiting_tool_result: bool,
    /// What the sub-agents of the running tool call have done, once it announced them.
    agent_runs: Option<AgentRunsRecorder>,
    /// Backdated at start so the first `Delta` publishes `LastDelta` at once.
    last_emit_at: Instant,
    /// The text of the step that ended the turn, once it has.
    reply: Option<String>,
    started_at: Instant,
    tokens: TokenCounter,
}

impl TurnTranslator {
    /// The wire event for `event`, or `None` when it has no wire form, after recording
    /// it: the turn's own into the transcript, a sub-agent's into its run's history.
    /// `LastDelta` goes straight to `events`' bus.
    async fn translate(&mut self, event: LlmEvent, events: &EventSink) -> Option<Event> {
        match event {
            LlmEvent::FromAgent { agent, event, .. } => {
                self.record_from_agent(agent, &event);
                wire_body(*event).map(|body| Event::from_agent(self.id.clone(), agent, body))
            }
            event => {
                self.record(&event, events).await;
                wire_body(event).map(|body| Event::new(self.id.clone(), body))
            }
        }
    }

    /// Record one of the turn's own agent's events.
    async fn record(&mut self, event: &LlmEvent, events: &EventSink) {
        match event {
            LlmEvent::StepDiscarded => self.step.discard(),
            LlmEvent::Delta { text } => {
                self.tokens.on_delta();
                self.accumulate_delta(events, text);
            }
            LlmEvent::ReasoningDelta { text } => {
                self.tokens.on_delta();
                self.step.push_reasoning(text);
            }
            LlmEvent::ToolCallsRequested { calls } => self.persist_tool_calls(events, calls).await,
            LlmEvent::ToolCall { .. } => self.awaiting_tool_result = true,
            LlmEvent::ToolResult {
                id,
                name,
                result,
                read_host,
                accesses,
            } => {
                self.persist_tool_result(id, name, result, *read_host, accesses)
                    .await;
            }
            LlmEvent::RelayPlanned { steps } => {
                self.agent_runs = Some(AgentRunsRecorder::planned(steps));
            }
            LlmEvent::RelayStepStarted { step, agent } => {
                if let Some(runs) = &mut self.agent_runs {
                    runs.start(*step, *agent);
                }
            }
            LlmEvent::RelayStepEnded {
                step,
                end,
                duration_ms,
            } => {
                if let Some(runs) = &mut self.agent_runs {
                    runs.end(*step, end, *duration_ms);
                }
            }
            LlmEvent::ContextSize { tokens, total } => {
                self.tokens.on_context_size(*tokens, *total);
            }
            LlmEvent::HistoryCompacted {
                summary,
                cut,
                elapsed,
            } => self.persist_summary(summary.clone(), *cut, *elapsed).await,
            LlmEvent::Interrupted { reply } => self.persist_interrupted_reply(reply.clone()).await,
            LlmEvent::Done => self.persist_final_text(events).await,
            LlmEvent::StepStarted
            | LlmEvent::Task { .. }
            | LlmEvent::Status { .. }
            | LlmEvent::FromAgent { .. } => {}
        }
    }

    /// Record what sub-agent `agent` streamed in the history of its run.
    fn record_from_agent(&mut self, agent: AgentId, event: &LlmEvent) {
        if let Some(runs) = &mut self.agent_runs {
            runs.record(agent, event);
        }
    }

    fn accumulate_delta(&mut self, events: &EventSink, text: &str) {
        self.step.push_text(text);
        if self.last_emit_at.elapsed() >= LAST_DELTA_DEBOUNCE {
            self.publish_last_delta(events, self.step.text());
            self.last_emit_at = Instant::now();
        }
    }

    /// `LastDelta` carries the whole reply so far, so building one costs a
    /// copy of it; skip that when no subscriber would receive it.
    fn publish_last_delta(&self, events: &EventSink, text: &str) {
        if events.publishes(EventKind::LastDelta) {
            events.publish(&Event::new(
                self.id.clone(),
                EventBody::LastDelta {
                    text: text.to_string(),
                },
            ));
        }
    }

    async fn persist_tool_calls(&mut self, events: &EventSink, calls: &[ToolCall]) {
        if !self.step.text().is_empty() {
            self.publish_last_delta(events, self.step.text());
        }
        let msg = self.step.requesting(calls);
        self.persist_message(msg, None).await;
    }

    async fn persist_tool_result(
        &mut self,
        call_id: &str,
        name: &str,
        result: &ToolReport,
        read_host: bool,
        accesses: &[Access],
    ) {
        self.awaiting_tool_result = false;
        let msg = tool_result_message(call_id, name, result, read_host);
        let trace = ToolTrace {
            accesses: accesses.to_vec(),
            agent_runs: self
                .agent_runs
                .take()
                .map_or_default(AgentRunsRecorder::finish),
        };
        self.persist_message(msg, Some(trace)).await;
    }

    async fn persist_summary(&self, summary: String, cut: CompactionCut, elapsed: Duration) {
        if let Some(target) = &self.persist {
            self.transcript
                .persist_summary(target.clone(), summary, cut, elapsed)
                .await;
        }
    }

    /// Persist the turn's last step and keep its text as the turn's reply.
    async fn persist_final_text(&mut self, events: &EventSink) {
        self.reply = Some(self.step.text().to_owned());
        let Some(msg) = self.step.reply() else {
            return;
        };
        self.publish_last_delta(events, &msg.content);
        self.persist_message(msg, None).await;
    }

    async fn persist_interrupted_reply(&mut self, reply: String) {
        let msg = self.step.interrupted(reply);
        self.persist_message(msg, None).await;
    }

    /// Record how long the turn ran and the tokens it produced.
    async fn persist_turn_stats(&self) {
        let Some(target) = &self.persist else {
            return;
        };
        let elapsed = self.started_at.elapsed();
        let stats = TurnStats {
            duration_ms: saturating_millis(elapsed),
            generated_tokens: self.tokens.tokens(),
        };
        self.transcript
            .persist_turn_stats(target.clone(), stats)
            .await;
    }

    async fn persist_message(&self, msg: PersistedMessage, trace: Option<ToolTrace>) {
        if let Some(target) = &self.persist {
            self.transcript
                .persist_message(target.clone(), msg, trace)
                .await;
        }
    }
}

enum NextLlmEvent {
    Event(LlmEvent),
    Idle,
    Closed,
}

/// Drives one turn: every agent event is persisted, sent to the turn's sink, and spoken
/// when the turn has a speech pipeline.
pub(crate) struct TurnDriver {
    translator: TurnTranslator,
    events: EventSink,
    speech: Option<SpeechPipeline>,
}

impl TurnDriver {
    /// A driver for turn `id` writing to `persist` in `transcript`, `None` when the turn
    /// is not persisted, and sending to `events`.
    pub(crate) fn new(
        id: String,
        transcript: Arc<TranscriptLog>,
        persist: Option<PersistTarget>,
        events: EventSink,
    ) -> Self {
        let translator = TurnTranslator {
            id,
            transcript,
            persist,
            step: StepDraft::default(),
            awaiting_tool_result: false,
            agent_runs: None,
            last_emit_at: Instant::now()
                .checked_sub(LAST_DELTA_DEBOUNCE)
                .unwrap_or_else(Instant::now),
            reply: None,
            started_at: Instant::now(),
            tokens: TokenCounter::new(),
        };
        Self {
            translator,
            events,
            speech: None,
        }
    }

    /// Speak the turn's reply through `speech`.
    pub(crate) fn speaking(mut self, speech: SpeechPipeline) -> Self {
        self.speech = Some(speech);
        self
    }

    /// Forward the turn's events until the agent finishes. A client that leaves cancels
    /// the turn through `cancel`, and its wind-down events are still persisted and
    /// published. Returns the turn's final reply once the agent finished it.
    pub(crate) async fn drive(
        mut self,
        mut llm_rx: mpsc::Receiver<LlmEvent>,
        cancel: &CancellationToken,
    ) -> Option<String> {
        let mut client_connected = true;
        loop {
            let next = tokio::select! {
                biased;
                () = self.events.closed(), if client_connected => {
                    client_connected = false;
                    self.cancel_for_departed_client(cancel);
                    continue;
                }
                next = next_llm_event(&mut llm_rx, self.idle_flush_after()) => next,
            };
            let llm_event = match next {
                NextLlmEvent::Event(event) => event,
                NextLlmEvent::Closed => break,
                NextLlmEvent::Idle => {
                    if let Some(speech) = &mut self.speech {
                        speech.flush_idle();
                    }
                    continue;
                }
            };
            let sentences = match &mut self.speech {
                Some(speech) => speech.sentences_of(&llm_event),
                None => Vec::new(),
            };
            if let Some(wire_event) = self.translator.translate(llm_event, &self.events).await
                && self.events.send(wire_event).await.is_err()
                && client_connected
            {
                client_connected = false;
                self.cancel_for_departed_client(cancel);
            }
            if let Some(speech) = &mut self.speech {
                speech.speak(sentences);
            }
        }
        self.translator.persist_turn_stats().await;
        self.translator.reply
    }

    /// How long the stream may stall before the unspoken partial sentence is flushed;
    /// `None` while a tool runs or nothing is spoken.
    fn idle_flush_after(&self) -> Option<Duration> {
        self.speech
            .as_ref()
            .and_then(|speech| speech.partial_flush)
            .filter(|_| !self.translator.awaiting_tool_result)
    }

    fn cancel_for_departed_client(&self, cancel: &CancellationToken) {
        debug!(
            target: "assistd::state",
            id = %self.translator.id,
            "client left mid-turn; cancelling the agent turn"
        );
        cancel.cancel();
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

/// What a client is sent of an agent's `event`, whichever agent of the turn streamed it;
/// `None` for an event with no wire form.
fn wire_body(event: LlmEvent) -> Option<EventBody> {
    Some(match event {
        LlmEvent::StepDiscarded => EventBody::StepDiscarded,
        LlmEvent::Delta { text } => EventBody::Delta { text },
        LlmEvent::ReasoningDelta { text } => EventBody::ReasoningDelta { text },
        LlmEvent::ToolCall {
            name, arguments, ..
        } => EventBody::ToolCall {
            name,
            args: arguments,
        },
        LlmEvent::ToolResult { name, result, .. } => EventBody::ToolResult {
            name,
            result: *result,
        },
        LlmEvent::RelayPlanned { steps } => EventBody::RelayPlanned { steps },
        LlmEvent::RelayStepStarted { step, agent } => EventBody::RelayStepStarted {
            step,
            sub_agent: agent,
        },
        LlmEvent::RelayStepEnded {
            step,
            end,
            duration_ms,
        } => EventBody::RelayStepEnded {
            step,
            end,
            duration_ms,
        },
        LlmEvent::ContextSize { tokens, total } => EventBody::ContextSize { tokens, total },
        LlmEvent::Status {
            severity,
            component,
            event,
            message,
        } => EventBody::Status {
            severity,
            component,
            event,
            message,
        },
        LlmEvent::StepStarted
        | LlmEvent::Task { .. }
        | LlmEvent::ToolCallsRequested { .. }
        | LlmEvent::FromAgent { .. }
        | LlmEvent::HistoryCompacted { .. }
        | LlmEvent::Interrupted { .. }
        | LlmEvent::Done => return None,
    })
}

pub(crate) async fn run_speech_worker(
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
                    let _ = events_bus.send(Event::new(
                        id.clone(),
                        EventBody::SpeakingState { speaking: true },
                    ));
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
        let _ = events_bus.send(Event::new(id, EventBody::SpeakingState { speaking: false }));
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
