//! `TurnDriver`: carries one turn's agent events to its sink, the transcript and the
//! speech worker, for user turns and task runs alike.

use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::debug;

use assistd_config::SynthesisConfig;
use assistd_ipc::{Event, EventBody, EventKind, RelayEvent};
use assistd_llm::{CompactionCut, LlmEvent, ToolCall};
use assistd_memory::{
    PersistedMessage, PersistedRelayStep, PersistedToolCall, Reasoning, ToolOutcome, TurnStats,
};
use assistd_tools::ToolOutput;
use assistd_utils::time::saturating_millis;
use assistd_utils::token_counter::TokenCounter;
use assistd_voice::SentenceBuffer;

use super::persistence::TranscriptLog;
use super::relay_trace::RelayTraceBuilder;
use super::transcript_writer::PersistTarget;
use crate::bus::EventSink;

const LAST_DELTA_DEBOUNCE: Duration = Duration::from_millis(100);

/// Splits a spoken turn's reply into sentences for the speech worker.
pub(super) struct SpeechPipeline {
    tx: mpsc::Sender<String>,
    sentences: SentenceBuffer,
    partial_flush: Option<Duration>,
    first_sentence_emitted: bool,
}

impl SpeechPipeline {
    /// A pipeline shaped by `synthesis`, and the sentences it will produce.
    pub(super) fn new(synthesis: &SynthesisConfig) -> (Self, mpsc::Receiver<String>) {
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

    /// The sentences `event` completes; a discarded step's unspoken text is dropped.
    fn sentences_of(&mut self, event: &LlmEvent) -> Vec<String> {
        match event {
            LlmEvent::Delta { text } => self.sentences.push(text),
            LlmEvent::Done => self.sentences.finish().into_iter().collect(),
            LlmEvent::StepDiscarded => {
                let _ = self.sentences.finish();
                Vec::new()
            }
            _ => Vec::new(),
        }
    }

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
            duration_ms: saturating_millis(elapsed),
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
    assistant_accum: String,
    reasoning: Option<ReasoningTrace>,
    awaiting_tool_result: bool,
    /// What the relay in the running tool call has done, once it announced its plan.
    relay: Option<RelayTraceBuilder>,
    /// Backdated at start so the first `Delta` publishes `LastDelta` at once.
    last_emit_at: Instant,
    /// The text of the step that ended the turn, once it has.
    reply: Option<String>,
    started_at: Instant,
    tokens: TokenCounter,
}

impl TurnTranslator {
    /// The wire event for `event`, or `None` when it has no wire form. `LastDelta`
    /// goes straight to `events`' bus.
    async fn translate(&mut self, event: LlmEvent, events: &EventSink) -> Option<Event> {
        let body = match event {
            LlmEvent::StepStarted => return None,
            LlmEvent::StepDiscarded => {
                self.assistant_accum.clear();
                self.reasoning = None;
                return None;
            }
            LlmEvent::Delta { text } => {
                self.tokens.on_delta();
                self.accumulate_delta(events, &text);
                EventBody::Delta { text }
            }
            LlmEvent::ReasoningDelta { text } => {
                self.tokens.on_delta();
                self.reasoning
                    .get_or_insert_with(ReasoningTrace::new)
                    .push(&text);
                EventBody::ReasoningDelta { text }
            }
            LlmEvent::ToolCallsRequested { calls } => {
                self.persist_tool_calls(events, &calls).await;
                return None;
            }
            LlmEvent::ToolCall {
                name, arguments, ..
            } => {
                self.awaiting_tool_result = true;
                EventBody::ToolCall {
                    name,
                    args: arguments,
                }
            }
            LlmEvent::ToolResult {
                id: call_id,
                name,
                result,
            } => {
                self.persist_tool_result(call_id, &name, &result).await;
                let result = result.report();
                EventBody::ToolResult { name, result }
            }
            LlmEvent::Relay(event) => {
                self.record_relay(&event);
                EventBody::Relay { event }
            }
            LlmEvent::ContextSize { tokens, total } => {
                self.tokens.on_context_size(tokens, total);
                EventBody::ContextSize { tokens, total }
            }
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
            LlmEvent::HistoryCompacted {
                summary,
                cut,
                elapsed,
            } => {
                self.persist_summary(summary, cut, elapsed).await;
                return None;
            }
            LlmEvent::Done => {
                self.persist_final_text(events).await;
                EventBody::Done
            }
        };
        Some(Event::new(self.id.clone(), body))
    }

    fn accumulate_delta(&mut self, events: &EventSink, text: &str) {
        self.assistant_accum.push_str(text);
        if self.last_emit_at.elapsed() >= LAST_DELTA_DEBOUNCE {
            self.publish_last_delta(events, &self.assistant_accum);
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
        let narration = std::mem::take(&mut self.assistant_accum);
        if !narration.is_empty() {
            self.publish_last_delta(events, &narration);
        }
        let reasoning = self.take_reasoning();
        let msg = tool_calls_message(narration, calls, reasoning);
        self.persist_message(msg, None).await;
    }

    async fn persist_tool_result(&mut self, call_id: String, name: &str, result: &ToolOutput) {
        self.awaiting_tool_result = false;
        let outcome = ToolOutcome {
            exit_code: result.exit_code.into(),
            duration_ms: result.duration_ms(),
        };
        let msg =
            PersistedMessage::tool_result(result.output.clone(), call_id, name.to_owned(), outcome);
        let relay_steps = self.relay.take().map(RelayTraceBuilder::finish);
        self.persist_message(msg, relay_steps).await;
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

    async fn persist_summary(&self, summary: String, cut: CompactionCut, elapsed: Duration) {
        if let Some(target) = &self.persist {
            self.transcript
                .persist_summary(target.clone(), summary, cut, elapsed)
                .await;
        }
    }

    /// Persist the turn's last step and keep its text as the turn's reply.
    async fn persist_final_text(&mut self, events: &EventSink) {
        let reasoning = self.take_reasoning();
        let final_text = std::mem::take(&mut self.assistant_accum);
        self.reply = Some(final_text.clone());
        if final_text.is_empty() {
            return;
        }
        self.publish_last_delta(events, &final_text);
        let msg = PersistedMessage::assistant_text(final_text, reasoning);
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

    fn take_reasoning(&mut self) -> Option<Reasoning> {
        self.reasoning.take().map(ReasoningTrace::into_reasoning)
    }

    async fn persist_message(
        &self,
        msg: PersistedMessage,
        relay_steps: Option<Vec<PersistedRelayStep>>,
    ) {
        if let Some(target) = &self.persist {
            self.transcript
                .persist_message(target.clone(), msg, relay_steps)
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
pub(super) struct TurnDriver {
    translator: TurnTranslator,
    events: EventSink,
    speech: Option<SpeechPipeline>,
}

impl TurnDriver {
    /// A driver for turn `id` writing to `persist` in `transcript`, `None` when the turn
    /// is not persisted, and sending to `events`.
    pub(super) fn new(
        id: String,
        transcript: Arc<TranscriptLog>,
        persist: Option<PersistTarget>,
        events: EventSink,
    ) -> Self {
        let translator = TurnTranslator {
            id,
            transcript,
            persist,
            assistant_accum: String::new(),
            reasoning: None,
            awaiting_tool_result: false,
            relay: None,
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
    pub(super) fn speaking(mut self, speech: SpeechPipeline) -> Self {
        self.speech = Some(speech);
        self
    }

    /// Forward the turn's events until the agent finishes. A client that leaves cancels
    /// the turn through `cancel`, and its wind-down events are still persisted and
    /// published. Returns the turn's final reply once it emitted `Done`.
    pub(super) async fn drive(
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
                        speech.flush_idle().await;
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
                speech.speak(sentences).await;
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
