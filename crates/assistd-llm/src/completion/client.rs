//! HTTP streaming client for the locally-managed llama-server.

use std::collections::BTreeMap;
use std::mem::take;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;

use assistd_config::{
    ChatConfig, ModelConfig, NonBlankString, PresencePenalty, TimeoutsConfig, UnitInterval,
};
use assistd_ipc::{Attachment, Component, StatusKind, StatusSeverity};
use assistd_utils::child_server::ApiKey;
use assistd_utils::http::local_client_builder;
use assistd_utils::text::neutralise_template_tokens;
use async_trait::async_trait;
use serde_json::{Map, Value};
use tokio::sync::{Mutex, MutexGuard, mpsc};
use tokio::time::{Instant, timeout, timeout_at};
use tracing::{debug, info, warn};

use super::error::LlamaClientError;
use super::history::{
    AssistantMessage, History, Message, Role, Summarizer, SummaryRequest, ToolCallRecord,
    UserMessage, approx_tokens_bytes, ensure_budget,
};
use super::sse::{SseEvent, SseLineReader};
use super::think_splitter::{Segment, ThinkSplitter};
use super::wire;
use crate::{
    HistoryEntry, HistoryReplay, HistoryRole, LlmBackend, LlmError, LlmEvent, LlmHealthProbe,
    LlmResult, OneShotReply, ReadyState, ReplyLimits, SlotAssignment, SlotHolder, SlotStash,
    StepOutcome, Thinking, ToolCall, ToolResultPayload, TurnBackend,
};

const ERROR_BODY_CAP: usize = 1024;
/// The `type` llama-server gives an error refusing a prompt larger than its context window.
const CONTEXT_OVERFLOW_ERROR: &str = "exceed_context_size_error";
/// Written into the model's thinking where its budget ran out, so it moves
/// on instead of starting over.
const REASONING_BUDGET_MESSAGE: &str =
    "\n\nThinking budget reached. Stop planning and act now: make the next tool call or answer.\n";
/// Tool-call markup the chat template emits; a summary containing it tried
/// to act instead of summarizing.
const TOOL_CALL_MARKUP: &str = "<tool_call>";
/// Sampling temperature of a summarization call: low, so a summary keeps to
/// what was said.
const SUMMARY_TEMPERATURE: f32 = 0.3;
/// The slot the main backend and those sharing it with a [`SlotStash`] use.
const SHARED_SLOT: u32 = 0;

/// HTTP streaming client backed by a locally-managed llama-server. The history lock
/// is held only while state is mutated, never across the HTTP stream, a compaction summary or
/// an event send, so neither a hung server nor a stalled consumer blocks `push_user` or
/// `set_transient_context`.
#[derive(Debug)]
pub struct LlamaClient {
    client: reqwest::Client,
    base_url: String,
    chat: ChatConfig,
    model: ModelConfig,
    timeouts: TimeoutsConfig,
    history: Mutex<History>,
    /// When set, no request is sent unless the probe reports a serving
    /// child, and a failure coinciding with a supervisor restart becomes
    /// [`LlmError::ServerRestarting`] instead of a transport fault.
    health: Option<Arc<dyn LlmHealthProbe>>,
    /// Changes only between requests, when the client moves to another slot.
    slot: parking_lot::Mutex<SlotBinding>,
}

impl LlamaClient {
    /// Build a client for `model.listen_addr()`, sending `api_key` when set. With a `health`
    /// probe, a request made while its child is not serving fails with
    /// [`LlmError::ServerRestarting`] without being sent.
    pub fn new(
        chat: &ChatConfig,
        model: &ModelConfig,
        timeouts: &TimeoutsConfig,
        api_key: Option<&ApiKey>,
        health: Option<Arc<dyn LlmHealthProbe>>,
    ) -> Result<Self, LlamaClientError> {
        let client = local_client_builder()
            .default_headers(api_key.map_or_default(ApiKey::authorization_headers))
            .connect_timeout(Duration::from_secs(10))
            .build()?;
        let base_url = format!("http://{}", model.listen_addr());
        let history = History::new(chat.system_prompt.clone());
        Ok(Self {
            client,
            base_url,
            chat: chat.clone(),
            model: model.clone(),
            timeouts: timeouts.clone(),
            history: Mutex::new(history),
            health,
            slot: parking_lot::Mutex::new(SlotBinding::Unassigned),
        })
    }

    /// Keep this, the main backend, on the shared slot in `stash`,
    /// switching the slot to it before each request.
    pub fn with_slot_stash(mut self, stash: Arc<SlotStash>) -> Self {
        *self.slot.get_mut() = SlotBinding::Shared {
            stash,
            holder: SlotHolder::Main,
        };
        self
    }

    async fn claim_slot(&self) {
        if let Some((stash, holder)) = self.shared_slot() {
            stash.switch_to(holder).await;
        }
    }

    fn shared_slot(&self) -> Option<(Arc<SlotStash>, SlotHolder)> {
        match &*self.slot.lock() {
            SlotBinding::Shared { stash, holder } => Some((Arc::clone(stash), *holder)),
            SlotBinding::Unassigned | SlotBinding::Pinned(_) => None,
        }
    }

    fn base_request<'a>(&'a self, messages: Vec<wire::ChatMessage<'a>>) -> wire::ChatRequest<'a> {
        wire::ChatRequest {
            model: self.model.name.as_str(),
            messages,
            stream: true,
            stream_options: Some(wire::StreamOptions {
                include_usage: true,
            }),
            temperature: self.chat.temperature.get(),
            max_tokens: self.chat.max_response_tokens.get(),
            top_p: self.chat.top_p.map(UnitInterval::get),
            top_k: self.chat.top_k.map(NonZeroU32::get),
            min_p: self.chat.min_p.map(UnitInterval::get),
            presence_penalty: self.chat.presence_penalty.map(PresencePenalty::get),
            tools: None,
            tool_choice: None,
            reasoning_budget_tokens: Some(self.chat.max_reasoning_tokens.get()),
            reasoning_budget_message: Some(REASONING_BUDGET_MESSAGE),
            chat_template_kwargs: self.template_kwargs(Thinking::Enabled),
            response_format: None,
            grammar: None,
            id_slot: self.slot.lock().index(),
        }
    }

    /// The configured chat-template variables, or `None` when the template
    /// defaults apply to all of them.
    fn template_kwargs(&self, thinking: Thinking) -> Option<wire::ChatTemplateKwargs<'_>> {
        let enable_thinking = (thinking == Thinking::Disabled).then_some(false);
        let reasoning_effort = self
            .chat
            .reasoning_effort
            .as_ref()
            .map(NonBlankString::as_str);
        (enable_thinking.is_some() || reasoning_effort.is_some()).then_some(
            wire::ChatTemplateKwargs {
                enable_thinking,
                reasoning_effort,
            },
        )
    }

    fn sample_greedily_without_thinking<'a>(&'a self, payload: &mut wire::ChatRequest<'a>) {
        payload.temperature = 0.0;
        payload.reasoning_budget_tokens = None;
        payload.reasoning_budget_message = None;
        payload.chat_template_kwargs = self.template_kwargs(Thinking::Disabled);
    }

    /// Classify a failed request: a restart-coincident failure becomes
    /// [`StreamOutcome::ServerRestart`]; anything else keeps `err` along
    /// with whatever was already streamed.
    async fn fail(
        &self,
        accum: StreamAccum,
        err: LlamaClientError,
        pid_at_request: Option<u32>,
    ) -> StreamOutcome {
        if self.looks_like_server_crash(pid_at_request).await {
            let pre_emit = !accum.has_output();
            return StreamOutcome::ServerRestart {
                accum: Box::new(accum),
                pre_emit,
            };
        }
        if accum.has_output() {
            warn!(
                target: "assistd::completion",
                "mid-stream error after {} bytes / {} tool-call builders: {err}",
                accum.text.len(),
                accum.tool_calls.len()
            );
            StreamOutcome::PartialAfterEmit {
                accum: Box::new(accum),
                err,
            }
        } else {
            StreamOutcome::PreEmitError(err)
        }
    }

    async fn looks_like_server_crash(&self, pid_at_request: Option<u32>) -> bool {
        let Some(probe) = self.health.as_ref() else {
            return false;
        };
        let current = probe.snapshot().await;
        let current_pid = current.and_then(|snapshot| snapshot.pid);
        let pid_changed = match (pid_at_request, current_pid) {
            (Some(_), None) => true,
            (Some(a), Some(b)) => a != b,
            _ => false,
        };
        let state_unhealthy = current.is_some_and(|snapshot| snapshot.state != ReadyState::Ready);
        pid_changed || state_unhealthy
    }

    /// The pid of the serving child a request is sent to, `None` when no
    /// supervisor is attached. Fails with [`LlamaClientError::NotReady`] when
    /// the supervised child is not serving.
    async fn serving_pid(&self) -> Result<Option<u32>, LlamaClientError> {
        let Some(probe) = self.health.as_ref() else {
            return Ok(None);
        };
        match probe.snapshot().await {
            Some(snapshot) if snapshot.is_serving() => Ok(snapshot.pid),
            _ => {
                warn!(target: "assistd::completion", "llama-server is not ready; request not sent");
                Err(LlamaClientError::NotReady)
            }
        }
    }

    /// Stream one completion. `compaction_headroom` is how many generated
    /// tokens may stream before [`StatusKind::CompactingSoon`] is sent;
    /// `None` never sends it.
    async fn stream_openai(
        &self,
        body: Vec<u8>,
        compaction_headroom: Option<NonZeroU32>,
        tx: &mpsc::Sender<LlmEvent>,
    ) -> StreamOutcome {
        let Ok(pid_at_request) = self.serving_pid().await else {
            return StreamOutcome::ServerRestart {
                accum: Box::default(),
                pre_emit: true,
            };
        };
        debug!(
            target: "assistd::voice::latency",
            stage = "llm_request_sent",
            "voice latency stage"
        );
        let first_byte_by = Instant::now() + self.request_timeout();
        let mut response = match self.send_request(body, first_byte_by, pid_at_request).await {
            Ok(response) => response,
            Err(outcome) => return outcome,
        };
        let mut accum = StreamAccum {
            compaction_headroom,
            ..StreamAccum::default()
        };
        let saw_done = match self
            .read_stream(&mut response, &mut accum, first_byte_by, tx, pid_at_request)
            .await
        {
            Ok(saw_done) => saw_done,
            Err(outcome) => return outcome,
        };
        if !saw_done {
            if accum.finish_reason.is_none() {
                let err =
                    LlamaClientError::Sse("stream ended without [DONE] or a finish_reason".into());
                return self.fail(accum, err, pid_at_request).await;
            }
            warn!(
                target: "assistd::completion",
                "stream ended before [DONE] marker; accumulated {} bytes text, {} tool-call builders",
                accum.text.len(),
                accum.tool_calls.len()
            );
        }
        flush_splitter(&mut accum, tx).await;
        StreamOutcome::Ok(Box::new(accum))
    }

    /// Stream one completion outside the history and return its text,
    /// with reasoning discarded; `caller` names the request in a crash error.
    async fn stream_text(&self, body: Vec<u8>, caller: &str) -> LlmResult<String> {
        match self.stream_unobserved(body).await {
            StreamOutcome::Ok(accum) | StreamOutcome::ClientDisconnected(accum) => Ok(accum.text),
            StreamOutcome::PreEmitError(e) | StreamOutcome::PartialAfterEmit { err: e, .. } => {
                Err(LlmError::Client(e))
            }
            StreamOutcome::ServerRestart { .. } => Err(LlmError::ServerRestarting(format!(
                "llama-server crashed during {caller}"
            ))),
        }
    }

    /// Stream one completion whose events nobody watches.
    async fn stream_unobserved(&self, body: Vec<u8>) -> StreamOutcome {
        let (tx, mut rx) = mpsc::channel::<LlmEvent>(64);
        let stream = async {
            let outcome = self.stream_openai(body, None, &tx).await;
            drop(tx);
            outcome
        };
        let drain = async { while rx.recv().await.is_some() {} };
        let (outcome, ()) = tokio::join!(stream, drain);
        outcome
    }

    /// Compact the history when it is over budget, summarizing with the
    /// same `tools` the coming request offers so its prompt prefix is
    /// unchanged, and return it locked for that request, with the events to
    /// send once it is unlocked.
    async fn fit_budget(
        &self,
        tools: &[Value],
        tx: &mpsc::Sender<LlmEvent>,
    ) -> LlmResult<(MutexGuard<'_, History>, Vec<LlmEvent>)> {
        let mut history = self.history.lock().await;
        history.set_offered_tools(tools.iter().map(|tool| tool.to_string().len()).sum());
        if !history.exceeds_budget(&self.chat, &self.model) {
            let compacting_soon = self
                .compaction_headroom(&history)
                .is_none()
                .then(|| compaction_status(StatusKind::CompactingSoon, "compacting soon"));
            return Ok((history, compacting_soon.into_iter().collect()));
        }
        let before = (history.message_count(), history.approx_total_tokens());
        drop(history);
        send_compaction_status(tx, StatusKind::CompactingHistory, "compacting history…").await;
        let started_at = Instant::now();
        let summarizer = ContinuationSummarizer {
            client: self,
            tools,
        };
        let outcome = ensure_budget(&self.history, &summarizer, &self.chat, &self.model).await;
        let mut history = self.history.lock().await;
        if let Err(e) = outcome {
            warn!(
                target: "assistd::completion",
                "ensure_budget failed ({e}); falling back to truncation"
            );
            if history.exceeds_budget(&self.chat, &self.model) {
                history.truncate_to_budget(&self.chat, &self.model);
            }
        }
        info!(
            target: "assistd::completion",
            messages_before = before.0,
            messages_after = history.message_count(),
            tokens_before = before.1,
            tokens_after = history.approx_total_tokens(),
            "history compacted to fit the token budget"
        );
        let events = compaction_events(&history, started_at);
        Ok((history, events))
    }

    /// Drop history down to the low-water mark of the estimate, which the
    /// server's refusal of the last prompt has just corrected.
    async fn truncate_after_overflow(&self, tx: &mpsc::Sender<LlmEvent>) {
        send_compaction_status(tx, StatusKind::CompactingHistory, "compacting history…").await;
        let started_at = Instant::now();
        let events = {
            let mut history = self.history.lock().await;
            let before = (history.message_count(), history.approx_total_tokens());
            history.truncate_to_budget(&self.chat, &self.model);
            warn!(
                target: "assistd::completion",
                messages_before = before.0,
                messages_after = history.message_count(),
                tokens_before = before.1,
                tokens_after = history.approx_total_tokens(),
                "prompt exceeded the context window; history truncated to retry"
            );
            compaction_events(&history, started_at)
        };
        send_events(tx, events).await;
    }

    /// Fit the history to its budget, send it with `tools` offered, and
    /// correct the token estimate by what the server measured.
    async fn attempt_step(
        &self,
        tools: &[Value],
        tx: &mpsc::Sender<LlmEvent>,
    ) -> LlmResult<StreamOutcome> {
        let (body_bytes, estimated_tokens, compaction_headroom, events) = {
            let (history, mut events) = self.fit_budget(tools, tx).await?;
            events.push(context_size(&history));
            let mut payload = self.base_request(history.as_wire_messages());
            offer_tools(&mut payload, tools.to_vec());
            let body = serde_json::to_vec(&payload)
                .map_err(|e| LlmError::Client(LlamaClientError::Json(e)))?;
            (
                body,
                history.heuristic_tokens(),
                self.compaction_headroom(&history),
                events,
            )
        };
        send_events(tx, events).await;

        self.claim_slot().await;
        let outcome = self
            .stream_openai(body_bytes, compaction_headroom, tx)
            .await;
        let mut history = self.history.lock().await;
        self.calibrate(&mut history, estimated_tokens, &outcome);
        Ok(outcome)
    }

    /// Record the prompt size the server measured for a request estimated at
    /// `estimated_tokens`; a refusal naming no size counts as a full window.
    fn calibrate(&self, history: &mut History, estimated_tokens: u32, outcome: &StreamOutcome) {
        let measured = match outcome {
            StreamOutcome::Ok(accum)
            | StreamOutcome::PartialAfterEmit { accum, .. }
            | StreamOutcome::ClientDisconnected(accum) => accum.prompt_tokens,
            StreamOutcome::PreEmitError(LlamaClientError::ContextOverflow { prompt_tokens }) => {
                Some(prompt_tokens.unwrap_or(self.model.context_length.get()))
            }
            StreamOutcome::PreEmitError(_) | StreamOutcome::ServerRestart { .. } => None,
        };
        if let Some(prompt_tokens) = measured {
            debug!(
                target: "assistd::completion",
                estimated_tokens,
                prompt_tokens,
                "calibrating token estimate from server usage"
            );
            history.calibrate(estimated_tokens, prompt_tokens);
        }
    }

    /// `None` when compaction is already near, so it is announced once.
    fn compaction_headroom(&self, history: &History) -> Option<NonZeroU32> {
        NonZeroU32::new(history.compaction_warning_headroom(&self.chat, &self.model))
    }

    fn request_timeout(&self) -> Duration {
        Duration::from_secs(self.chat.request_timeout_secs.get())
    }

    fn stream_inactivity(&self) -> Duration {
        Duration::from_secs(self.timeouts.stream_inactivity_secs)
    }

    async fn send_request(
        &self,
        body: Vec<u8>,
        first_byte_by: Instant,
        pid_at_request: Option<u32>,
    ) -> Result<reqwest::Response, StreamOutcome> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let send = self
            .client
            .post(&url)
            .header("Accept", "text/event-stream")
            .header("Content-Type", "application/json")
            .body(body)
            .send();
        let mut response = match timeout_at(first_byte_by, send).await {
            Ok(Ok(response)) => response,
            Ok(Err(e)) => {
                return Err(self
                    .fail(
                        StreamAccum::default(),
                        LlamaClientError::Http(e),
                        pid_at_request,
                    )
                    .await);
            }
            Err(_) => {
                let err = LlamaClientError::Sse(format!(
                    "no response headers within {}s",
                    self.request_timeout().as_secs()
                ));
                return Err(self.fail(StreamAccum::default(), err, pid_at_request).await);
            }
        };
        if self.looks_like_server_crash(pid_at_request).await {
            return Err(StreamOutcome::ServerRestart {
                accum: Box::default(),
                pre_emit: true,
            });
        }
        let status = response.status();
        if !status.is_success() {
            let body =
                read_body_capped(&mut response, ERROR_BODY_CAP, self.stream_inactivity()).await;
            let err = http_error(status.as_u16(), body);
            return Err(self.fail(StreamAccum::default(), err, pid_at_request).await);
        }
        Ok(response)
    }

    async fn read_stream(
        &self,
        response: &mut reqwest::Response,
        accum: &mut StreamAccum,
        first_byte_by: Instant,
        tx: &mpsc::Sender<LlmEvent>,
        pid_at_request: Option<u32>,
    ) -> Result<bool, StreamOutcome> {
        let mut reader = SseLineReader::new();
        let inter_chunk = self.stream_inactivity();
        let mut saw_bytes = false;
        loop {
            let (deadline, next) = if saw_bytes {
                (inter_chunk, timeout(inter_chunk, response.chunk()).await)
            } else {
                (
                    self.request_timeout(),
                    timeout_at(first_byte_by, response.chunk()).await,
                )
            };
            let chunk = match next {
                Ok(Ok(Some(chunk))) => chunk,
                Ok(Ok(None)) => return Ok(false),
                Ok(Err(e)) => {
                    return Err(self
                        .fail(take(accum), LlamaClientError::Http(e), pid_at_request)
                        .await);
                }
                Err(_) => {
                    warn!(
                        target: "assistd::completion",
                        timeout_secs = deadline.as_secs(),
                        phase = if saw_bytes { "inter_chunk" } else { "first_byte" },
                        bytes_so_far = accum.text.len(),
                        tool_call_builders = accum.tool_calls.len(),
                        "SSE stream inactive past deadline; aborting read"
                    );
                    let err = LlamaClientError::Sse(format!(
                        "no bytes received for {}s",
                        deadline.as_secs()
                    ));
                    return Err(self.fail(take(accum), err, pid_at_request).await);
                }
            };
            saw_bytes = true;
            reader.feed(&chunk);
            loop {
                match reader.next_event() {
                    Ok(Some(SseEvent::Data(payload))) => {
                        self.apply_chunk(&payload, accum, tx, pid_at_request)
                            .await?;
                    }
                    Ok(Some(SseEvent::Error(payload))) => {
                        let err = error_event(&payload);
                        return Err(self.fail(take(accum), err, pid_at_request).await);
                    }
                    Ok(Some(SseEvent::Done)) => return Ok(true),
                    Ok(None) => break,
                    Err(e) => return Err(self.fail(take(accum), e, pid_at_request).await),
                }
            }
        }
    }

    /// Fold one `data:` payload into `accum`, forwarding its deltas.
    async fn apply_chunk(
        &self,
        payload: &str,
        accum: &mut StreamAccum,
        tx: &mpsc::Sender<LlmEvent>,
        pid_at_request: Option<u32>,
    ) -> Result<(), StreamOutcome> {
        let parsed: wire::ChatCompletionChunk = match serde_json::from_str(payload) {
            Ok(parsed) => parsed,
            Err(e) => {
                return Err(self
                    .fail(take(accum), LlamaClientError::Json(e), pid_at_request)
                    .await);
            }
        };
        if let Some(error) = parsed.error {
            return Err(self
                .fail(take(accum), server_error(error), pid_at_request)
                .await);
        }
        if let Some(usage) = parsed.usage {
            accum.prompt_tokens = Some(usage.prompt_tokens);
        }
        let Some(choice) = parsed.choices.into_iter().next() else {
            return Ok(());
        };
        if let Some(reason) = choice.finish_reason {
            accum.finish_reason = Some(reason);
        }
        for delta in choice.delta.tool_calls.unwrap_or_default() {
            accum.merge_tool_call_delta(delta);
        }
        if let Some(text) = choice.delta.reasoning_content
            && !text.is_empty()
        {
            forward_reasoning(tx, text, accum).await?;
        }
        if let Some(text) = choice.delta.content
            && !text.is_empty()
        {
            for segment in accum.splitter.feed(&text) {
                forward_segment(tx, segment, accum).await?;
            }
        }
        warn_when_compaction_nears(tx, accum).await;
        Ok(())
    }
}

#[async_trait]
impl TurnBackend for LlamaClient {
    async fn push_user(&self, text: String, attachments: Vec<Attachment>) -> LlmResult<()> {
        let mut history = self.history.lock().await;
        if attachments.is_empty() {
            history.push_user(text);
        } else {
            history.push_user_with_attachments(text, attachments);
        }
        Ok(())
    }

    async fn push_tool_results(&self, results: Vec<ToolResultPayload>) -> LlmResult<()> {
        let mut history = self.history.lock().await;
        for result in results {
            if result.attachments.is_empty() {
                history.push_tool_result(result.call_id, result.content);
            } else {
                history.push_tool_result_with_attachments(
                    result.name,
                    result.content,
                    result.attachments,
                );
            }
        }
        Ok(())
    }

    async fn step(&self, tools: Vec<Value>, tx: mpsc::Sender<LlmEvent>) -> LlmResult<StepOutcome> {
        let mut outcome = self.attempt_step(&tools, &tx).await?;
        if outcome.is_context_overflow() {
            self.truncate_after_overflow(&tx).await;
            outcome = self.attempt_step(&tools, &tx).await?;
        }
        let mut history = self.history.lock().await;
        match outcome {
            StreamOutcome::Ok(accum) | StreamOutcome::ClientDisconnected(accum) => {
                commit_step(&mut history, *accum)
            }
            StreamOutcome::PreEmitError(e) | StreamOutcome::PartialAfterEmit { err: e, .. } => {
                history.rollback_last_user();
                Err(LlmError::Client(e))
            }
            StreamOutcome::ServerRestart { accum, pre_emit } => {
                let bytes_so_far = accum.text.len();
                let tool_builders = accum.tool_calls.len();
                drop(accum);
                Err(LlmError::ServerRestarting(format!(
                    "llama-server died mid-{} ({} bytes / {} tool-call builders streamed)",
                    if pre_emit { "request" } else { "response" },
                    bytes_so_far,
                    tool_builders,
                )))
            }
        }
    }

    async fn push_interrupted_reply(&self, reply: String) -> LlmResult<()> {
        let mut history = self.history.lock().await;
        history.consume_transient_note();
        history.push_assistant(reply);
        Ok(())
    }

    async fn set_transient_note(&self, text: String) -> LlmResult<()> {
        let mut history = self.history.lock().await;
        history.set_transient_note(text);
        Ok(())
    }
}

#[async_trait]
impl LlmBackend for LlamaClient {
    async fn set_transient_context(&self, text: String) -> LlmResult<()> {
        let mut history = self.history.lock().await;
        history.set_transient_context(text);
        Ok(())
    }

    async fn replace_history(&self, replay: HistoryReplay) -> LlmResult<()> {
        let folded_steps: Vec<_> = replay
            .folded_steps
            .into_iter()
            .map(history_message)
            .collect();
        let messages = replay.entries.into_iter().map(history_message).collect();
        let mut history = self.history.lock().await;
        history.replace_messages(replay.summary.as_deref(), &folded_steps, messages);
        Ok(())
    }

    async fn truncate_to_last_real_user(&self) -> LlmResult<usize> {
        let mut history = self.history.lock().await;
        Ok(history.truncate_to_last_real_user())
    }

    async fn complete_oneshot(&self, prompt: String, reply: OneShotReply) -> LlmResult<String> {
        let body_bytes = {
            let mut payload = self.base_request(vec![wire::ChatMessage::plain(
                "user",
                wire::ContentBody::Text(neutralise_template_tokens(&prompt)),
            )]);
            payload.max_tokens = self.chat.max_summary_tokens();
            payload.id_slot = None;
            match &reply {
                OneShotReply::Text(thinking) => {
                    payload.reasoning_budget_tokens = Some(
                        self.chat
                            .max_reasoning_tokens
                            .get()
                            .min(payload.max_tokens / 2),
                    );
                    payload.chat_template_kwargs = self.template_kwargs(*thinking);
                }
                OneShotReply::Json(schema) => {
                    self.sample_greedily_without_thinking(&mut payload);
                    payload.response_format = Some(wire::ResponseFormat::json_schema(schema));
                }
                OneShotReply::OneOf(words) => {
                    self.sample_greedily_without_thinking(&mut payload);
                    payload.grammar = Some(one_word_grammar(words));
                }
            }
            serde_json::to_vec(&payload).map_err(|e| LlmError::Client(LlamaClientError::Json(e)))?
        };
        let request = self.stream_text(body_bytes, "complete_oneshot");
        match self.shared_slot() {
            Some((stash, _)) => stash.preserve_across(request).await,
            None => request.await,
        }
    }

    async fn recent_user_messages(&self, limit: usize) -> Vec<String> {
        self.history.lock().await.recent_user_messages(limit)
    }

    async fn complete_with_schema(
        &self,
        instruction: String,
        schema: Map<String, Value>,
    ) -> LlmResult<String> {
        let body_bytes = {
            let history = self.history.lock().await;
            let mut messages = history.as_wire_messages();
            messages.push(wire::ChatMessage::plain(
                Role::User.as_wire(),
                wire::ContentBody::Text(instruction.as_str().into()),
            ));
            let mut payload = self.base_request(messages);
            payload.chat_template_kwargs = self.template_kwargs(Thinking::Disabled);
            payload.reasoning_budget_tokens = None;
            payload.reasoning_budget_message = None;
            payload.response_format = Some(wire::ResponseFormat::json_schema(&schema));
            serde_json::to_vec(&payload).map_err(|e| LlmError::Client(LlamaClientError::Json(e)))?
        };
        self.claim_slot().await;
        self.stream_text(body_bytes, "complete_with_schema").await
    }

    fn isolated_backend(
        &self,
        system_prompt: String,
        slot: SlotAssignment,
        limits: ReplyLimits,
    ) -> LlmResult<Arc<dyn LlmBackend>> {
        let chat = ChatConfig {
            system_prompt: system_prompt.clone(),
            max_response_tokens: limits.reply_tokens.unwrap_or(self.chat.max_response_tokens),
            max_reasoning_tokens: limits
                .reasoning_tokens
                .unwrap_or(self.chat.max_reasoning_tokens),
            ..self.chat.clone()
        };
        Ok(Arc::new(Self {
            client: self.client.clone(),
            base_url: self.base_url.clone(),
            chat,
            model: self.model.clone(),
            timeouts: self.timeouts.clone(),
            history: Mutex::new(History::new(system_prompt)),
            health: self.health.clone(),
            slot: parking_lot::Mutex::new(self.slot.lock().assign(slot)),
        }))
    }

    fn move_to_slot(&self, slot: SlotAssignment) {
        let mut binding = self.slot.lock();
        *binding = binding.assign(slot);
    }
}

/// The slot a client's requests use.
#[derive(Debug)]
enum SlotBinding {
    Unassigned,
    /// The shared slot, its state kept in `stash` under `holder` while others use it.
    Shared {
        stash: Arc<SlotStash>,
        holder: SlotHolder,
    },
    Pinned(u32),
}

impl SlotBinding {
    /// The binding `assignment` names for a client on this one's server; a shared
    /// slot without a stash is left unassigned.
    fn assign(&self, assignment: SlotAssignment) -> Self {
        match (assignment, self) {
            (SlotAssignment::Shared(holder), SlotBinding::Shared { stash, .. }) => {
                SlotBinding::Shared {
                    stash: Arc::clone(stash),
                    holder,
                }
            }
            (SlotAssignment::Pinned(index), _) => SlotBinding::Pinned(index),
            _ => SlotBinding::Unassigned,
        }
    }

    fn index(&self) -> Option<u32> {
        match self {
            SlotBinding::Unassigned => None,
            SlotBinding::Shared { .. } => Some(SHARED_SLOT),
            SlotBinding::Pinned(index) => Some(*index),
        }
    }
}

/// Summarizes by continuing the live history with the request's own
/// tools and template settings, so llama-server reuses its cached prompt; a
/// zero thinking budget skips thinking without changing the template.
struct ContinuationSummarizer<'a> {
    client: &'a LlamaClient,
    tools: &'a [Value],
}

#[async_trait]
impl Summarizer for ContinuationSummarizer<'_> {
    /// Ask with the request's tools first, so the cached prompt is reused; a
    /// reply that calls a tool or is empty is asked again with no tools offered.
    async fn summarize(&self, request: SummaryRequest<'_>) -> Result<String, LlamaClientError> {
        self.client.claim_slot().await;
        let SummaryRequest {
            mut messages,
            instruction,
        } = request;
        messages.push(wire::ChatMessage::plain(
            Role::User.as_wire(),
            wire::ContentBody::Text(instruction.as_str().into()),
        ));
        match self.request_summary(messages.clone(), self.tools).await {
            Err(LlamaClientError::Summarize(reason)) if !self.tools.is_empty() => {
                warn!(
                    target: "assistd::completion",
                    %reason,
                    "summary attempt failed; asking again without tools"
                );
                self.request_summary(messages, &[]).await
            }
            outcome => outcome,
        }
    }
}

impl ContinuationSummarizer<'_> {
    async fn request_summary(
        &self,
        messages: Vec<wire::ChatMessage<'_>>,
        tools: &[Value],
    ) -> Result<String, LlamaClientError> {
        let client = self.client;
        let mut payload = client.base_request(messages);
        offer_tools(&mut payload, tools.to_vec());
        payload.temperature = SUMMARY_TEMPERATURE;
        payload.max_tokens = client.chat.max_summary_tokens();
        payload.reasoning_budget_tokens = Some(0);
        payload.reasoning_budget_message = None;
        let body = serde_json::to_vec(&payload)?;
        match client.stream_unobserved(body).await {
            StreamOutcome::Ok(accum) | StreamOutcome::ClientDisconnected(accum) => {
                summary_text(*accum)
            }
            StreamOutcome::PreEmitError(err) | StreamOutcome::PartialAfterEmit { err, .. } => {
                Err(err)
            }
            StreamOutcome::ServerRestart { pre_emit: true, .. } => Err(LlamaClientError::NotReady),
            StreamOutcome::ServerRestart { .. } => Err(LlamaClientError::Restarted),
        }
    }
}

/// The GBNF grammar that admits exactly one of `words`.
fn one_word_grammar(words: &[&str]) -> String {
    let alternatives: Vec<String> = words.iter().map(|word| format!("\"{word}\"")).collect();
    format!("root ::= {}", alternatives.join(" | "))
}

/// Offer `tools` on `payload`, leaving it tool-free when there are none.
fn offer_tools(payload: &mut wire::ChatRequest<'_>, tools: Vec<Value>) {
    if !tools.is_empty() {
        payload.tools = Some(tools);
        payload.tool_choice = Some("auto");
    }
}

/// The summary a finished stream holds, refusing a reply that called a tool or said nothing.
fn summary_text(reply: StreamAccum) -> Result<String, LlamaClientError> {
    let text = reply.text;
    if !reply.tool_calls.is_empty() || text.contains(TOOL_CALL_MARKUP) {
        return Err(LlamaClientError::Summarize(
            "the model called a tool instead of summarizing".into(),
        ));
    }
    if text.trim().is_empty() {
        return Err(LlamaClientError::Summarize(
            "summarizer returned empty text".into(),
        ));
    }
    Ok(text)
}

/// Running accumulator for one `stream_openai` call.
#[derive(Debug, Default)]
struct StreamAccum {
    text: String,
    reasoning: String,
    /// Keyed by the model's `index` so finalization keeps emission order.
    tool_calls: BTreeMap<u32, ToolCallBuilder>,
    finish_reason: Option<String>,
    prompt_tokens: Option<u32>,
    has_emitted: bool,
    splitter: ThinkSplitter,
    /// Generated tokens left before compaction is near; `None` once
    /// announced or when not watched.
    compaction_headroom: Option<NonZeroU32>,
}

impl StreamAccum {
    /// Whether anything reached the consumer or a tool call is in progress.
    fn has_output(&self) -> bool {
        self.has_emitted || !self.tool_calls.is_empty()
    }

    /// Bytes streamed so far that history will keep.
    fn streamed_bytes(&self) -> usize {
        let tool_call_bytes: usize = self
            .tool_calls
            .values()
            .map(|call| call.id.len() + call.name.len() + call.arguments.len())
            .sum();
        self.text.len() + self.reasoning.len() + tool_call_bytes
    }

    fn merge_tool_call_delta(&mut self, delta: wire::ToolCallDelta) {
        let entry = self.tool_calls.entry(delta.index).or_default();
        if let Some(id) = delta.id {
            entry.id = id;
        }
        if let Some(function) = delta.function {
            if let Some(name) = function.name {
                entry.name = name;
            }
            if let Some(args) = function.arguments {
                entry.arguments.push_str(&args);
            }
        }
    }

    fn finalize_tool_calls(self) -> LlmResult<(Vec<ToolCallRecord>, Vec<ToolCall>)> {
        let mut records = Vec::with_capacity(self.tool_calls.len());
        let mut parsed = Vec::with_capacity(self.tool_calls.len());
        for (index, builder) in self.tool_calls {
            if builder.name.is_empty() {
                return Err(LlmError::ToolCallParse(format!(
                    "tool call at index {index} has no name"
                )));
            }
            let id = if builder.id.is_empty() {
                format!("call-{index}")
            } else {
                builder.id
            };
            let arguments_json = if builder.arguments.is_empty() {
                "{}".to_string()
            } else {
                builder.arguments
            };
            let arguments_value = serde_json::from_str::<Value>(&arguments_json).map_err(|e| {
                LlmError::ToolCallParse(format!("tool call {id}: malformed arguments JSON: {e}"))
            })?;
            records.push(ToolCallRecord {
                id: id.clone(),
                name: builder.name.clone(),
                arguments: arguments_json,
            });
            parsed.push(ToolCall {
                id,
                name: builder.name,
                arguments: arguments_value,
            });
        }
        Ok((records, parsed))
    }
}

#[derive(Debug, Default)]
struct ToolCallBuilder {
    id: String,
    name: String,
    arguments: String,
}

enum StreamOutcome {
    /// Stream ended with a `[DONE]` marker or a finish reason.
    Ok(Box<StreamAccum>),
    /// Stream failed after deltas were forwarded; `accum` holds what arrived.
    PartialAfterEmit {
        accum: Box<StreamAccum>,
        err: LlamaClientError,
    },
    /// The consumer dropped the receiver mid-stream; stop quietly.
    ClientDisconnected(Box<StreamAccum>),
    /// Stream errored before any deltas were forwarded; propagate as `Err`.
    PreEmitError(LlamaClientError),
    /// The failure coincided with a supervisor restart. `pre_emit` is
    /// true when nothing had been streamed to the consumer yet.
    ServerRestart {
        accum: Box<StreamAccum>,
        pre_emit: bool,
    },
}

impl StreamOutcome {
    fn is_context_overflow(&self) -> bool {
        matches!(
            self,
            Self::PreEmitError(LlamaClientError::ContextOverflow { .. })
        )
    }
}

/// The events telling the consumer compaction finished, led by the summary
/// now heading the history when there is one.
fn compaction_events(history: &History, started_at: Instant) -> Vec<LlmEvent> {
    let summarized = history
        .summary_snapshot()
        .map(|(summary, cut)| LlmEvent::HistoryCompacted {
            summary: summary.to_string(),
            cut,
            elapsed: started_at.elapsed(),
        });
    let finished = compaction_status(StatusKind::HistoryCompacted, "history compacted");
    summarized.into_iter().chain([finished]).collect()
}

fn context_size(history: &History) -> LlmEvent {
    LlmEvent::ContextSize {
        tokens: history.approx_total_tokens(),
        total: history.total_tokens(),
    }
}

fn compaction_status(event: StatusKind, message: &str) -> LlmEvent {
    LlmEvent::Status {
        severity: StatusSeverity::Info,
        component: Component::Llm,
        event,
        message: message.into(),
    }
}

async fn send_compaction_status(tx: &mpsc::Sender<LlmEvent>, event: StatusKind, message: &str) {
    let _ = tx.send(compaction_status(event, message)).await;
}

async fn send_events(tx: &mpsc::Sender<LlmEvent>, events: Vec<LlmEvent>) {
    for event in events {
        let _ = tx.send(event).await;
    }
}

async fn warn_when_compaction_nears(tx: &mpsc::Sender<LlmEvent>, accum: &mut StreamAccum) {
    let Some(headroom) = accum.compaction_headroom else {
        return;
    };
    if approx_tokens_bytes(accum.streamed_bytes()) < headroom.get() {
        return;
    }
    accum.compaction_headroom = None;
    send_compaction_status(tx, StatusKind::CompactingSoon, "compacting soon").await;
}

async fn forward_segment(
    tx: &mpsc::Sender<LlmEvent>,
    segment: Segment,
    accum: &mut StreamAccum,
) -> Result<(), StreamOutcome> {
    match segment {
        Segment::Reasoning(text) => forward_reasoning(tx, text, accum).await,
        Segment::Visible(text) => {
            if text.is_empty() {
                return Ok(());
            }
            if !accum.has_emitted {
                debug!(
                    target: "assistd::voice::latency",
                    stage = "llm_first_token",
                    "voice latency stage"
                );
            }
            accum.text.push_str(&text);
            accum.has_emitted = true;
            forward(tx, LlmEvent::Delta { text }, accum).await
        }
    }
}

async fn forward_reasoning(
    tx: &mpsc::Sender<LlmEvent>,
    text: String,
    accum: &mut StreamAccum,
) -> Result<(), StreamOutcome> {
    accum.reasoning.push_str(&text);
    forward(tx, LlmEvent::ReasoningDelta { text }, accum).await
}

async fn forward(
    tx: &mpsc::Sender<LlmEvent>,
    event: LlmEvent,
    accum: &mut StreamAccum,
) -> Result<(), StreamOutcome> {
    if tx.send(event).await.is_err() {
        debug!(target: "assistd::completion", "client disconnected mid-stream");
        return Err(StreamOutcome::ClientDisconnected(Box::new(take(accum))));
    }
    Ok(())
}

/// Record the step in `history` and consume its transient note, which a failed
/// step leaves for the retry. Tool calls cut off before the finish chunk are
/// discarded rather than run half-built; a token-limit cutoff is not recorded.
fn commit_step(history: &mut History, mut accum: StreamAccum) -> LlmResult<StepOutcome> {
    history.consume_transient_note();
    if accum.finish_reason.as_deref() == Some("length") {
        warn!(
            target: "assistd::completion",
            text_bytes = accum.text.len(),
            reasoning_bytes = accum.reasoning.len(),
            tool_calls = accum.tool_calls.len(),
            "response hit max_tokens; discarding it"
        );
        return Ok(StepOutcome::Truncated);
    }
    if accum.tool_calls.is_empty() {
        history.push_assistant(accum.text);
        return Ok(StepOutcome::Final);
    }
    if accum.finish_reason.is_none() {
        return Err(LlmError::ToolCallParse(format!(
            "stream ended before its {} tool call(s) were complete; discarded",
            accum.tool_calls.len()
        )));
    }
    if accum.finish_reason.as_deref() != Some("tool_calls") {
        warn!(
            target: "assistd::completion",
            finish_reason = accum.finish_reason.as_deref().unwrap_or("<none>"),
            tool_calls = accum.tool_calls.len(),
            "finish_reason disagrees with emitted tool calls; running them anyway"
        );
    }
    let narration = take(&mut accum.text);
    let reasoning = take(&mut accum.reasoning);
    let (records, parsed) = accum.finalize_tool_calls()?;
    history.push_assistant_with_tool_calls(
        (!narration.trim().is_empty()).then_some(narration),
        reasoning,
        records,
    );
    Ok(StepOutcome::ToolCalls(parsed))
}

fn server_error(error: wire::StreamError) -> LlamaClientError {
    if error.kind.as_deref() == Some(CONTEXT_OVERFLOW_ERROR) {
        return LlamaClientError::ContextOverflow {
            prompt_tokens: error.n_prompt_tokens,
        };
    }
    LlamaClientError::Server {
        status: error.code,
        body: error.message,
    }
}

/// The failure a non-success response reports, keeping its raw body unless
/// it is a context overflow.
fn http_error(status: u16, body: String) -> LlamaClientError {
    match serde_json::from_str::<wire::ErrorBody>(&body).map(|parsed| server_error(parsed.error)) {
        Ok(overflow @ LlamaClientError::ContextOverflow { .. }) => overflow,
        _ => LlamaClientError::Server { status, body },
    }
}

/// The failure an `error:` event reports, keeping its raw payload when it
/// is not llama-server's error object.
fn error_event(payload: &str) -> LlamaClientError {
    serde_json::from_str(payload).map_or_else(
        |_| LlamaClientError::Sse(format!("error event: {payload}")),
        server_error,
    )
}

fn history_message(entry: HistoryEntry) -> Message {
    match entry.role {
        HistoryRole::User => Message::User(UserMessage {
            content: entry.content,
            ..UserMessage::default()
        }),
        HistoryRole::Assistant => Message::Assistant(AssistantMessage {
            content: entry.content,
            reasoning: entry.reasoning,
            tool_calls: entry.tool_calls,
        }),
        HistoryRole::Tool => match entry.tool_call_id {
            Some(call_id) => Message::ToolResult {
                call_id,
                content: entry.content,
            },
            None => Message::ToolImageResult {
                tool_name: entry.tool_name.unwrap_or_default(),
                content: entry.content,
                attachments: Vec::new(),
            },
        },
    }
}

async fn flush_splitter(accum: &mut StreamAccum, tx: &mpsc::Sender<LlmEvent>) {
    match accum.splitter.finish() {
        Some(Segment::Reasoning(text)) => {
            accum.reasoning.push_str(&text);
            let _ = tx.send(LlmEvent::ReasoningDelta { text }).await;
        }
        Some(Segment::Visible(text)) if !text.is_empty() => {
            accum.text.push_str(&text);
            accum.has_emitted = true;
            let _ = tx.send(LlmEvent::Delta { text }).await;
        }
        _ => {}
    }
}

async fn read_body_capped(response: &mut reqwest::Response, cap: usize, idle: Duration) -> String {
    let mut buf = Vec::new();
    while let Ok(Ok(Some(chunk))) = timeout(idle, response.chunk()).await {
        let remaining = cap.saturating_sub(buf.len());
        if remaining == 0 {
            buf.extend_from_slice(b"...<truncated>");
            break;
        }
        let take = chunk.len().min(remaining);
        buf.extend_from_slice(&chunk[..take]);
        if take < chunk.len() {
            buf.extend_from_slice(b"...<truncated>");
            break;
        }
    }
    String::from_utf8_lossy(&buf).into_owned()
}
