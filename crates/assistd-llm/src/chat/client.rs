//! HTTP streaming chat client for the locally-managed llama-server.
//!
//! The conversation mutex is held only across the cheap state-mutation
//! phases before and after a request; the HTTP stream itself runs
//! lock-free, so a hung server never blocks a concurrent `push_user` or
//! `set_transient_context`.

use std::collections::BTreeMap;
use std::mem::take;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;

use assistd_config::{ChatConfig, LlamaServerConfig, ModelConfig, TimeoutsConfig};
use assistd_tools::Attachment;
use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::{Mutex, mpsc};
use tokio::time::timeout;
use tracing::{debug, warn};

use super::conversation::{Conversation, Summarizer, ToolCallRecord};
use super::conversation::{Message, Role};
use super::error::ChatClientError;
use super::sse::{SseEvent, SseLineReader};
use super::think_splitter::{Segment, ThinkSplitter};
use super::wire;
use crate::{
    HistoryEntry, HistoryRole, LlmBackend, LlmError, LlmEvent, LlmHealthProbe, LlmResult,
    ReadyState, StepOutcome, Thinking, ToolCall, ToolResultPayload,
};

const ERROR_BODY_CAP: usize = 1024;
const SUMMARY_SYSTEM_PROMPT: &str = "You are a conversation summarizer. Produce a concise \
    summary of the following dialogue that preserves all factual claims, user requests, and \
    assistant conclusions. Write in past tense. Do not add commentary.";

/// HTTP streaming chat client backed by a locally-managed llama-server.
pub struct LlamaChatClient {
    client: reqwest::Client,
    base_url: String,
    chat: ChatConfig,
    model: ModelConfig,
    timeouts: TimeoutsConfig,
    conv: Mutex<Conversation>,
    /// Without a probe every HTTP failure is a transport fault; with
    /// one, a failure that coincides with a supervisor restart becomes
    /// [`LlmError::ServerRestarting`] so the caller can replay.
    health: Option<Arc<dyn LlmHealthProbe>>,
}

impl LlamaChatClient {
    /// Build a client for the server at `server.host:server.port`. Pass
    /// `health: None` when no supervisor is attached.
    pub fn new(
        chat: &ChatConfig,
        server: &LlamaServerConfig,
        model: &ModelConfig,
        timeouts: &TimeoutsConfig,
        health: Option<Arc<dyn LlmHealthProbe>>,
    ) -> Result<Self, ChatClientError> {
        let client = reqwest::Client::builder()
            .no_proxy()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(chat.request_timeout_secs.get()))
            .build()?;
        let base_url = format!("http://{}:{}", server.host, server.port);
        let conv = Conversation::new(chat.system_prompt.clone());
        Ok(Self {
            client,
            base_url,
            chat: chat.clone(),
            model: model.clone(),
            timeouts: timeouts.clone(),
            conv: Mutex::new(conv),
            health,
        })
    }

    /// A request carrying the configured sampling parameters and no
    /// tools; callers set what differs.
    fn base_request<'a>(&'a self, messages: Vec<wire::ChatMessage<'a>>) -> wire::ChatRequest<'a> {
        wire::ChatRequest {
            model: self.model.name.as_str(),
            messages,
            stream: true,
            temperature: self.chat.temperature,
            max_tokens: self.chat.max_response_tokens.get(),
            top_p: self.chat.top_p,
            top_k: self.chat.top_k.map(NonZeroU32::get),
            min_p: self.chat.min_p,
            presence_penalty: self.chat.presence_penalty,
            tools: None,
            tool_choice: None,
            chat_template_kwargs: None,
        }
    }

    /// Classify a failed request: a restart-coincident failure becomes
    /// [`StreamOutcome::ServerRestart`]; anything else keeps whatever
    /// was already streamed or propagates `err` if nothing was.
    fn fail(
        &self,
        accum: StreamAccum,
        err: ChatClientError,
        pid_at_request: Option<u32>,
    ) -> StreamOutcome {
        if self.looks_like_server_crash(pid_at_request) {
            let pre_emit = !accum.has_output();
            return StreamOutcome::ServerRestart { accum, pre_emit };
        }
        if accum.has_output() {
            warn!(
                target: "assistd::chat",
                "mid-stream error after {} bytes / {} tool-call builders: {err}",
                accum.text.len(),
                accum.tool_calls.len()
            );
            StreamOutcome::PartialAfterEmit(accum)
        } else {
            StreamOutcome::PreEmitError(err)
        }
    }

    /// Whether an HTTP failure coincides with a supervisor restart: the
    /// child's pid changed or vanished since the request was sent, or
    /// the readiness state left `Ready`. Always false without a probe.
    fn looks_like_server_crash(&self, pid_at_request: Option<u32>) -> bool {
        let Some(probe) = self.health.as_ref() else {
            return false;
        };
        let current_pid = probe.pid();
        let current_state = probe.state();
        let pid_changed = match (pid_at_request, current_pid) {
            (Some(_), None) => true,
            (Some(a), Some(b)) => a != b,
            _ => false,
        };
        let state_unhealthy = !matches!(current_state, Some(ReadyState::Ready) | None);
        pid_changed || state_unhealthy
    }

    async fn stream_openai(&self, body: Vec<u8>, tx: &mpsc::Sender<LlmEvent>) -> StreamOutcome {
        let pid_at_request = self.health.as_ref().and_then(|h| h.pid());
        tracing::debug!(
            target: "assistd::voice::latency",
            stage = "llm_request_sent",
            "voice latency stage"
        );
        let mut response = match self.send_request(body, pid_at_request).await {
            Ok(response) => response,
            Err(outcome) => return outcome,
        };
        let mut accum = StreamAccum::default();
        let saw_done = match self
            .read_stream(&mut response, &mut accum, tx, pid_at_request)
            .await
        {
            Ok(saw_done) => saw_done,
            Err(outcome) => return outcome,
        };
        if !saw_done {
            warn!(
                target: "assistd::chat",
                "stream ended before [DONE] marker; accumulated {} bytes text, {} tool-call builders",
                accum.text.len(),
                accum.tool_calls.len()
            );
        }
        match accum.splitter.finish() {
            Some(Segment::Reasoning(text)) => {
                let _ = tx.send(LlmEvent::ReasoningDelta { text }).await;
            }
            Some(Segment::Visible(text)) if !text.is_empty() => {
                accum.text.push_str(&text);
                accum.has_emitted = true;
                let _ = tx.send(LlmEvent::Delta { text }).await;
            }
            _ => {}
        }
        StreamOutcome::Ok(accum)
    }

    /// POST the request and return the response once it is known to be
    /// a success from a server that has not restarted underneath us.
    async fn send_request(
        &self,
        body: Vec<u8>,
        pid_at_request: Option<u32>,
    ) -> Result<reqwest::Response, StreamOutcome> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let mut response = match self
            .client
            .post(&url)
            .header("Accept", "text/event-stream")
            .header("Content-Type", "application/json")
            .body(body)
            .send()
            .await
        {
            Ok(response) => response,
            Err(e) => {
                return Err(self.fail(
                    StreamAccum::default(),
                    ChatClientError::Http(e),
                    pid_at_request,
                ));
            }
        };
        if self.looks_like_server_crash(pid_at_request) {
            // A 200 that raced the supervisor's teardown.
            return Err(StreamOutcome::ServerRestart {
                accum: StreamAccum::default(),
                pre_emit: true,
            });
        }
        let status = response.status();
        if !status.is_success() {
            let body = read_body_capped(&mut response, ERROR_BODY_CAP).await;
            let err = ChatClientError::Server {
                status: status.as_u16(),
                body,
            };
            return Err(self.fail(StreamAccum::default(), err, pid_at_request));
        }
        Ok(response)
    }

    /// Drive the SSE stream until `[DONE]` or EOF, forwarding events
    /// through `tx`. Returns whether `[DONE]` was seen.
    async fn read_stream(
        &self,
        response: &mut reqwest::Response,
        accum: &mut StreamAccum,
        tx: &mpsc::Sender<LlmEvent>,
        pid_at_request: Option<u32>,
    ) -> Result<bool, StreamOutcome> {
        let mut reader = SseLineReader::new();
        let first_byte = Duration::from_secs(self.chat.request_timeout_secs.get());
        let inter_chunk = Duration::from_secs(self.timeouts.stream_inactivity_secs);
        let mut saw_bytes = false;
        loop {
            let deadline = if saw_bytes { inter_chunk } else { first_byte };
            let chunk = match timeout(deadline, response.chunk()).await {
                Ok(Ok(Some(chunk))) => chunk,
                Ok(Ok(None)) => return Ok(false),
                Ok(Err(e)) => {
                    return Err(self.fail(take(accum), ChatClientError::Http(e), pid_at_request));
                }
                Err(_) => {
                    warn!(
                        target: "assistd::chat",
                        timeout_secs = deadline.as_secs(),
                        phase = if saw_bytes { "inter_chunk" } else { "first_byte" },
                        bytes_so_far = accum.text.len(),
                        tool_call_builders = accum.tool_calls.len(),
                        "SSE stream inactive past deadline; aborting read"
                    );
                    let err = ChatClientError::Sse(format!(
                        "no bytes received for {}s",
                        deadline.as_secs()
                    ));
                    return Err(self.fail(take(accum), err, pid_at_request));
                }
            };
            saw_bytes = true;
            reader.feed(&chunk);
            loop {
                match reader.next_event() {
                    Ok(Some(SseEvent::Data(payload))) => {
                        self.handle_chunk(&payload, accum, tx, pid_at_request)
                            .await?;
                    }
                    Ok(Some(SseEvent::Done)) => return Ok(true),
                    Ok(None) => break,
                    Err(e) => return Err(self.fail(take(accum), e, pid_at_request)),
                }
            }
        }
    }

    /// Fold one `data:` payload into `accum`, forwarding its deltas.
    async fn handle_chunk(
        &self,
        payload: &str,
        accum: &mut StreamAccum,
        tx: &mpsc::Sender<LlmEvent>,
        pid_at_request: Option<u32>,
    ) -> Result<(), StreamOutcome> {
        let parsed: wire::ChatCompletionChunk = match serde_json::from_str(payload) {
            Ok(parsed) => parsed,
            Err(e) => return Err(self.fail(take(accum), ChatClientError::Json(e), pid_at_request)),
        };
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
            forward(tx, LlmEvent::ReasoningDelta { text }, accum).await?;
        }
        if let Some(text) = choice.delta.content
            && !text.is_empty()
        {
            for segment in accum.splitter.feed(&text) {
                forward_segment(tx, segment, accum).await?;
            }
        }
        Ok(())
    }
}

/// Send one classified segment, recording visible text on `accum`.
async fn forward_segment(
    tx: &mpsc::Sender<LlmEvent>,
    segment: Segment,
    accum: &mut StreamAccum,
) -> Result<(), StreamOutcome> {
    match segment {
        Segment::Reasoning(text) => forward(tx, LlmEvent::ReasoningDelta { text }, accum).await,
        Segment::Visible(text) => {
            if text.is_empty() {
                return Ok(());
            }
            if !accum.has_emitted {
                tracing::debug!(
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

/// Send `event`, or hand back everything accumulated so far when the
/// consumer has gone away.
async fn forward(
    tx: &mpsc::Sender<LlmEvent>,
    event: LlmEvent,
    accum: &mut StreamAccum,
) -> Result<(), StreamOutcome> {
    if tx.send(event).await.is_err() {
        debug!(target: "assistd::chat", "client disconnected mid-stream");
        return Err(StreamOutcome::ClientDisconnected(take(accum)));
    }
    Ok(())
}

#[async_trait]
impl LlmBackend for LlamaChatClient {
    async fn generate(&self, prompt: String, tx: mpsc::Sender<LlmEvent>) -> LlmResult<()> {
        let body_bytes = {
            let lock_start = std::time::Instant::now();
            let mut conv = self.conv.lock().await;
            if lock_start.elapsed() > Duration::from_secs(1) {
                warn!(
                    target: "assistd::chat",
                    "chat lock contended for {:?}",
                    lock_start.elapsed()
                );
            }
            conv.push_user(prompt);
            if let Err(e) = conv.ensure_budget(self, &self.chat, &self.model).await {
                warn!(
                    target: "assistd::chat",
                    "ensure_budget failed ({e}); falling back to truncation"
                );
                conv.truncate_to_budget(&self.chat, &self.model);
            }
            let payload = self.base_request(conv.as_wire_messages());
            match serde_json::to_vec(&payload) {
                Ok(b) => b,
                Err(e) => {
                    conv.rollback_last_user();
                    return Err(LlmError::Chat(ChatClientError::Json(e)));
                }
            }
        };

        let outcome = self.stream_openai(body_bytes, &tx).await;

        let mut conv = self.conv.lock().await;
        match outcome {
            StreamOutcome::Ok(accum) => {
                conv.push_assistant(accum.text);
                let _ = tx.send(LlmEvent::Done).await;
                Ok(())
            }
            StreamOutcome::PartialAfterEmit(accum) => {
                conv.push_assistant(accum.text);
                let _ = tx.send(LlmEvent::Done).await;
                Ok(())
            }
            StreamOutcome::ClientDisconnected(accum) => {
                conv.push_assistant(accum.text);
                Ok(())
            }
            StreamOutcome::PreEmitError(e) => {
                conv.rollback_last_user();
                Err(LlmError::Chat(e))
            }
            StreamOutcome::ServerRestart { .. } => Err(LlmError::ServerRestarting(
                "llama-server crashed during generate".into(),
            )),
        }
    }

    async fn push_user(&self, text: String, attachments: Vec<Attachment>) -> LlmResult<()> {
        let mut conv = self.conv.lock().await;
        if attachments.is_empty() {
            conv.push_user(text);
        } else {
            conv.push_user_with_attachments(text, attachments);
        }
        Ok(())
    }

    async fn push_tool_results(&self, results: Vec<ToolResultPayload>) -> LlmResult<()> {
        let mut conv = self.conv.lock().await;
        for r in results {
            if r.attachments.is_empty() {
                conv.push_tool_result(r.call_id, r.content);
            } else {
                // Image parts only render on a user turn, so a result
                // carrying one keeps the tagged user-message shape.
                let content = format!("[tool:{}]\n{}", r.name, r.content);
                conv.push_user_with_attachments(content, r.attachments);
            }
        }
        Ok(())
    }

    async fn step(&self, tools: Vec<Value>, tx: mpsc::Sender<LlmEvent>) -> LlmResult<StepOutcome> {
        let body_bytes = {
            let mut conv = self.conv.lock().await;
            if let Err(e) = conv.ensure_budget(self, &self.chat, &self.model).await {
                warn!(
                    target: "assistd::chat",
                    "ensure_budget failed ({e}); falling back to truncation"
                );
                conv.truncate_to_budget(&self.chat, &self.model);
            }
            let mut payload = self.base_request(conv.as_wire_messages());
            if !tools.is_empty() {
                payload.tools = Some(tools);
                payload.tool_choice = Some("auto");
            }
            serde_json::to_vec(&payload).map_err(|e| LlmError::Chat(ChatClientError::Json(e)))?
        };

        let outcome = self.stream_openai(body_bytes, &tx).await;

        let mut conv = self.conv.lock().await;
        match outcome {
            StreamOutcome::Ok(accum)
            | StreamOutcome::PartialAfterEmit(accum)
            | StreamOutcome::ClientDisconnected(accum) => {
                let result = commit_step(&mut conv, accum);
                // `PreEmitError` leaves the transient in place so a retry
                // sees the same injected block.
                let _ = conv.consume_transient_context();
                result
            }
            StreamOutcome::PreEmitError(e) => Err(LlmError::Chat(e)),
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

    async fn set_transient_context(&self, text: String) -> LlmResult<()> {
        let mut conv = self.conv.lock().await;
        conv.set_transient_context(text);
        Ok(())
    }

    async fn replace_history(&self, entries: Vec<HistoryEntry>) -> LlmResult<()> {
        let mut msgs = Vec::with_capacity(entries.len());
        for entry in entries {
            match entry.role {
                HistoryRole::System => msgs.push(Message {
                    role: Role::System,
                    content: entry.content,
                    attachments: Vec::new(),
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                }),
                HistoryRole::User => msgs.push(Message {
                    role: Role::User,
                    content: entry.content,
                    attachments: Vec::new(),
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                }),
                HistoryRole::Assistant => {
                    let calls = parse_tool_calls(&entry.tool_calls_json)?;
                    msgs.push(Message {
                        role: Role::Assistant,
                        content: entry.content,
                        attachments: Vec::new(),
                        tool_calls: calls,
                        tool_call_id: None,
                    });
                }
                // A row with no call id was written by the vision path;
                // replaying it as a tool message would leave the template
                // without the id it needs, so it keeps the tagged user shape.
                HistoryRole::Tool => match entry.tool_call_id {
                    Some(call_id) => msgs.push(Message {
                        role: Role::Tool,
                        content: entry.content,
                        attachments: Vec::new(),
                        tool_calls: Vec::new(),
                        tool_call_id: Some(call_id),
                    }),
                    None => {
                        let name = entry.tool_name.unwrap_or_default();
                        msgs.push(Message {
                            role: Role::User,
                            content: format!("[tool:{name}]\n{}", entry.content),
                            attachments: Vec::new(),
                            tool_calls: Vec::new(),
                            tool_call_id: None,
                        });
                    }
                },
            }
        }
        let mut conv = self.conv.lock().await;
        conv.replace_messages(msgs);
        Ok(())
    }

    async fn truncate_to_last_real_user(&self) -> LlmResult<usize> {
        let mut conv = self.conv.lock().await;
        Ok(conv.truncate_to_last_real_user())
    }

    async fn complete_oneshot(&self, prompt: String, thinking: Thinking) -> LlmResult<String> {
        let body_bytes = {
            let mut payload = self.base_request(vec![wire::ChatMessage {
                role: "user",
                content: Some(wire::ContentBody::Text(prompt.as_str())),
                tool_calls: None,
                tool_call_id: None,
            }]);
            payload.max_tokens = self.chat.max_summary_tokens();
            payload.chat_template_kwargs = match thinking {
                Thinking::Enabled => None,
                Thinking::Disabled => Some(wire::ChatTemplateKwargs {
                    enable_thinking: false,
                }),
            };
            serde_json::to_vec(&payload).map_err(|e| LlmError::Chat(ChatClientError::Json(e)))?
        };

        let (tx, mut rx) = mpsc::channel::<LlmEvent>(64);
        let stream = async {
            let outcome = self.stream_openai(body_bytes, &tx).await;
            drop(tx);
            outcome
        };
        let collect = async {
            let mut buf = String::new();
            while let Some(ev) = rx.recv().await {
                if let LlmEvent::Delta { text } = ev {
                    buf.push_str(&text);
                }
            }
            buf
        };
        let (outcome, buf) = tokio::join!(stream, collect);
        match outcome {
            StreamOutcome::Ok(accum)
            | StreamOutcome::PartialAfterEmit(accum)
            | StreamOutcome::ClientDisconnected(accum) => {
                if !accum.text.is_empty() {
                    Ok(accum.text)
                } else {
                    Ok(buf)
                }
            }
            StreamOutcome::PreEmitError(e) => Err(LlmError::Chat(e)),
            StreamOutcome::ServerRestart { .. } => Err(LlmError::ServerRestarting(
                "llama-server crashed during complete_oneshot".into(),
            )),
        }
    }
}

fn parse_tool_calls(json: &Option<Value>) -> LlmResult<Vec<super::conversation::ToolCallRecord>> {
    use super::conversation::ToolCallRecord;
    let Some(value) = json else {
        return Ok(Vec::new());
    };
    let Some(arr) = value.as_array() else {
        return Ok(Vec::new());
    };
    let mut out = Vec::with_capacity(arr.len());
    for entry in arr {
        let id = entry
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let name = entry
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| LlmError::ToolCallParse("history tool_call missing name".into()))?
            .to_string();
        let arguments = match entry.get("arguments") {
            Some(Value::String(s)) => s.clone(),
            Some(other) => other.to_string(),
            None => "{}".to_string(),
        };
        out.push(ToolCallRecord {
            id,
            name,
            arguments,
        });
    }
    Ok(out)
}

fn commit_step(conv: &mut Conversation, mut accum: StreamAccum) -> LlmResult<StepOutcome> {
    if accum.tool_calls.is_empty() {
        conv.push_assistant(accum.text);
        return Ok(StepOutcome::Final);
    }
    if !matches!(accum.finish_reason.as_deref(), None | Some("tool_calls")) {
        warn!(
            target: "assistd::chat",
            finish_reason = accum.finish_reason.as_deref().unwrap_or("<none>"),
            tool_calls = accum.tool_calls.len(),
            "finish_reason disagrees with emitted tool calls; running them anyway"
        );
    }
    let narration = std::mem::take(&mut accum.text);
    let (records, parsed) = accum.finalize_tool_calls()?;
    conv.push_assistant_with_tool_calls(
        (!narration.trim().is_empty()).then_some(narration),
        records,
    );
    Ok(StepOutcome::ToolCalls(parsed))
}

#[async_trait]
impl Summarizer for LlamaChatClient {
    async fn summarize(
        &self,
        dialogue: String,
        _target_tokens: u32,
        max_tokens: u32,
    ) -> Result<String, ChatClientError> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let payload = wire::ChatRequest {
            model: self.model.name.as_str(),
            messages: vec![
                wire::ChatMessage {
                    role: "system",
                    content: Some(wire::ContentBody::Text(SUMMARY_SYSTEM_PROMPT)),
                    tool_calls: None,
                    tool_call_id: None,
                },
                wire::ChatMessage {
                    role: "user",
                    content: Some(wire::ContentBody::Text(&dialogue)),
                    tool_calls: None,
                    tool_call_id: None,
                },
            ],
            stream: false,
            temperature: self.chat.summary_temperature,
            max_tokens,
            top_p: None,
            top_k: None,
            min_p: None,
            presence_penalty: None,
            tools: None,
            tool_choice: None,
            chat_template_kwargs: None,
        };

        let mut response = self.client.post(&url).json(&payload).send().await?;
        let status = response.status();
        if !status.is_success() {
            let body = read_body_capped(&mut response, ERROR_BODY_CAP).await;
            return Err(ChatClientError::Server {
                status: status.as_u16(),
                body,
            });
        }

        let body: wire::ChatResponse = response.json().await?;
        let text = body
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .ok_or_else(|| ChatClientError::Summarize("no choices returned".into()))?;
        Ok(text)
    }
}

/// Running accumulator for one `stream_openai` call.
#[derive(Debug, Default)]
struct StreamAccum {
    text: String,
    /// Keyed by the model's `index` so finalization keeps emission order.
    tool_calls: BTreeMap<u32, ToolCallBuilder>,
    finish_reason: Option<String>,
    has_emitted: bool,
    splitter: ThinkSplitter,
}

impl StreamAccum {
    /// Whether anything reached the consumer or a tool call is being
    /// assembled.
    fn has_output(&self) -> bool {
        self.has_emitted || !self.tool_calls.is_empty()
    }

    fn merge_tool_call_delta(&mut self, delta: wire::ToolCallDelta) {
        let entry = self.tool_calls.entry(delta.index).or_default();
        if let Some(id) = delta.id {
            entry.id = id;
        }
        if let Some(f) = delta.function {
            if let Some(name) = f.name {
                entry.name = name;
            }
            if let Some(args) = f.arguments {
                entry.arguments.push_str(&args);
            }
        }
    }

    fn finalize_tool_calls(self) -> LlmResult<(Vec<ToolCallRecord>, Vec<ToolCall>)> {
        let mut records = Vec::with_capacity(self.tool_calls.len());
        let mut parsed = Vec::with_capacity(self.tool_calls.len());
        for (index, b) in self.tool_calls {
            if b.name.is_empty() {
                return Err(LlmError::ToolCallParse(format!(
                    "tool call at index {index} has no name"
                )));
            }
            let id = if b.id.is_empty() {
                format!("call-{index}")
            } else {
                b.id.clone()
            };
            let arguments_json = if b.arguments.is_empty() {
                "{}".to_string()
            } else {
                b.arguments.clone()
            };
            let arguments_value = serde_json::from_str::<Value>(&arguments_json).map_err(|e| {
                LlmError::ToolCallParse(format!("tool call {id}: malformed arguments JSON: {e}"))
            })?;
            records.push(ToolCallRecord {
                id: id.clone(),
                name: b.name.clone(),
                arguments: arguments_json,
            });
            parsed.push(ToolCall {
                id,
                name: b.name,
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
    /// Stream completed cleanly with a `[DONE]` marker (or EOF after deltas).
    Ok(StreamAccum),
    /// Stream errored after we'd already forwarded deltas; return what we have.
    PartialAfterEmit(StreamAccum),
    /// The consumer dropped the receiver mid-stream; stop quietly.
    ClientDisconnected(StreamAccum),
    /// Stream errored before any deltas were forwarded; propagate as `Err`.
    PreEmitError(ChatClientError),
    /// The failure coincided with a supervisor restart. `pre_emit` is
    /// true when nothing had been streamed to the consumer yet.
    ServerRestart { accum: StreamAccum, pre_emit: bool },
}

async fn read_body_capped(response: &mut reqwest::Response, cap: usize) -> String {
    let mut buf = Vec::new();
    loop {
        match response.chunk().await {
            Ok(Some(chunk)) => {
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
            Ok(None) => break,
            Err(_) => break,
        }
    }
    String::from_utf8_lossy(&buf).into_owned()
}
