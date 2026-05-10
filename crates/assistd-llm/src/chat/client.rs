//! HTTP streaming chat client for the locally-managed llama-server.
//!
//! `LlamaChatClient` implements `LlmBackend`, so the daemon can drop it into
//! `AppState::llm` exactly where `EchoBackend` used to live. A
//! `tokio::sync::Mutex<Conversation>` guards the chat history. The mutex is
//! held only across the cheap state-mutation phases (`push_user`,
//! `ensure_budget`, building the wire payload, and the post-stream
//! `push_assistant`/`rollback`). The HTTP streaming call itself runs
//! lock-free, so a slow or hung server never blocks a concurrent
//! `push_user`/`set_transient_context` call from another caller.
//!
//! Tool-use support comes via three extra `LlmBackend` methods:
//! `push_user`, `push_tool_results`, and `step`. The agent loop in
//! `assistd-core` drives them: `push_user(text)` at the start of a turn,
//! then `step` → handle the outcome → `push_tool_results(...)` if needed →
//! `step` again, until `StepOutcome::Final`.

use std::collections::BTreeMap;
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
use super::wire;
use crate::{
    HistoryEntry, HistoryRole, LlmBackend, LlmError, LlmEvent, LlmHealthProbe, LlmResult,
    ReadyState, StepOutcome, ToolCall, ToolResultPayload,
};

const ERROR_BODY_CAP: usize = 1024;
const SUMMARY_SYSTEM_PROMPT: &str = "You are a conversation summarizer. Produce a concise \
    summary of the following dialogue that preserves all factual claims, user requests, and \
    assistant conclusions. Write in past tense. Do not add commentary.";

/// HTTP streaming chat client backed by a locally-managed llama-server.
///
/// Implements [`crate::LlmBackend`]. A `tokio::sync::Mutex<Conversation>` guards
/// the chat history; the HTTP streaming call itself runs lock-free so a slow
/// server never blocks concurrent state mutations.
pub struct LlamaChatClient {
    client: reqwest::Client,
    base_url: String,
    chat: ChatConfig,
    model: ModelConfig,
    timeouts: TimeoutsConfig,
    conv: Mutex<Conversation>,
    /// Optional handle into the supervisor so we can classify HTTP
    /// errors as crash-induced (server died → reply `ServerRestarting`
    /// so the agent loop can replay) vs. genuine transport faults.
    /// `None` in tests that drive the client without a real supervisor.
    health: Option<Arc<dyn LlmHealthProbe>>,
}

impl LlamaChatClient {
    /// `chat` carries the sampling + history-window knobs, `server` provides
    /// the host:port the request is sent to, and `model` is the identifier
    /// llama-server has loaded (plus its context length for budget math).
    /// All three are cloned; the caller keeps ownership.
    ///
    /// `health` is the optional probe used to detect mid-stream
    /// llama-server crashes. Pass `None` when there is no supervisor
    /// (test backends, fake-server harnesses); production callers wire
    /// the `PresenceManager`-backed probe so crash → restart → replay
    /// works end-to-end.
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
            .timeout(Duration::from_secs(chat.request_timeout_secs))
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

    /// Snapshot the probe state at the moment of an HTTP failure.
    /// Returns `Some((pid_at_request, current_state))` when a probe is
    /// attached, allowing the caller to decide whether the failure
    /// looks like a server crash (pid changed or state != Ready). The
    /// outer `Option` is `None` when the client was built without a
    /// probe — in that case the caller treats every error as a
    /// transport fault (no replay).
    fn classify_failure(&self, pid_at_request: Option<u32>) -> bool {
        let Some(probe) = self.health.as_ref() else {
            return false;
        };
        let current_pid = probe.pid();
        let current_state = probe.state();
        // Crash-induced when:
        // - we had a pid before the request and either the pid has
        //   changed (supervisor already respawned) or it's now None
        //   (child died, supervisor not yet respawned), OR
        // - the state went non-Ready (Starting / BackingOff / Degraded).
        let pid_changed = match (pid_at_request, current_pid) {
            (Some(_), None) => true,
            (Some(a), Some(b)) => a != b,
            _ => false,
        };
        let state_unhealthy = !matches!(current_state, Some(ReadyState::Ready) | None);
        pid_changed || state_unhealthy
    }

    async fn stream_openai(&self, body: Vec<u8>, tx: &mpsc::Sender<LlmEvent>) -> StreamOutcome {
        let url = format!("{}/v1/chat/completions", self.base_url);
        // Snapshot the supervisor's PID at request time so we can
        // distinguish a crash-induced HTTP failure (pid changed or
        // disappeared, state went non-Ready) from a transport-level
        // hiccup (network, timeout) when classifying errors below.
        let pid_at_request = self.health.as_ref().and_then(|h| h.pid());
        tracing::debug!(
            target: "assistd::voice::latency",
            stage = "llm_request_sent",
            "voice latency stage"
        );
        let mut response = match self
            .client
            .post(&url)
            .header("Accept", "text/event-stream")
            .header("Content-Type", "application/json")
            .body(body)
            .send()
            .await
        {
            Ok(r) => {
                if self.classify_failure(pid_at_request) {
                    // Server crashed between snapshot and response; the
                    // 200 we just received was racing the supervisor's
                    // teardown. Treat as restart so the agent replays.
                    return StreamOutcome::ServerRestart {
                        accum: StreamAccum::default(),
                        pre_emit: true,
                    };
                }
                r
            }
            Err(e) => {
                if self.classify_failure(pid_at_request) {
                    return StreamOutcome::ServerRestart {
                        accum: StreamAccum::default(),
                        pre_emit: true,
                    };
                }
                return StreamOutcome::PreEmitError(ChatClientError::Http(e));
            }
        };

        let status = response.status();
        if !status.is_success() {
            let body = read_body_capped(&mut response, ERROR_BODY_CAP).await;
            if self.classify_failure(pid_at_request) {
                return StreamOutcome::ServerRestart {
                    accum: StreamAccum::default(),
                    pre_emit: true,
                };
            }
            return StreamOutcome::PreEmitError(ChatClientError::Server {
                status: status.as_u16(),
                body,
            });
        }

        let mut reader = SseLineReader::new();
        let mut accum = StreamAccum::default();
        let mut saw_done = false;
        let inactivity = Duration::from_secs(self.timeouts.stream_inactivity_secs);

        loop {
            let chunk = match timeout(inactivity, response.chunk()).await {
                Ok(Ok(Some(c))) => c,
                Ok(Ok(None)) => break,
                Ok(Err(e)) => {
                    if self.classify_failure(pid_at_request) {
                        let pre_emit = !accum_was_emitted(&accum);
                        return StreamOutcome::ServerRestart { accum, pre_emit };
                    }
                    return fold_mid_stream_error(accum, ChatClientError::Http(e));
                }
                Err(_) => {
                    warn!(
                        target: "assistd::chat",
                        timeout_secs = self.timeouts.stream_inactivity_secs,
                        bytes_so_far = accum.text.len(),
                        tool_call_builders = accum.tool_calls.len(),
                        "SSE stream inactive past deadline; aborting read"
                    );
                    if self.classify_failure(pid_at_request) {
                        let pre_emit = !accum_was_emitted(&accum);
                        return StreamOutcome::ServerRestart { accum, pre_emit };
                    }
                    return fold_mid_stream_error(
                        accum,
                        ChatClientError::Sse(format!(
                            "no bytes received for {}s",
                            self.timeouts.stream_inactivity_secs
                        )),
                    );
                }
            };
            reader.feed(&chunk);

            loop {
                let event = match reader.next_event() {
                    Ok(Some(e)) => e,
                    Ok(None) => break,
                    Err(e) => {
                        if self.classify_failure(pid_at_request) {
                            let pre_emit = !accum_was_emitted(&accum);
                            return StreamOutcome::ServerRestart { accum, pre_emit };
                        }
                        return fold_mid_stream_error(accum, e);
                    }
                };
                match event {
                    SseEvent::Data(payload) => {
                        let parsed: wire::ChatCompletionChunk = match serde_json::from_str(&payload)
                        {
                            Ok(p) => p,
                            Err(e) => {
                                if self.classify_failure(pid_at_request) {
                                    let pre_emit = !accum_was_emitted(&accum);
                                    return StreamOutcome::ServerRestart { accum, pre_emit };
                                }
                                return fold_mid_stream_error(accum, ChatClientError::Json(e));
                            }
                        };
                        let Some(choice) = parsed.choices.into_iter().next() else {
                            continue;
                        };
                        if let Some(reason) = choice.finish_reason {
                            accum.finish_reason = Some(reason);
                        }
                        if let Some(calls) = choice.delta.tool_calls {
                            for delta in calls {
                                accum.merge_tool_call_delta(delta);
                            }
                        }
                        if let Some(text) = choice.delta.content
                            && !text.is_empty()
                        {
                            if !accum.has_emitted {
                                tracing::debug!(
                                    target: "assistd::voice::latency",
                                    stage = "llm_first_token",
                                    "voice latency stage"
                                );
                            }
                            accum.text.push_str(&text);
                            accum.has_emitted = true;
                            if tx.send(LlmEvent::Delta { text }).await.is_err() {
                                debug!(
                                    target: "assistd::chat",
                                    "client disconnected mid-stream"
                                );
                                return StreamOutcome::ClientDisconnected(accum);
                            }
                        }
                    }
                    SseEvent::Done => {
                        saw_done = true;
                        break;
                    }
                }
            }

            if saw_done {
                break;
            }
        }

        if !saw_done {
            warn!(
                target: "assistd::chat",
                "stream ended before [DONE] marker; accumulated {} bytes text, {} tool-call builders",
                accum.text.len(),
                accum.tool_calls.len()
            );
        }
        StreamOutcome::Ok(accum)
    }
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
            let wire_messages = conv.as_wire_messages();
            let payload = wire::ChatRequest {
                model: self.model.name.as_str(),
                messages: wire_messages,
                stream: true,
                temperature: self.chat.temperature,
                max_tokens: self.chat.max_response_tokens,
                top_p: self.chat.top_p,
                top_k: self.chat.top_k,
                min_p: self.chat.min_p,
                presence_penalty: self.chat.presence_penalty,
                tools: None,
                tool_choice: None,
            };
            match serde_json::to_vec(&payload) {
                Ok(b) => b,
                Err(e) => {
                    // Roll back the user push so the caller can retry
                    // without observing the half-committed message.
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
            StreamOutcome::ServerRestart { .. } => {
                // `generate` is the legacy single-turn API; callers cannot
                // replay. Leave the user message in place — a follow-up call
                // would re-push, and double-pushing would duplicate.
                Err(LlmError::ServerRestarting(
                    "llama-server crashed during generate".into(),
                ))
            }
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
            // `[tool:<name>]\n<body>` — the prefix is the truncator's
            // pair-detection anchor (see TOOL_RESULT_PREFIX) and gives
            // the model a stable header it can visually spot in replayed
            // history. Attachments (if any) ride along as multimodal
            // content parts.
            let content = format!("[tool:{}]\n{}", r.name, r.content);
            if r.attachments.is_empty() {
                conv.push_user(content);
            } else {
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
            let wire_messages = conv.as_wire_messages();
            let has_tools = !tools.is_empty();
            let payload = wire::ChatRequest {
                model: self.model.name.as_str(),
                messages: wire_messages,
                stream: true,
                temperature: self.chat.temperature,
                max_tokens: self.chat.max_response_tokens,
                top_p: self.chat.top_p,
                top_k: self.chat.top_k,
                min_p: self.chat.min_p,
                presence_penalty: self.chat.presence_penalty,
                tools: if has_tools { Some(tools) } else { None },
                tool_choice: if has_tools { Some("auto") } else { None },
            };
            match serde_json::to_vec(&payload) {
                Ok(b) => b,
                Err(e) => {
                    return Err(LlmError::Chat(ChatClientError::Json(e)));
                }
            }
        };

        let outcome = self.stream_openai(body_bytes, &tx).await;

        let mut conv = self.conv.lock().await;
        match outcome {
            StreamOutcome::Ok(accum)
            | StreamOutcome::PartialAfterEmit(accum)
            | StreamOutcome::ClientDisconnected(accum) => {
                let result = commit_step(&mut conv, accum);
                // Consume so the next turn re-runs retrieval; `PreEmitError`
                // leaves it in place so a retry sees the same injected block.
                let _ = conv.consume_transient_context();
                result
            }
            StreamOutcome::PreEmitError(e) => Err(LlmError::Chat(e)),
            StreamOutcome::ServerRestart { accum, pre_emit } => {
                // Crash detected. The conversation state on this side
                // is already clean (we never reached `commit_step`),
                // but we may have streamed partial bytes to the client
                // — the agent loop's Status emission tells the client
                // to bracket those before the replay's deltas arrive.
                //
                // Important: we do NOT roll back the user message on
                // pre_emit, even though `PreEmitError` does. The user
                // push happens once before the agent loop iterates;
                // rolling it back here would mean the replay's
                // re-rendered wire payload has no user prompt at the
                // tail and the model would see an empty turn.
                //
                // We also preserve the transient context (skip
                // `consume_transient_context`) so the replay sees the
                // same retrieval block it would have seen on the
                // original attempt.
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
                }),
                HistoryRole::User => msgs.push(Message {
                    role: Role::User,
                    content: entry.content,
                    attachments: Vec::new(),
                    tool_calls: Vec::new(),
                }),
                HistoryRole::Assistant => {
                    let calls = parse_tool_calls(&entry.tool_calls_json)?;
                    msgs.push(Message {
                        role: Role::Assistant,
                        content: entry.content,
                        attachments: Vec::new(),
                        tool_calls: calls,
                    });
                }
                HistoryRole::Tool => {
                    // Tool results round-trip as Role::User with the
                    // [tool:<name>]\n prefix the truncator depends on
                    // for tool-call/result pair detection.
                    let name = entry.tool_name.unwrap_or_default();
                    let tagged = format!("[tool:{name}]\n{}", entry.content);
                    msgs.push(Message {
                        role: Role::User,
                        content: tagged,
                        attachments: Vec::new(),
                        tool_calls: Vec::new(),
                    });
                }
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
        // arguments is stored verbatim as JSON. It may be either a
        // JSON-encoded string or an object — store the literal string
        // so the wire payload matches what the model originally emitted.
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

/// Translate a completed stream into conversation commits + [`StepOutcome`].
///
/// On `finish_reason: "tool_calls"` (or an inferred tool-call outcome —
/// non-empty builders even without an explicit reason) we drop any
/// accumulated narrative text. Qwen3 and similar reasoning models
/// sometimes prefix tool calls with `<think>...</think>` content that
/// would poison the assistant-with-tool_calls message on replay.
fn commit_step(conv: &mut Conversation, accum: StreamAccum) -> LlmResult<StepOutcome> {
    let wants_tool_calls =
        accum.finish_reason.as_deref() == Some("tool_calls") && !accum.tool_calls.is_empty();
    if wants_tool_calls || (!accum.tool_calls.is_empty() && accum.finish_reason.is_none()) {
        let (records, parsed) = accum.finalize_tool_calls()?;
        conv.push_assistant_with_tool_calls(None, records);
        Ok(StepOutcome::ToolCalls(parsed))
    } else {
        conv.push_assistant(accum.text);
        Ok(StepOutcome::Final)
    }
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
    /// `BTreeMap` so a fallback iteration order (ascending `index`) is
    /// stable when we finalize, matching the order in which the model
    /// emitted the calls.
    tool_calls: BTreeMap<u32, ToolCallBuilder>,
    finish_reason: Option<String>,
    has_emitted: bool,
}

impl StreamAccum {
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

    /// Turn the builder map into `(records, parsed_calls)`:
    /// - `records` go into conversation history (arguments verbatim)
    /// - `parsed_calls` are what the agent loop dispatches (arguments as
    ///   `Value`). We parse arguments here so a malformed call surfaces
    ///   as an `Err` from `step` rather than propagating garbage into
    ///   tool dispatch.
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
    /// Stream errored after we'd already forwarded deltas — return what we have.
    PartialAfterEmit(StreamAccum),
    /// The consumer dropped the receiver mid-stream; stop quietly.
    ClientDisconnected(StreamAccum),
    /// Stream errored before any deltas were forwarded — propagate as `Err`.
    PreEmitError(ChatClientError),
    /// The HTTP failure looks crash-induced: the supervisor's PID
    /// changed under us or the readiness state went non-Ready. The
    /// agent loop will see [`LlmError::ServerRestarting`] and replay
    /// the same payload once after waiting for the supervisor to
    /// restore Ready. `pre_emit` distinguishes "no deltas streamed
    /// yet" (so the caller can clean up an unstreamed response) from
    /// "had partial output" (caller must consider what the user has
    /// already seen on the wire).
    ServerRestart { accum: StreamAccum, pre_emit: bool },
}

fn fold_mid_stream_error(accum: StreamAccum, err: ChatClientError) -> StreamOutcome {
    if accum.has_emitted || !accum.tool_calls.is_empty() {
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

fn accum_was_emitted(accum: &StreamAccum) -> bool {
    accum.has_emitted || !accum.tool_calls.is_empty()
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
