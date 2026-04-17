//! HTTP streaming chat client for the locally-managed llama-server.
//!
//! `LlamaChatClient` implements `LlmBackend`, so the daemon can drop it into
//! `AppState::llm` exactly where `EchoBackend` used to live. A single
//! `tokio::sync::Mutex<Conversation>` guards the entire chat history, and
//! it is held for the full duration of the streaming HTTP call — serializing
//! concurrent queries is the right trade-off for a single-user desktop
//! assistant: the second query always sees the first query's final reply.
//!
//! Tool-use support comes via three extra `LlmBackend` methods:
//! `push_user`, `push_tool_results`, and `step`. The agent loop in
//! `assistd-core` drives them: `push_user(text)` at the start of a turn,
//! then `step` → handle the outcome → `push_tool_results(...)` if needed →
//! `step` again, until `StepOutcome::Final`.

use std::collections::BTreeMap;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::{Mutex, mpsc};
use tracing::{debug, warn};

use super::config::ChatSpec;
use super::conversation::{Conversation, Summarizer, ToolCallRecord};
use super::error::ChatClientError;
use super::sse::{SseEvent, SseLineReader};
use super::wire;
use crate::{LlmBackend, LlmEvent, StepOutcome, ToolCall, ToolResultPayload};

const MODEL_NAME: &str = "local";
const ERROR_BODY_CAP: usize = 1024;
const SUMMARY_TEMPERATURE: f32 = 0.3;
const SUMMARY_SYSTEM_PROMPT: &str = "You are a conversation summarizer. Produce a concise \
    summary of the following dialogue that preserves all factual claims, user requests, and \
    assistant conclusions. Write in past tense. Do not add commentary.";

pub struct LlamaChatClient {
    client: reqwest::Client,
    base_url: String,
    spec: ChatSpec,
    conv: Mutex<Conversation>,
}

impl LlamaChatClient {
    pub fn new(spec: ChatSpec) -> Result<Self, ChatClientError> {
        let client = reqwest::Client::builder()
            .no_proxy()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(spec.request_timeout_secs))
            .build()?;
        let base_url = format!("http://{}:{}", spec.host, spec.port);
        let conv = Conversation::new(spec.system_prompt.clone());
        Ok(Self {
            client,
            base_url,
            spec,
            conv: Mutex::new(conv),
        })
    }

    async fn stream_openai(
        &self,
        payload: &wire::ChatRequest<'_>,
        tx: &mpsc::Sender<LlmEvent>,
    ) -> StreamOutcome {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let mut response = match self
            .client
            .post(&url)
            .header("Accept", "text/event-stream")
            .json(payload)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => return StreamOutcome::PreEmitError(ChatClientError::Http(e)),
        };

        let status = response.status();
        if !status.is_success() {
            let body = read_body_capped(&mut response, ERROR_BODY_CAP).await;
            return StreamOutcome::PreEmitError(ChatClientError::Server {
                status: status.as_u16(),
                body,
            });
        }

        let mut reader = SseLineReader::new();
        let mut accum = StreamAccum::default();
        let mut saw_done = false;

        loop {
            let chunk = match response.chunk().await {
                Ok(Some(c)) => c,
                Ok(None) => break,
                Err(e) => {
                    return fold_mid_stream_error(accum, ChatClientError::Http(e));
                }
            };
            reader.feed(&chunk);

            loop {
                let event = match reader.next_event() {
                    Ok(Some(e)) => e,
                    Ok(None) => break,
                    Err(e) => {
                        return fold_mid_stream_error(accum, e);
                    }
                };
                match event {
                    SseEvent::Data(payload) => {
                        let parsed: wire::ChatCompletionChunk = match serde_json::from_str(&payload)
                        {
                            Ok(p) => p,
                            Err(e) => {
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
    async fn generate(&self, prompt: String, tx: mpsc::Sender<LlmEvent>) -> Result<()> {
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

        if let Err(e) = conv.ensure_budget(self, &self.spec).await {
            warn!(
                target: "assistd::chat",
                "ensure_budget failed ({e}); falling back to truncation"
            );
            conv.truncate_to_budget(&self.spec);
        }

        let wire_messages = conv.as_wire_messages();
        let payload = wire::ChatRequest {
            model: MODEL_NAME,
            messages: wire_messages,
            stream: true,
            temperature: self.spec.temperature,
            max_tokens: self.spec.max_response_tokens,
            tools: None,
            tool_choice: None,
        };

        match self.stream_openai(&payload, &tx).await {
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
                Err(anyhow::Error::new(e))
            }
        }
    }

    async fn push_user(&self, text: String) -> Result<()> {
        let mut conv = self.conv.lock().await;
        conv.push_user(text);
        Ok(())
    }

    async fn push_tool_results(&self, results: Vec<ToolResultPayload>) -> Result<()> {
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

    async fn step(&self, tools: Vec<Value>, tx: mpsc::Sender<LlmEvent>) -> Result<StepOutcome> {
        let mut conv = self.conv.lock().await;

        if let Err(e) = conv.ensure_budget(self, &self.spec).await {
            warn!(
                target: "assistd::chat",
                "ensure_budget failed ({e}); falling back to truncation"
            );
            conv.truncate_to_budget(&self.spec);
        }

        let wire_messages = conv.as_wire_messages();
        let has_tools = !tools.is_empty();
        let payload = wire::ChatRequest {
            model: MODEL_NAME,
            messages: wire_messages,
            stream: true,
            temperature: self.spec.temperature,
            max_tokens: self.spec.max_response_tokens,
            tools: if has_tools { Some(tools) } else { None },
            tool_choice: if has_tools { Some("auto") } else { None },
        };

        let outcome = self.stream_openai(&payload, &tx).await;
        match outcome {
            StreamOutcome::Ok(accum)
            | StreamOutcome::PartialAfterEmit(accum)
            | StreamOutcome::ClientDisconnected(accum) => commit_step(&mut conv, accum),
            StreamOutcome::PreEmitError(e) => Err(anyhow::Error::new(e)),
        }
    }
}

/// Translate a completed stream into conversation commits + [`StepOutcome`].
///
/// On `finish_reason: "tool_calls"` (or an inferred tool-call outcome —
/// non-empty builders even without an explicit reason) we drop any
/// accumulated narrative text. Qwen3 and similar reasoning models
/// sometimes prefix tool calls with `<think>...</think>` content that
/// would poison the assistant-with-tool_calls message on replay.
fn commit_step(conv: &mut Conversation, accum: StreamAccum) -> Result<StepOutcome> {
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
            model: MODEL_NAME,
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
            temperature: SUMMARY_TEMPERATURE,
            max_tokens,
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
    fn finalize_tool_calls(self) -> Result<(Vec<ToolCallRecord>, Vec<ToolCall>)> {
        let mut records = Vec::with_capacity(self.tool_calls.len());
        let mut parsed = Vec::with_capacity(self.tool_calls.len());
        for (index, b) in self.tool_calls {
            if b.name.is_empty() {
                anyhow::bail!("tool call at index {index} has no name");
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
            let arguments_value = serde_json::from_str::<Value>(&arguments_json)
                .map_err(|e| anyhow::anyhow!("tool call {id}: malformed arguments JSON: {e}"))?;
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
