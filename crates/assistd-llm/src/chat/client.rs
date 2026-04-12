//! HTTP streaming chat client for the locally-managed llama-server.
//!
//! `LlamaChatClient` implements `LlmBackend`, so the daemon can drop it into
//! `AppState::llm` exactly where `EchoBackend` used to live. A single
//! `tokio::sync::Mutex<Conversation>` guards the entire chat history, and
//! it is held for the full duration of the streaming HTTP call — serializing
//! concurrent queries is the right trade-off for a single-user desktop
//! assistant: the second query always sees the first query's final reply.

use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::{Mutex, mpsc};
use tracing::{debug, warn};

use super::config::ChatSpec;
use super::conversation::{Conversation, Summarizer};
use super::error::ChatClientError;
use super::sse::{SseEvent, SseLineReader};
use super::wire;
use crate::{LlmBackend, LlmEvent};

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

    async fn stream_chat(
        &self,
        payload: &wire::ChatRequest<'_>,
        tx: &mpsc::Sender<LlmEvent>,
    ) -> StreamResult {
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
            Err(e) => return StreamResult::PreEmitError(ChatClientError::Http(e)),
        };

        let status = response.status();
        if !status.is_success() {
            let body = read_body_capped(&mut response, ERROR_BODY_CAP).await;
            return StreamResult::PreEmitError(ChatClientError::Server {
                status: status.as_u16(),
                body,
            });
        }

        let mut reader = SseLineReader::new();
        let mut accumulated = String::new();
        let mut has_emitted = false;
        let mut saw_done = false;

        loop {
            let chunk = match response.chunk().await {
                Ok(Some(c)) => c,
                Ok(None) => break,
                Err(e) => {
                    return fold_mid_stream_error(
                        accumulated,
                        has_emitted,
                        ChatClientError::Http(e),
                    );
                }
            };
            reader.feed(&chunk);

            loop {
                let event = match reader.next_event() {
                    Ok(Some(e)) => e,
                    Ok(None) => break,
                    Err(e) => {
                        return fold_mid_stream_error(accumulated, has_emitted, e);
                    }
                };
                match event {
                    SseEvent::Data(payload) => {
                        let parsed: wire::ChatCompletionChunk = match serde_json::from_str(&payload)
                        {
                            Ok(p) => p,
                            Err(e) => {
                                return fold_mid_stream_error(
                                    accumulated,
                                    has_emitted,
                                    ChatClientError::Json(e),
                                );
                            }
                        };
                        let Some(choice) = parsed.choices.into_iter().next() else {
                            continue;
                        };
                        let Some(text) = choice.delta.content else {
                            continue;
                        };
                        if text.is_empty() {
                            continue;
                        }
                        accumulated.push_str(&text);
                        has_emitted = true;
                        if tx.send(LlmEvent::Delta { text }).await.is_err() {
                            debug!(target: "assistd::chat", "client disconnected mid-stream");
                            return StreamResult::ClientDisconnected(accumulated);
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
                "stream ended before [DONE] marker; accumulated {} bytes",
                accumulated.len()
            );
        }
        StreamResult::Ok(accumulated)
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
        };

        match self.stream_chat(&payload, &tx).await {
            StreamResult::Ok(text) => {
                conv.push_assistant(text);
                let _ = tx.send(LlmEvent::Done).await;
                Ok(())
            }
            StreamResult::PartialAfterEmit(text) => {
                conv.push_assistant(text);
                let _ = tx.send(LlmEvent::Done).await;
                Ok(())
            }
            StreamResult::ClientDisconnected(text) => {
                conv.push_assistant(text);
                Ok(())
            }
            StreamResult::PreEmitError(e) => {
                conv.rollback_last_user();
                Err(anyhow::Error::new(e))
            }
        }
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
                    content: SUMMARY_SYSTEM_PROMPT,
                },
                wire::ChatMessage {
                    role: "user",
                    content: &dialogue,
                },
            ],
            stream: false,
            temperature: SUMMARY_TEMPERATURE,
            max_tokens,
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

enum StreamResult {
    /// Stream completed cleanly with a `[DONE]` marker (or EOF after deltas).
    Ok(String),
    /// Stream errored after we'd already forwarded deltas — return what we have.
    PartialAfterEmit(String),
    /// The consumer dropped the receiver mid-stream; stop quietly.
    ClientDisconnected(String),
    /// Stream errored before any deltas were forwarded — propagate as `Err`.
    PreEmitError(ChatClientError),
}

fn fold_mid_stream_error(
    accumulated: String,
    has_emitted: bool,
    err: ChatClientError,
) -> StreamResult {
    if has_emitted {
        warn!(
            target: "assistd::chat",
            "mid-stream error after {} bytes emitted: {err}",
            accumulated.len()
        );
        StreamResult::PartialAfterEmit(accumulated)
    } else {
        StreamResult::PreEmitError(err)
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
