//! Serde types for the OpenAI-shaped `/v1/chat/completions` endpoint as
//! served by llama.cpp. We only model the subset the client actually uses.
//!
//! `ChatMessage.content` is modeled as an untagged enum so text-only turns
//! render as a plain string (matching classic OpenAI payloads) while
//! multimodal turns render as an array of content parts. llama.cpp
//! accepts both shapes; a vision-capable model + mmproj is required for
//! the `image_url` parts to actually reach the projector.
//!
//! Tool-calling shape: an assistant message carrying `tool_calls` keeps
//! whatever narration the model streamed before the call as `content`, and
//! omits the field entirely when there was none. Some llama.cpp Jinja
//! templates reject `"content": null` but accept an omitted key;
//! `skip_serializing_if` takes care of that.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Outgoing chat request. Uses borrowed strings so history can be rendered
/// into wire messages without copying.
#[derive(Debug, Clone, Serialize)]
pub struct ChatRequest<'a> {
    pub model: &'a str,
    pub messages: Vec<ChatMessage<'a>>,
    pub stream: bool,
    pub temperature: f32,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    /// OpenAI-compatible tool schemas: `[{"type": "function", "function": {...}}]`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Value>>,
    /// `"auto"` lets the model decide whether to call tools; `"none"`
    /// forces a text reply.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<&'a str>,
    /// Extra variables handed to llama.cpp's Jinja chat template.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template_kwargs: Option<ChatTemplateKwargs>,
}

/// Chat-template variables. `enable_thinking: false` is the convention
/// reasoning models (Qwen3, DeepSeek-R1) use to skip the `<think>`
/// block; templates without the variable ignore it.
#[derive(Debug, Clone, Serialize)]
pub struct ChatTemplateKwargs {
    pub enable_thinking: bool,
}

/// One message in the outgoing `messages` array.
#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage<'a> {
    pub role: &'a str,
    /// Message text. `None` for assistant messages that carry only
    /// `tool_calls`; the OpenAI spec allows (and many servers require)
    /// the field to be omitted entirely in that case.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<ContentBody<'a>>,
    /// Tool calls the assistant is requesting. Set on assistant turns
    /// that finish with `finish_reason: "tool_calls"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallSpec<'a>>>,
    /// Only set on messages with `role: "tool"`: the id of the assistant
    /// tool call this message answers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<&'a str>,
}

/// Wire shape of a message's `content` field.
///
/// Untagged serde keeps the two shapes indistinguishable on the outgoing
/// wire: `Text(s)` serializes as the bare string `"..."`, `Parts(v)`
/// serializes as a JSON array `[{"type": "text", "text": "..."}, ...]`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub enum ContentBody<'a> {
    Text(&'a str),
    Parts(Vec<ContentPart<'a>>),
}

/// One slot in a multimodal `content` array.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart<'a> {
    Text { text: &'a str },
    ImageUrl { image_url: ImageUrl },
}

/// `{"url": "data:image/png;base64,..."}`. Owned because we build the
/// data URI on the fly when rendering wire messages.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ImageUrl {
    pub url: String,
}

/// One entry in an outgoing `tool_calls` array on an assistant message.
#[derive(Debug, Clone, Serialize)]
pub struct ToolCallSpec<'a> {
    pub id: &'a str,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: FunctionCallSpec<'a>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FunctionCallSpec<'a> {
    pub name: &'a str,
    /// OpenAI's spec is explicit: `arguments` is a JSON-encoded **string**,
    /// not a JSON object. Strict parsers reject the object form.
    pub arguments: &'a str,
}

/// Non-streaming response body, used only for the summarization call.
#[derive(Debug, Deserialize)]
pub struct ChatResponse {
    pub choices: Vec<ChatChoice>,
}

/// One choice in a non-streaming [`ChatResponse`].
#[derive(Debug, Deserialize)]
pub struct ChatChoice {
    pub message: ChatChoiceMessage,
    #[serde(default)]
    #[allow(dead_code)]
    pub finish_reason: Option<String>,
}

/// The assistant message inside a non-streaming choice.
#[derive(Debug, Deserialize)]
pub struct ChatChoiceMessage {
    #[allow(dead_code)]
    pub role: String,
    pub content: String,
}

/// One streamed chunk from `/v1/chat/completions` when `stream: true`.
#[derive(Debug, Deserialize)]
pub struct ChatCompletionChunk {
    pub choices: Vec<ChatChunkChoice>,
}

/// One choice slot in a streaming [`ChatCompletionChunk`].
#[derive(Debug, Deserialize)]
pub struct ChatChunkChoice {
    pub delta: ChatChunkDelta,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

/// Incremental fields emitted in a single streaming chunk.
#[derive(Debug, Deserialize, Default)]
pub struct ChatChunkDelta {
    #[serde(default)]
    #[allow(dead_code)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    /// llama.cpp's separated reasoning channel, present when the server
    /// runs with `--reasoning-format`; otherwise reasoning arrives inline
    /// in `content` between `<think>...</think>` tags.
    #[serde(default)]
    pub reasoning_content: Option<String>,
    /// Tool-call fragments streamed across multiple chunks; accumulate by `index`.
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

/// One tool-call slot streamed in a [`ChatChunkDelta`].
#[derive(Debug, Deserialize, Default)]
pub struct ToolCallDelta {
    /// Stable across chunks for the same call. Required to reassemble
    /// arguments that stream in pieces.
    #[serde(default)]
    pub index: u32,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default, rename = "type")]
    #[allow(dead_code)]
    pub kind: Option<String>,
    #[serde(default)]
    pub function: Option<FunctionCallDelta>,
}

/// Incremental function-call fields within a [`ToolCallDelta`].
#[derive(Debug, Deserialize, Default)]
pub struct FunctionCallDelta {
    #[serde(default)]
    pub name: Option<String>,
    /// JSON-encoded arguments string, chunked. Concatenate across deltas
    /// keyed by the same `index`.
    #[serde(default)]
    pub arguments: Option<String>,
}

#[cfg(test)]
mod tests;
