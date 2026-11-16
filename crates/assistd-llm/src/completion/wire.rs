//! Serde types for the subset of llama.cpp's OpenAI-shaped
//! `/v1/chat/completions` endpoint that the chat client uses.

use std::borrow::Cow;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Outgoing chat request. Borrows from the history wherever a message
/// goes out verbatim, so rendering copies only the text it has to build.
#[derive(Debug, Clone, Serialize)]
pub struct ChatRequest<'a> {
    pub model: &'a str,
    pub messages: Vec<ChatMessage<'a>>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
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
    /// Most tokens the model may think for before llama-server ends its
    /// `<think>` block; a sampler setting, so the prompt is unchanged.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_budget_tokens: Option<u32>,
    /// Text llama-server writes into the thinking where the budget cut it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_budget_message: Option<&'a str>,
    /// Extra variables handed to llama.cpp's Jinja chat template.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template_kwargs: Option<ChatTemplateKwargs<'a>>,
    /// Holds the reply to a JSON schema through llama-server's grammar sampler.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat<'a>>,
    /// The server slot to decode in; the server picks an idle one when unset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id_slot: Option<u32>,
}

/// OpenAI's `response_format`, in its `json_schema` form.
#[derive(Debug, Clone, Serialize)]
pub struct ResponseFormat<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    json_schema: JsonSchemaFormat<'a>,
}

#[derive(Debug, Clone, Serialize)]
struct JsonSchemaFormat<'a> {
    name: &'static str,
    strict: bool,
    schema: &'a Map<String, Value>,
}

impl<'a> ResponseFormat<'a> {
    /// A reply that must be a JSON value matching `schema`.
    pub fn json_schema(schema: &'a Map<String, Value>) -> Self {
        Self {
            kind: "json_schema",
            json_schema: JsonSchemaFormat {
                name: "answer",
                strict: true,
                schema,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct StreamOptions {
    /// Ask for a final chunk carrying the request's token [`Usage`].
    pub include_usage: bool,
}

/// Chat-template variables; templates ignore the ones they don't use.
/// `enable_thinking: false` is the convention reasoning models (Qwen3,
/// DeepSeek-R1) use to skip the `<think>` block.
#[derive(Debug, Clone, Serialize)]
pub struct ChatTemplateKwargs<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_thinking: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<&'a str>,
}

/// One message in the outgoing `messages` array.
#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage<'a> {
    pub role: &'a str,
    /// Message text. `None` for an assistant message whose `tool_calls`
    /// had no narration before them; the key is then omitted rather than
    /// sent as `null`, which some llama.cpp Jinja templates reject.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<ContentBody<'a>>,
    /// Tool calls the assistant is requesting. Set on assistant turns
    /// that finish with `finish_reason: "tool_calls"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallSpec<'a>>>,
    /// Only set on messages with `role: "tool"`: the id of the assistant
    /// tool call this message answers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<Cow<'a, str>>,
    /// The reasoning that led to an assistant message's `tool_calls`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<Cow<'a, str>>,
}

impl<'a> ChatMessage<'a> {
    /// A message carrying only `role` and `content`.
    pub fn plain(role: &'a str, content: ContentBody<'a>) -> Self {
        Self {
            role,
            content: Some(content),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }
}

/// Wire shape of a message's `content`: a bare string, or an array of
/// multimodal content parts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub enum ContentBody<'a> {
    Text(Cow<'a, str>),
    Parts(Vec<ContentPart<'a>>),
}

/// One slot in a multimodal `content` array.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart<'a> {
    Text { text: Cow<'a, str> },
    ImageUrl { image_url: ImageUrl<'a> },
}

/// `{"url": "data:image/png;base64,..."}`, borrowed from the history.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ImageUrl<'a> {
    pub url: &'a str,
}

/// One entry in an outgoing `tool_calls` array on an assistant message.
#[derive(Debug, Clone, Serialize)]
pub struct ToolCallSpec<'a> {
    pub id: Cow<'a, str>,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: FunctionCallSpec<'a>,
}

/// The `function` object of a [`ToolCallSpec`].
#[derive(Debug, Clone, Serialize)]
pub struct FunctionCallSpec<'a> {
    pub name: Cow<'a, str>,
    /// JSON-encoded string, never an object; strict parsers reject objects.
    pub arguments: Cow<'a, str>,
}

/// One streamed chunk from `/v1/chat/completions` when `stream: true`.
#[derive(Debug, Deserialize)]
pub struct ChatCompletionChunk {
    #[serde(default)]
    pub choices: Vec<ChatChunkChoice>,
    /// Present only on the final chunk, and only when requested.
    #[serde(default)]
    pub usage: Option<Usage>,
    /// Set instead of `choices` when generation failed mid-stream.
    #[serde(default)]
    pub error: Option<StreamError>,
}

/// The error object llama-server sends in place of a chunk when
/// generation fails after the response headers went out.
#[derive(Debug, Deserialize)]
pub struct StreamError {
    /// The HTTP status the failure would have had before streaming began.
    pub code: u16,
    pub message: String,
    /// llama-server's name for the class of failure.
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
    /// The prompt size measured, on a context-overflow error.
    #[serde(default)]
    pub n_prompt_tokens: Option<u32>,
}

/// The body of a response llama-server fails with before streaming begins.
#[derive(Debug, Deserialize)]
pub struct ErrorBody {
    pub error: StreamError,
}

/// Token counts the server measured for one request.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Usage {
    /// Tokens in the rendered prompt, cached or not.
    pub prompt_tokens: u32,
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
    pub content: Option<String>,
    /// Separate reasoning channel, present only under `--reasoning-format`.
    #[serde(default)]
    pub reasoning_content: Option<String>,
    /// Tool-call fragments streamed across multiple chunks; accumulate by `index`.
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

/// One tool-call slot streamed in a [`ChatChunkDelta`].
#[derive(Debug, Deserialize, Default)]
pub struct ToolCallDelta {
    /// Stable across chunks for the same call.
    #[serde(default)]
    pub index: u32,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub function: Option<FunctionCallDelta>,
}

/// Incremental function-call fields within a [`ToolCallDelta`].
#[derive(Debug, Deserialize, Default)]
pub struct FunctionCallDelta {
    #[serde(default)]
    pub name: Option<String>,
    /// A fragment of the JSON-encoded arguments string.
    #[serde(default)]
    pub arguments: Option<String>,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn message<'a>(role: &'a str, content: Option<ContentBody<'a>>) -> ChatMessage<'a> {
        ChatMessage {
            role,
            content,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    fn request(messages: Vec<ChatMessage<'_>>) -> ChatRequest<'_> {
        ChatRequest {
            model: "local",
            messages,
            stream: true,
            stream_options: None,
            temperature: 0.5,
            max_tokens: 128,
            top_p: None,
            top_k: None,
            min_p: None,
            presence_penalty: None,
            tools: None,
            tool_choice: None,
            reasoning_budget_tokens: None,
            reasoning_budget_message: None,
            chat_template_kwargs: None,
            response_format: None,
            id_slot: None,
        }
    }

    #[test]
    fn chat_request_omits_unset_optional_fields() {
        let req = request(vec![
            message("system", Some(ContentBody::Text("you are helpful".into()))),
            message("user", Some(ContentBody::Text("hi".into()))),
        ]);
        assert_eq!(
            serde_json::to_value(&req).unwrap(),
            json!({
                "model": "local",
                "messages": [
                    {"role": "system", "content": "you are helpful"},
                    {"role": "user", "content": "hi"},
                ],
                "stream": true,
                "temperature": 0.5,
                "max_tokens": 128,
            })
        );
    }

    #[test]
    fn serializes_multimodal_content_as_parts_array() {
        let msg = message(
            "user",
            Some(ContentBody::Parts(vec![
                ContentPart::Text {
                    text: "what's in this image?".into(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "data:image/png;base64,AAAA",
                    },
                },
            ])),
        );
        assert_eq!(
            serde_json::to_value(&msg).unwrap(),
            json!({
                "role": "user",
                "content": [
                    {"type": "text", "text": "what's in this image?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            })
        );
    }

    #[test]
    fn deserializes_tool_call_delta_chunk() {
        let payload = r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call-1","type":"function","function":{"name":"run","arguments":"{\"com"}}]},"finish_reason":null}]}"#;
        let parsed: ChatCompletionChunk = serde_json::from_str(payload).unwrap();
        let [call] = parsed.choices[0].delta.tool_calls.as_deref().unwrap() else {
            panic!("expected exactly one tool-call delta");
        };
        assert_eq!(call.index, 0);
        assert_eq!(call.id.as_deref(), Some("call-1"));
        let function = call.function.as_ref().unwrap();
        assert_eq!(function.name.as_deref(), Some("run"));
        assert_eq!(function.arguments.as_deref(), Some("{\"com"));
    }

    #[test]
    fn final_chunk_carries_usage_with_no_choices() {
        let chunk: ChatCompletionChunk = serde_json::from_value(json!({
            "choices": [],
            "usage": {
                "completion_tokens": 3,
                "prompt_tokens": 275,
                "total_tokens": 278,
                "prompt_tokens_details": {"cached_tokens": 271},
            },
            "timings": {"cache_n": 271, "prompt_n": 4},
        }))
        .unwrap();
        assert_eq!(chunk.usage.map(|u| u.prompt_tokens), Some(275));

        let plain: ChatCompletionChunk =
            serde_json::from_value(json!({"choices": [{"delta": {"content": "hi"}}]})).unwrap();
        assert!(plain.usage.is_none());
    }
}
