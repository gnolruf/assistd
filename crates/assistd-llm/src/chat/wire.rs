//! Serde types for the OpenAI-shaped `/v1/chat/completions` endpoint as
//! served by llama.cpp. We only model the subset the client actually uses.
//!
//! `ChatMessage.content` is modeled as an untagged enum so text-only turns
//! render as a plain string (matching classic OpenAI payloads) while
//! multimodal turns render as an array of content parts. llama.cpp
//! accepts both shapes; a vision-capable model + mmproj is required for
//! the `image_url` parts to actually reach the projector.
//!
//! Tool-calling shape: when an assistant message carries `tool_calls`, the
//! `content` field must be absent (`None`). Some llama.cpp Jinja templates
//! reject `"content": null` but accept an omitted key — `skip_serializing_if`
//! takes care of that.

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
    /// Only emitted when non-None so requests without tool use remain
    /// byte-identical to the pre-agent-loop format.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Value>>,
    /// Controls whether the model is allowed/required to call tools.
    /// `"auto"` lets the model decide; `"none"` forces a text reply.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<&'a str>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage<'a> {
    pub role: &'a str,
    /// Message text. `None` for assistant messages that carry only
    /// `tool_calls` — the OpenAI spec allows (and many servers require)
    /// the field to be omitted entirely in that case.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<ContentBody<'a>>,
    /// Tool calls the assistant is requesting. Set on assistant turns
    /// that finish with `finish_reason: "tool_calls"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallSpec<'a>>>,
    /// Only set on messages with `role: "tool"`. We currently route tool
    /// results back through synthetic user messages instead, so this is
    /// usually absent — but keeping the field on the wire type lets the
    /// client interop if a future change flips back.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<&'a str>,
}

/// Wire shape of a message's `content` field.
///
/// Untagged serde keeps the two shapes indistinguishable on the outgoing
/// wire — `Text(s)` serializes as the bare string `"..."`, `Parts(v)`
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

/// Non-streaming response shape — used only for the summarization call.
#[derive(Debug, Deserialize)]
pub struct ChatResponse {
    pub choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
pub struct ChatChoice {
    pub message: ChatChoiceMessage,
    #[serde(default)]
    #[allow(dead_code)]
    pub finish_reason: Option<String>,
}

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

#[derive(Debug, Deserialize)]
pub struct ChatChunkChoice {
    pub delta: ChatChunkDelta,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
pub struct ChatChunkDelta {
    #[serde(default)]
    #[allow(dead_code)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    /// When the model calls a tool, llama.cpp streams call fragments across
    /// multiple chunks. Each fragment carries a stable `index` so the
    /// receiver can accumulate by slot.
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

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
mod tests {
    use super::*;

    #[test]
    fn serializes_chat_request_with_expected_fields() {
        let req = ChatRequest {
            model: "local",
            messages: vec![
                ChatMessage {
                    role: "system",
                    content: Some(ContentBody::Text("you are helpful")),
                    tool_calls: None,
                    tool_call_id: None,
                },
                ChatMessage {
                    role: "user",
                    content: Some(ContentBody::Text("hi")),
                    tool_calls: None,
                    tool_call_id: None,
                },
            ],
            stream: true,
            temperature: 0.5,
            max_tokens: 128,
            top_p: None,
            top_k: None,
            min_p: None,
            presence_penalty: None,
            tools: None,
            tool_choice: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "local");
        assert_eq!(json["stream"], true);
        let temp = json["temperature"]
            .as_f64()
            .expect("temperature is a number");
        assert!((temp - 0.5).abs() < 1e-4);
        assert_eq!(json["max_tokens"], 128);
        assert_eq!(json["messages"][0]["role"], "system");
        assert_eq!(json["messages"][1]["content"], "hi");
        // Absent tools/tool_choice must not appear on the wire.
        assert!(json.get("tools").is_none());
        assert!(json.get("tool_choice").is_none());
        // Text-only messages must not carry tool_calls/tool_call_id keys.
        assert!(json["messages"][0].get("tool_calls").is_none());
        assert!(json["messages"][0].get("tool_call_id").is_none());
    }

    #[test]
    fn serializes_multimodal_content_as_parts_array() {
        let req = ChatRequest {
            model: "local",
            messages: vec![ChatMessage {
                role: "user",
                content: Some(ContentBody::Parts(vec![
                    ContentPart::Text {
                        text: "what's in this image?",
                    },
                    ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: "data:image/png;base64,AAAA".into(),
                        },
                    },
                ])),
                tool_calls: None,
                tool_call_id: None,
            }],
            stream: false,
            temperature: 0.5,
            max_tokens: 64,
            top_p: None,
            top_k: None,
            min_p: None,
            presence_penalty: None,
            tools: None,
            tool_choice: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        let content = &json["messages"][0]["content"];
        assert!(content.is_array(), "{content}");
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "what's in this image?");
        assert_eq!(content[1]["type"], "image_url");
        assert_eq!(content[1]["image_url"]["url"], "data:image/png;base64,AAAA");
    }

    #[test]
    fn serializes_tools_array_when_set() {
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "run",
                "description": "run a command",
                "parameters": {"type": "object"},
                "strict": true,
            }
        })];
        let req = ChatRequest {
            model: "local",
            messages: Vec::new(),
            stream: true,
            temperature: 0.5,
            max_tokens: 64,
            top_p: None,
            top_k: None,
            min_p: None,
            presence_penalty: None,
            tools: Some(tools),
            tool_choice: Some("auto"),
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["tool_choice"], "auto");
        assert_eq!(json["tools"][0]["function"]["name"], "run");
        assert_eq!(json["tools"][0]["function"]["strict"], true);
    }

    #[test]
    fn serializes_assistant_with_tool_calls_omits_content() {
        let msg = ChatMessage {
            role: "assistant",
            content: None,
            tool_calls: Some(vec![ToolCallSpec {
                id: "call-1",
                kind: "function",
                function: FunctionCallSpec {
                    name: "run",
                    arguments: r#"{"command":"ls /tmp"}"#,
                },
            }]),
            tool_call_id: None,
        };
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["role"], "assistant");
        // content must be absent (not null) so strict templates accept it.
        assert!(
            json.get("content").is_none(),
            "content should be omitted, got {json}"
        );
        assert_eq!(json["tool_calls"][0]["id"], "call-1");
        assert_eq!(json["tool_calls"][0]["type"], "function");
        assert_eq!(json["tool_calls"][0]["function"]["name"], "run");
        assert_eq!(
            json["tool_calls"][0]["function"]["arguments"],
            r#"{"command":"ls /tmp"}"#
        );
    }

    #[test]
    fn deserializes_streaming_chunk_with_content_delta() {
        let payload = r#"{"id":"x","object":"chat.completion.chunk","created":1,"model":"local","choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}"#;
        let parsed: ChatCompletionChunk = serde_json::from_str(payload).unwrap();
        assert_eq!(parsed.choices.len(), 1);
        assert_eq!(parsed.choices[0].delta.content.as_deref(), Some("hello"));
        assert!(parsed.choices[0].delta.tool_calls.is_none());
    }

    #[test]
    fn deserializes_role_only_first_chunk() {
        let payload =
            r#"{"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#;
        let parsed: ChatCompletionChunk = serde_json::from_str(payload).unwrap();
        assert_eq!(parsed.choices[0].delta.role.as_deref(), Some("assistant"));
        assert!(parsed.choices[0].delta.content.is_none());
    }

    #[test]
    fn deserializes_tool_call_delta_chunk() {
        // llama.cpp's typical tool-call emission — id + name in one chunk,
        // arguments streamed across subsequent chunks under the same index.
        let payload = r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call-1","type":"function","function":{"name":"run","arguments":"{\"com"}}]},"finish_reason":null}]}"#;
        let parsed: ChatCompletionChunk = serde_json::from_str(payload).unwrap();
        let calls = parsed.choices[0].delta.tool_calls.as_ref().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].index, 0);
        assert_eq!(calls[0].id.as_deref(), Some("call-1"));
        let fn_delta = calls[0].function.as_ref().unwrap();
        assert_eq!(fn_delta.name.as_deref(), Some("run"));
        assert_eq!(fn_delta.arguments.as_deref(), Some("{\"com"));
    }

    #[test]
    fn deserializes_tool_calls_finish_reason() {
        let payload = r#"{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#;
        let parsed: ChatCompletionChunk = serde_json::from_str(payload).unwrap();
        assert_eq!(
            parsed.choices[0].finish_reason.as_deref(),
            Some("tool_calls")
        );
    }

    #[test]
    fn deserializes_non_streaming_response() {
        let body = r#"{"id":"y","object":"chat.completion","created":1,"model":"local","choices":[{"index":0,"message":{"role":"assistant","content":"summary text"},"finish_reason":"stop"}]}"#;
        let parsed: ChatResponse = serde_json::from_str(body).unwrap();
        assert_eq!(parsed.choices.len(), 1);
        assert_eq!(parsed.choices[0].message.content, "summary text");
    }
}
