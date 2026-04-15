//! Serde types for the OpenAI-shaped `/v1/chat/completions` endpoint as
//! served by llama.cpp. We only model the subset the client actually uses.
//!
//! `ChatMessage.content` is modeled as an untagged enum so text-only turns
//! render as a plain string (matching classic OpenAI payloads) while
//! multimodal turns render as an array of content parts. llama.cpp
//! accepts both shapes; a vision-capable model + mmproj is required for
//! the `image_url` parts to actually reach the projector.

use serde::{Deserialize, Serialize};

/// Outgoing chat request. Uses borrowed strings so history can be rendered
/// into wire messages without copying.
#[derive(Debug, Clone, Serialize)]
pub struct ChatRequest<'a> {
    pub model: &'a str,
    pub messages: Vec<ChatMessage<'a>>,
    pub stream: bool,
    pub temperature: f32,
    pub max_tokens: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage<'a> {
    pub role: &'a str,
    pub content: ContentBody<'a>,
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
    #[allow(dead_code)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
pub struct ChatChunkDelta {
    #[serde(default)]
    #[allow(dead_code)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
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
                    content: ContentBody::Text("you are helpful"),
                },
                ChatMessage {
                    role: "user",
                    content: ContentBody::Text("hi"),
                },
            ],
            stream: true,
            temperature: 0.5,
            max_tokens: 128,
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
    }

    #[test]
    fn serializes_multimodal_content_as_parts_array() {
        let req = ChatRequest {
            model: "local",
            messages: vec![ChatMessage {
                role: "user",
                content: ContentBody::Parts(vec![
                    ContentPart::Text {
                        text: "what's in this image?",
                    },
                    ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: "data:image/png;base64,AAAA".into(),
                        },
                    },
                ]),
            }],
            stream: false,
            temperature: 0.5,
            max_tokens: 64,
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
    fn deserializes_streaming_chunk_with_content_delta() {
        let payload = r#"{"id":"x","object":"chat.completion.chunk","created":1,"model":"local","choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}"#;
        let parsed: ChatCompletionChunk = serde_json::from_str(payload).unwrap();
        assert_eq!(parsed.choices.len(), 1);
        assert_eq!(parsed.choices[0].delta.content.as_deref(), Some("hello"));
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
    fn deserializes_non_streaming_response() {
        let body = r#"{"id":"y","object":"chat.completion","created":1,"model":"local","choices":[{"index":0,"message":{"role":"assistant","content":"summary text"},"finish_reason":"stop"}]}"#;
        let parsed: ChatResponse = serde_json::from_str(body).unwrap();
        assert_eq!(parsed.choices.len(), 1);
        assert_eq!(parsed.choices[0].message.content, "summary text");
    }
}
