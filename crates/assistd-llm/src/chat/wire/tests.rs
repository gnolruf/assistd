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
        chat_template_kwargs: None,
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
