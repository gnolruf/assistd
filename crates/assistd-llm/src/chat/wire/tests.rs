use super::*;
use serde_json::json;

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
fn serializes_tools_and_tool_choice_when_set() {
    let tool = json!({
        "type": "function",
        "function": {
            "name": "run",
            "description": "run a command",
            "parameters": {"type": "object"},
            "strict": true,
        }
    });
    let mut req = request(Vec::new());
    req.tools = Some(vec![tool.clone()]);
    req.tool_choice = Some("auto");
    let json = serde_json::to_value(&req).unwrap();
    assert_eq!(json["tool_choice"], "auto");
    assert_eq!(json["tools"], json!([tool]));
}

#[test]
fn serializes_assistant_tool_calls_with_content_omitted() {
    let mut msg = message("assistant", None);
    msg.tool_calls = Some(vec![ToolCallSpec {
        id: "call-1",
        kind: "function",
        function: FunctionCallSpec {
            name: "run",
            arguments: r#"{"command":"ls /tmp"}"#,
        },
    }]);
    // Absent, not null: strict chat templates reject `"content": null`.
    assert_eq!(
        serde_json::to_value(&msg).unwrap(),
        json!({
            "role": "assistant",
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "run", "arguments": r#"{"command":"ls /tmp"}"#},
            }],
        })
    );
}

#[test]
fn deserializes_text_and_reasoning_deltas() {
    let cases = [
        (
            r#"{"id":"x","object":"chat.completion.chunk","created":1,"model":"local","choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}"#,
            Some("hello"),
            None,
        ),
        (
            r#"{"choices":[{"index":0,"delta":{"reasoning_content":"let me think"},"finish_reason":null}]}"#,
            None,
            Some("let me think"),
        ),
        (
            r#"{"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#,
            None,
            None,
        ),
    ];
    for (payload, content, reasoning) in cases {
        let parsed: ChatCompletionChunk = serde_json::from_str(payload).unwrap();
        let delta = &parsed.choices[0].delta;
        assert_eq!(delta.content.as_deref(), content, "{payload}");
        assert_eq!(delta.reasoning_content.as_deref(), reasoning, "{payload}");
        assert!(delta.tool_calls.is_none(), "{payload}");
    }
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
