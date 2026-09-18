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
        chat_template_kwargs: None,
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
        chat_template_kwargs: None,
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
        chat_template_kwargs: None,
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
    assert!(parsed.choices[0].delta.reasoning_content.is_none());
}

#[test]
fn deserializes_reasoning_content_delta() {
    // llama.cpp's separated-reasoning shape: `reasoning_content`
    // alongside (or in place of) `content`.
    let payload = r#"{"choices":[{"index":0,"delta":{"reasoning_content":"let me think"},"finish_reason":null}]}"#;
    let parsed: ChatCompletionChunk = serde_json::from_str(payload).unwrap();
    assert_eq!(
        parsed.choices[0].delta.reasoning_content.as_deref(),
        Some("let me think")
    );
    assert!(parsed.choices[0].delta.content.is_none());
}

#[test]
fn deserializes_content_with_inline_think_tags() {
    // Inline-reasoning shape: the model emits `<think>...</think>`
    // verbatim inside `content`. We parse it as a plain string
    // here; the splitter in client.rs is what classifies it.
    let payload = r#"{"choices":[{"index":0,"delta":{"content":"<think>maybe</think>4"},"finish_reason":null}]}"#;
    let parsed: ChatCompletionChunk = serde_json::from_str(payload).unwrap();
    assert_eq!(
        parsed.choices[0].delta.content.as_deref(),
        Some("<think>maybe</think>4")
    );
}

#[test]
fn deserializes_role_only_first_chunk() {
    let payload = r#"{"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#;
    let parsed: ChatCompletionChunk = serde_json::from_str(payload).unwrap();
    assert_eq!(parsed.choices[0].delta.role.as_deref(), Some("assistant"));
    assert!(parsed.choices[0].delta.content.is_none());
}

#[test]
fn deserializes_tool_call_delta_chunk() {
    // llama.cpp's typical tool-call emission: id + name in one chunk,
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
