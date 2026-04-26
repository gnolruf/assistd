//! Integration tests for the streaming chat client.
//!
//! These tests live behind the `test-support` feature because they rely on a
//! hand-rolled in-process HTTP/1.1 server that speaks chunked SSE. Run with:
//!     cargo test -p assistd-llm --features test-support

#![cfg(feature = "test-support")]

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use assistd_config::{ChatConfig, LlamaServerConfig, ModelConfig};
use assistd_llm::{LlamaChatClient, LlmBackend, LlmEvent, StepOutcome};
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::{Mutex, mpsc};
use tokio::task::JoinHandle;

/// Scripted behavior for a single incoming request, keyed by the `stream`
/// field of the request body.
#[derive(Clone)]
struct Script {
    /// Responses returned to streaming (`stream: true`) requests, in order.
    /// Each call pops the front of the queue; if empty the last entry is reused.
    stream_responses: Arc<Mutex<VecDeque<StreamResponse>>>,
    /// Responses returned to non-streaming (summarize, `stream: false`) calls.
    summary_responses: Arc<Mutex<VecDeque<SummaryResponse>>>,
    /// Request bodies the fake server observed.
    captured: Arc<Mutex<Vec<CapturedRequest>>>,
}

#[derive(Clone)]
struct CapturedRequest {
    path: String,
    body: Value,
    stream: bool,
}

#[derive(Clone)]
enum StreamResponse {
    /// Serve a 200 OK chunked SSE stream of these deltas, then `[DONE]`.
    Deltas(Vec<String>),
    /// Serve a 200 OK chunked stream of raw SSE frames (caller supplies the
    /// exact `data: ...\n\n` lines). Used to inject role-only first chunks.
    RawFrames(Vec<String>),
    /// Serve a 200 OK chunked stream of the deltas, then drop the connection
    /// before emitting `[DONE]`.
    DropAfterDeltas(Vec<String>),
    /// Respond with a non-200 status and the given body.
    HttpError(u16, String),
}

#[derive(Clone)]
enum SummaryResponse {
    Ok(String),
    HttpError(u16, String),
}

impl Script {
    fn new() -> Self {
        Self {
            stream_responses: Arc::new(Mutex::new(VecDeque::new())),
            summary_responses: Arc::new(Mutex::new(VecDeque::new())),
            captured: Arc::new(Mutex::new(Vec::new())),
        }
    }

    async fn push_stream(&self, r: StreamResponse) {
        self.stream_responses.lock().await.push_back(r);
    }

    async fn push_summary(&self, r: SummaryResponse) {
        self.summary_responses.lock().await.push_back(r);
    }

    async fn captured(&self) -> Vec<CapturedRequest> {
        self.captured.lock().await.clone()
    }

    async fn next_stream(&self) -> Option<StreamResponse> {
        let mut q = self.stream_responses.lock().await;
        if q.len() > 1 {
            q.pop_front()
        } else {
            q.front().cloned()
        }
    }

    async fn next_summary(&self) -> Option<SummaryResponse> {
        let mut q = self.summary_responses.lock().await;
        if q.len() > 1 {
            q.pop_front()
        } else {
            q.front().cloned()
        }
    }
}

async fn spawn_fake(script: Script) -> (u16, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let handle = tokio::spawn(async move {
        loop {
            let (sock, _) = match listener.accept().await {
                Ok(x) => x,
                Err(_) => return,
            };
            let s = script.clone();
            tokio::spawn(async move {
                if let Err(e) = handle_connection(sock, s).await {
                    eprintln!("fake chat server connection error: {e}");
                }
            });
        }
    });
    (port, handle)
}

async fn handle_connection(mut sock: tokio::net::TcpStream, script: Script) -> std::io::Result<()> {
    let mut buf = Vec::with_capacity(4096);
    let mut header_end = None;
    while header_end.is_none() {
        let mut tmp = [0u8; 2048];
        let n = sock.read(&mut tmp).await?;
        if n == 0 {
            return Ok(());
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(pos) = find_double_crlf(&buf) {
            header_end = Some(pos);
        }
        if buf.len() > 128 * 1024 {
            return Ok(());
        }
    }
    let header_end = header_end.unwrap();
    let headers_bytes = &buf[..header_end];
    let headers_str = std::str::from_utf8(headers_bytes).unwrap_or("");

    let mut lines = headers_str.split("\r\n");
    let request_line = lines.next().unwrap_or("");
    let path = request_line
        .split_whitespace()
        .nth(1)
        .unwrap_or("/")
        .to_string();

    let mut content_length: usize = 0;
    for line in lines {
        if let Some((name, value)) = line.split_once(':')
            && name.eq_ignore_ascii_case("content-length")
        {
            content_length = value.trim().parse().unwrap_or(0);
        }
    }

    let body_start = header_end + 4;
    while buf.len() < body_start + content_length {
        let mut tmp = [0u8; 2048];
        let n = sock.read(&mut tmp).await?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
    }
    let body_bytes = &buf[body_start..body_start + content_length.min(buf.len() - body_start)];
    let body_json: Value = serde_json::from_slice(body_bytes).unwrap_or(Value::Null);
    let is_stream = body_json
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    script.captured.lock().await.push(CapturedRequest {
        path: path.clone(),
        body: body_json,
        stream: is_stream,
    });

    if is_stream {
        let response = script.next_stream().await.unwrap_or_else(|| {
            StreamResponse::HttpError(500, "no scripted stream response".into())
        });
        handle_stream_response(&mut sock, response).await?;
    } else {
        let response = script.next_summary().await.unwrap_or_else(|| {
            SummaryResponse::HttpError(500, "no scripted summary response".into())
        });
        handle_summary_response(&mut sock, response).await?;
    }
    Ok(())
}

async fn handle_stream_response(
    sock: &mut tokio::net::TcpStream,
    resp: StreamResponse,
) -> std::io::Result<()> {
    match resp {
        StreamResponse::Deltas(deltas) => {
            write_sse_headers(sock).await?;
            for d in deltas {
                let frame = format!(
                    "data: {{\"choices\":[{{\"delta\":{{\"content\":{}}}}}]}}\n\n",
                    serde_json::to_string(&d).unwrap()
                );
                write_chunk(sock, frame.as_bytes()).await?;
            }
            write_chunk(sock, b"data: [DONE]\n\n").await?;
            write_final_chunk(sock).await?;
        }
        StreamResponse::RawFrames(frames) => {
            write_sse_headers(sock).await?;
            for f in frames {
                write_chunk(sock, f.as_bytes()).await?;
            }
            write_final_chunk(sock).await?;
        }
        StreamResponse::DropAfterDeltas(deltas) => {
            write_sse_headers(sock).await?;
            for d in deltas {
                let frame = format!(
                    "data: {{\"choices\":[{{\"delta\":{{\"content\":{}}}}}]}}\n\n",
                    serde_json::to_string(&d).unwrap()
                );
                write_chunk(sock, frame.as_bytes()).await?;
            }
            // Deliberately do not write [DONE] or a final chunk — just close.
        }
        StreamResponse::HttpError(status, body) => {
            write_error(sock, status, &body).await?;
        }
    }
    let _ = sock.shutdown().await;
    Ok(())
}

async fn handle_summary_response(
    sock: &mut tokio::net::TcpStream,
    resp: SummaryResponse,
) -> std::io::Result<()> {
    match resp {
        SummaryResponse::Ok(text) => {
            let body = format!(
                "{{\"choices\":[{{\"message\":{{\"role\":\"assistant\",\"content\":{}}}}}]}}",
                serde_json::to_string(&text).unwrap()
            );
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            sock.write_all(response.as_bytes()).await?;
        }
        SummaryResponse::HttpError(status, body) => {
            write_error(sock, status, &body).await?;
        }
    }
    let _ = sock.shutdown().await;
    Ok(())
}

async fn write_sse_headers(sock: &mut tokio::net::TcpStream) -> std::io::Result<()> {
    let headers = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n";
    sock.write_all(headers).await
}

async fn write_chunk(sock: &mut tokio::net::TcpStream, payload: &[u8]) -> std::io::Result<()> {
    let header = format!("{:x}\r\n", payload.len());
    sock.write_all(header.as_bytes()).await?;
    sock.write_all(payload).await?;
    sock.write_all(b"\r\n").await?;
    Ok(())
}

async fn write_final_chunk(sock: &mut tokio::net::TcpStream) -> std::io::Result<()> {
    sock.write_all(b"0\r\n\r\n").await
}

async fn write_error(
    sock: &mut tokio::net::TcpStream,
    status: u16,
    body: &str,
) -> std::io::Result<()> {
    let reason = match status {
        400 => "Bad Request",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        _ => "Error",
    };
    let resp = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        status,
        reason,
        body.len(),
        body
    );
    sock.write_all(resp.as_bytes()).await
}

fn find_double_crlf(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

struct ClientCfg {
    chat: ChatConfig,
    server: LlamaServerConfig,
    model: ModelConfig,
}

fn chat_spec(port: u16) -> ClientCfg {
    ClientCfg {
        chat: ChatConfig {
            system_prompt: "test system prompt".into(),
            max_history_tokens: 10_000,
            summary_target_tokens: 1000,
            preserve_recent_turns: 2,
            temperature: 0.5,
            max_response_tokens: 256,
            max_summary_tokens: 500,
            request_timeout_secs: 5,
            summary_temperature: 0.3,
            top_p: None,
            top_k: None,
            min_p: None,
            presence_penalty: None,
        },
        server: LlamaServerConfig {
            binary_path: "llama-server".into(),
            host: "127.0.0.1".into(),
            port,
            gpu_layers: 9999,
            ready_timeout_secs: 60,
            alias: None,
            override_tensor: None,
            flash_attn: None,
            cache_type_k: None,
            cache_type_v: None,
            threads: None,
            batch_size: None,
            ubatch_size: None,
            n_cpu_moe: None,
            cache_ram_mib: None,
            mlock: None,
            mmproj_offload: None,
        },
        model: ModelConfig {
            name: "test-model".into(),
            context_length: 12_000,
        },
    }
}

fn build_client(cfg: &ClientCfg) -> LlamaChatClient {
    LlamaChatClient::new(&cfg.chat, &cfg.server, &cfg.model).unwrap()
}

async fn drain(rx: &mut mpsc::Receiver<LlmEvent>) -> Vec<LlmEvent> {
    let mut out = Vec::new();
    while let Some(ev) = rx.recv().await {
        out.push(ev);
    }
    out
}

#[tokio::test]
async fn single_turn_streams_deltas_and_finishes() {
    let script = Script::new();
    script
        .push_stream(StreamResponse::Deltas(vec![
            "Hello".into(),
            " ".into(),
            "world".into(),
        ]))
        .await;
    let (port, _server) = spawn_fake(script.clone()).await;

    let client = build_client(&chat_spec(port));
    let (tx, mut rx) = mpsc::channel(32);
    client.generate("hi".into(), tx).await.unwrap();

    let events = drain(&mut rx).await;
    assert_eq!(events.len(), 4);
    assert_eq!(
        events[0],
        LlmEvent::Delta {
            text: "Hello".into()
        }
    );
    assert_eq!(events[1], LlmEvent::Delta { text: " ".into() });
    assert_eq!(
        events[2],
        LlmEvent::Delta {
            text: "world".into()
        }
    );
    assert_eq!(events[3], LlmEvent::Done);

    let captured = script.captured().await;
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].path, "/v1/chat/completions");
    assert!(captured[0].stream);
    let messages = captured[0].body["messages"].as_array().unwrap();
    assert_eq!(messages[0]["role"], "system");
    assert_eq!(messages[0]["content"], "test system prompt");
    assert_eq!(messages[1]["role"], "user");
    assert_eq!(messages[1]["content"], "hi");
}

#[tokio::test]
async fn multi_turn_request_includes_prior_exchange() {
    let script = Script::new();
    script
        .push_stream(StreamResponse::Deltas(vec![
            "first".into(),
            " reply".into(),
        ]))
        .await;
    script
        .push_stream(StreamResponse::Deltas(vec!["second reply".into()]))
        .await;
    let (port, _server) = spawn_fake(script.clone()).await;

    let client = build_client(&chat_spec(port));

    let (tx1, mut rx1) = mpsc::channel(32);
    client.generate("question one".into(), tx1).await.unwrap();
    drain(&mut rx1).await;

    let (tx2, mut rx2) = mpsc::channel(32);
    client.generate("question two".into(), tx2).await.unwrap();
    drain(&mut rx2).await;

    let captured = script.captured().await;
    assert_eq!(captured.len(), 2);
    let second_messages = captured[1].body["messages"].as_array().unwrap();
    // system, user1, assistant1, user2
    assert_eq!(second_messages.len(), 4);
    assert_eq!(second_messages[0]["role"], "system");
    assert_eq!(second_messages[1]["role"], "user");
    assert_eq!(second_messages[1]["content"], "question one");
    assert_eq!(second_messages[2]["role"], "assistant");
    assert_eq!(second_messages[2]["content"], "first reply");
    assert_eq!(second_messages[3]["role"], "user");
    assert_eq!(second_messages[3]["content"], "question two");
}

#[tokio::test]
async fn connection_refused_returns_typed_error_not_panic() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);

    let mut spec = chat_spec(port);
    spec.chat.request_timeout_secs = 2;
    let client = build_client(&spec);
    let (tx, mut rx) = mpsc::channel(32);
    let result = client.generate("hi".into(), tx).await;
    assert!(result.is_err(), "expected error for connection refused");
    let msg = format!("{:#}", result.unwrap_err());
    assert!(
        msg.contains("HTTP transport error") || msg.to_lowercase().contains("connect"),
        "unexpected error message: {msg}"
    );
    assert!(drain(&mut rx).await.is_empty());
}

#[tokio::test]
async fn http_500_returns_server_error() {
    let script = Script::new();
    script
        .push_stream(StreamResponse::HttpError(500, "oom".into()))
        .await;
    let (port, _server) = spawn_fake(script).await;

    let client = build_client(&chat_spec(port));
    let (tx, mut rx) = mpsc::channel(32);
    let result = client.generate("hi".into(), tx).await;
    assert!(result.is_err());
    let msg = format!("{:#}", result.unwrap_err());
    assert!(msg.contains("500"), "unexpected error message: {msg}");
    assert!(drain(&mut rx).await.is_empty());
}

#[tokio::test]
async fn mid_stream_drop_after_deltas_emits_done() {
    let script = Script::new();
    script
        .push_stream(StreamResponse::DropAfterDeltas(vec![
            "partial".into(),
            " reply".into(),
        ]))
        .await;
    script
        .push_stream(StreamResponse::Deltas(vec!["final".into()]))
        .await;
    let (port, _server) = spawn_fake(script.clone()).await;

    let client = build_client(&chat_spec(port));

    let (tx, mut rx) = mpsc::channel(32);
    client.generate("hi".into(), tx).await.unwrap();
    let events = drain(&mut rx).await;
    assert!(
        events
            .iter()
            .any(|e| matches!(e, LlmEvent::Delta { text } if text == "partial"))
    );
    assert!(events.iter().any(|e| matches!(e, LlmEvent::Done)));

    let (tx2, mut rx2) = mpsc::channel(32);
    client.generate("followup".into(), tx2).await.unwrap();
    drain(&mut rx2).await;

    let captured = script.captured().await;
    assert_eq!(captured.len(), 2);
    let second_messages = captured[1].body["messages"].as_array().unwrap();
    let assistant_before = second_messages
        .iter()
        .find(|m| m["role"] == "assistant")
        .expect("assistant message carried forward");
    assert_eq!(assistant_before["content"], "partial reply");
}

#[tokio::test]
async fn first_chunk_role_only_delta_is_ignored() {
    let script = Script::new();
    script
        .push_stream(StreamResponse::RawFrames(vec![
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n".into(),
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n".into(),
            "data: [DONE]\n\n".into(),
        ]))
        .await;
    let (port, _server) = spawn_fake(script).await;

    let client = build_client(&chat_spec(port));
    let (tx, mut rx) = mpsc::channel(32);
    client.generate("hi".into(), tx).await.unwrap();
    let events = drain(&mut rx).await;
    let deltas: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            LlmEvent::Delta { text } => Some(text.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(deltas, vec!["hi"]);
    assert!(matches!(events.last(), Some(LlmEvent::Done)));
}

#[tokio::test]
async fn summarization_triggered_when_over_budget() {
    let script = Script::new();
    // Enough deltas to blow the tiny budget on the next turn.
    let long_reply: String = "long ".repeat(30);
    script
        .push_stream(StreamResponse::Deltas(vec![long_reply.clone()]))
        .await;
    script
        .push_stream(StreamResponse::Deltas(vec![long_reply.clone()]))
        .await;
    script
        .push_stream(StreamResponse::Deltas(vec!["final".into()]))
        .await;
    script
        .push_summary(SummaryResponse::Ok(
            "prior conversation covered various topics".into(),
        ))
        .await;
    let (port, _server) = spawn_fake(script.clone()).await;

    let mut spec = chat_spec(port);
    spec.chat.max_history_tokens = 60;
    spec.chat.summary_target_tokens = 15;
    spec.chat.preserve_recent_turns = 1;
    let client = build_client(&spec);

    for i in 0..3 {
        let (tx, mut rx) = mpsc::channel(32);
        client
            .generate(format!("turn {i} with padding"), tx)
            .await
            .unwrap();
        drain(&mut rx).await;
    }

    let captured = script.captured().await;
    assert!(
        captured.iter().any(|r| !r.stream),
        "expected at least one non-streaming summarize request"
    );

    let last_stream = captured
        .iter()
        .rev()
        .find(|r| r.stream)
        .expect("at least one stream request");
    let messages = last_stream.body["messages"].as_array().unwrap();
    assert!(
        messages.iter().any(|m| {
            m["role"] == "system"
                && m["content"]
                    .as_str()
                    .map(|s| s.contains("[Conversation summary]"))
                    .unwrap_or(false)
        }),
        "final request should include the synthetic summary message"
    );
}

#[tokio::test]
async fn summarize_failure_falls_back_to_truncation_and_still_responds() {
    let script = Script::new();
    let long_reply: String = "padding word ".repeat(20);
    script
        .push_stream(StreamResponse::Deltas(vec![long_reply.clone()]))
        .await;
    script
        .push_stream(StreamResponse::Deltas(vec![long_reply.clone()]))
        .await;
    script
        .push_stream(StreamResponse::Deltas(vec!["final".into()]))
        .await;
    script
        .push_summary(SummaryResponse::HttpError(500, "boom".into()))
        .await;
    let (port, _server) = spawn_fake(script.clone()).await;

    let mut spec = chat_spec(port);
    spec.chat.max_history_tokens = 50;
    spec.chat.summary_target_tokens = 10;
    spec.chat.preserve_recent_turns = 1;
    let client = build_client(&spec);

    for i in 0..3 {
        let (tx, mut rx) = mpsc::channel(32);
        let result = client.generate(format!("turn {i}"), tx).await;
        assert!(
            result.is_ok(),
            "generate should not fail despite summarize error"
        );
        let events = drain(&mut rx).await;
        assert!(matches!(events.last(), Some(LlmEvent::Done)));
    }

    let captured = script.captured().await;
    assert!(
        captured.iter().any(|r| !r.stream),
        "summarize endpoint should have been attempted"
    );
}

// ---------------------------------------------------------------------------
// Agent-loop step API
// ---------------------------------------------------------------------------

/// Frame the `tool_calls` path as a sequence of SSE payloads suitable for
/// `StreamResponse::RawFrames`. Matches llama.cpp's typical shape: role
/// first, then one or more tool_call deltas with accumulating `arguments`,
/// terminated by a finish_reason chunk.
fn tool_call_frames(call_id: &str, name: &str, arg_chunks: &[&str]) -> Vec<String> {
    let mut frames = Vec::new();
    frames.push("data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n".to_string());
    let head = format!(
        "data: {{\"choices\":[{{\"delta\":{{\"tool_calls\":[{{\"index\":0,\"id\":{},\"type\":\"function\",\"function\":{{\"name\":{},\"arguments\":\"\"}}}}]}}}}]}}\n\n",
        serde_json::to_string(call_id).unwrap(),
        serde_json::to_string(name).unwrap()
    );
    frames.push(head);
    for chunk in arg_chunks {
        let encoded = serde_json::to_string(chunk).unwrap();
        frames.push(format!(
            "data: {{\"choices\":[{{\"delta\":{{\"tool_calls\":[{{\"index\":0,\"function\":{{\"arguments\":{}}}}}]}}}}]}}\n\n",
            encoded
        ));
    }
    frames.push(
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n".to_string(),
    );
    frames.push("data: [DONE]\n\n".to_string());
    frames
}

#[tokio::test]
async fn step_with_stop_finish_reason_returns_final() {
    let script = Script::new();
    script
        .push_stream(StreamResponse::Deltas(vec!["answer".into()]))
        .await;
    let (port, _server) = spawn_fake(script.clone()).await;

    let client = build_client(&chat_spec(port));
    client.push_user("what is 2+2?".into(), Vec::new()).await.unwrap();
    let (tx, mut rx) = mpsc::channel(32);
    let outcome = client.step(Vec::new(), tx).await.unwrap();
    assert!(matches!(outcome, StepOutcome::Final));
    let events = drain(&mut rx).await;
    assert_eq!(
        events,
        vec![LlmEvent::Delta {
            text: "answer".into()
        }]
    );
    // No tools were passed — request should omit the `tools` key entirely.
    let captured = script.captured().await;
    assert_eq!(captured.len(), 1);
    assert!(
        captured[0].body.get("tools").is_none(),
        "request should not carry tools when argument is empty"
    );
}

#[tokio::test]
async fn step_parses_tool_call_across_argument_chunks() {
    let script = Script::new();
    script
        .push_stream(StreamResponse::RawFrames(tool_call_frames(
            "call-42",
            "run",
            &[r#"{"com"#, r#"mand":"ls "#, r#"/tmp"}"#],
        )))
        .await;
    let (port, _server) = spawn_fake(script.clone()).await;

    let client = build_client(&chat_spec(port));
    client.push_user("list /tmp".into(), Vec::new()).await.unwrap();
    let (tx, mut rx) = mpsc::channel(32);
    let tools = vec![serde_json::json!({
        "type": "function",
        "function": {
            "name": "run",
            "description": "run a command",
            "parameters": {"type":"object","properties":{"command":{"type":"string"}}},
            "strict": true
        }
    })];
    let outcome = client.step(tools, tx).await.unwrap();
    let calls = match outcome {
        StepOutcome::ToolCalls(c) => c,
        other => panic!("expected ToolCalls, got {other:?}"),
    };
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id, "call-42");
    assert_eq!(calls[0].name, "run");
    assert_eq!(calls[0].arguments["command"], "ls /tmp");

    // No visible text deltas for a tool-calls-only step — Qwen3-style
    // <think> blocks would have been emitted via `content`, but none
    // appeared in our scripted frames.
    let events = drain(&mut rx).await;
    assert!(
        !events.iter().any(|e| matches!(e, LlmEvent::Delta { .. })),
        "tool-call-only step should emit no text deltas, got {events:?}"
    );

    let captured = script.captured().await;
    assert_eq!(captured[0].body["tool_choice"], "auto");
    let tools_arr = captured[0].body["tools"].as_array().unwrap();
    assert_eq!(tools_arr[0]["function"]["name"], "run");
}

#[tokio::test]
async fn agent_round_trip_commits_tool_calls_and_result_to_history() {
    let script = Script::new();
    // Turn 1: model asks for a tool call.
    script
        .push_stream(StreamResponse::RawFrames(tool_call_frames(
            "call-7",
            "run",
            &[r#"{"command":"echo hi"}"#],
        )))
        .await;
    // Turn 2 (after we push_tool_results): plain text answer.
    script
        .push_stream(StreamResponse::Deltas(vec!["done".into()]))
        .await;
    let (port, _server) = spawn_fake(script.clone()).await;

    let client = build_client(&chat_spec(port));
    client.push_user("please echo hi".into(), Vec::new()).await.unwrap();

    let tools = vec![serde_json::json!({
        "type": "function",
        "function": {"name":"run","parameters":{"type":"object"},"strict":true}
    })];
    let (tx1, mut rx1) = mpsc::channel(32);
    let outcome1 = client.step(tools.clone(), tx1).await.unwrap();
    let calls = match outcome1 {
        StepOutcome::ToolCalls(c) => c,
        _ => panic!("expected ToolCalls"),
    };
    drain(&mut rx1).await;

    // Simulate the agent-loop side of things: feed a result back.
    let result = assistd_llm::ToolResultPayload {
        call_id: calls[0].id.clone(),
        name: "run".into(),
        content: "hi\n[exit:0 | 2ms]".into(),
        attachments: Vec::new(),
    };
    client.push_tool_results(vec![result]).await.unwrap();

    let (tx2, mut rx2) = mpsc::channel(32);
    let outcome2 = client.step(tools, tx2).await.unwrap();
    assert!(matches!(outcome2, StepOutcome::Final));
    drain(&mut rx2).await;

    // Inspect the wire shape of the second request — it must carry the
    // assistant tool_calls message AND the tool-result user message.
    let captured = script.captured().await;
    assert_eq!(captured.len(), 2);
    let messages = captured[1].body["messages"].as_array().unwrap();

    let assistant_with_calls = messages
        .iter()
        .find(|m| m["role"] == "assistant" && m.get("tool_calls").is_some())
        .expect("assistant message with tool_calls");
    assert!(
        assistant_with_calls.get("content").is_none(),
        "content key must be absent on tool-call assistant message"
    );
    let tool_calls = assistant_with_calls["tool_calls"].as_array().unwrap();
    assert_eq!(tool_calls[0]["id"], "call-7");
    assert_eq!(tool_calls[0]["function"]["name"], "run");

    // The tool result should be routed as a user message with the
    // [tool:run] prefix (per the design: user-role routing for tool
    // results instead of the OpenAI tool role).
    let tool_result_user = messages
        .iter()
        .rev()
        .find(|m| {
            m["role"] == "user"
                && m["content"]
                    .as_str()
                    .map(|s| s.starts_with("[tool:run]"))
                    .unwrap_or(false)
        })
        .expect("tool result user message");
    assert!(
        tool_result_user["content"]
            .as_str()
            .unwrap()
            .contains("[exit:0 | 2ms]"),
        "expected footer: {:?}",
        tool_result_user["content"]
    );
}

#[tokio::test]
async fn request_timeout_surfaces_as_error() {
    // Bind but never respond: reqwest will time out on read.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        loop {
            if let Ok((sock, _)) = listener.accept().await {
                // Hold the socket open without writing anything.
                tokio::time::sleep(Duration::from_secs(30)).await;
                drop(sock);
            }
        }
    });

    let mut spec = chat_spec(port);
    spec.chat.request_timeout_secs = 1;
    let client = build_client(&spec);
    let (tx, _rx) = mpsc::channel(32);
    let result = client.generate("hi".into(), tx).await;
    assert!(result.is_err());
}
