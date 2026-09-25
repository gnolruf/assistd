//! Integration tests for the streaming chat client against a hand-rolled
//! in-process HTTP/1.1 server that speaks chunked SSE.

#![cfg(feature = "test-support")]

use std::collections::VecDeque;
use std::io;
use std::net::Ipv4Addr;
use std::num::NonZeroU16;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Mutex, mpsc};
use tokio::task::JoinHandle;
use tokio::time::timeout;

use assistd_config::defaults::{nz32, nz64};
use assistd_config::{ChatConfig, LlamaServerConfig, ModelConfig, TimeoutsConfig};
use assistd_llm::{
    ChatClientError, LlamaChatClient, LlmBackend, LlmError, LlmEvent, StepOutcome, Thinking,
    ToolCall, ToolResultPayload,
};

const MAX_REQUEST_HEADER_BYTES: usize = 128 * 1024;

/// Scripted replies for the fake server, chosen by the request's `stream`
/// field. Each request pops the front reply; the last one is reused.
#[derive(Clone)]
struct Script {
    stream_responses: Arc<Mutex<VecDeque<StreamResponse>>>,
    summary_responses: Arc<Mutex<VecDeque<SummaryResponse>>>,
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
    /// The deltas, then `[DONE]`.
    Deltas(Vec<String>),
    /// These exact `data: ...\n\n` frames.
    RawFrames(Vec<String>),
    /// The deltas, then close without `[DONE]`.
    DropAfterDeltas(Vec<String>),
    /// The deltas, then silence past any client-side inactivity timeout.
    StallAfterDeltas(Vec<String>),
    /// The deltas with the given gap before each one, then `[DONE]`.
    PacedDeltas(Vec<String>, Duration),
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

    async fn push_stream(&self, response: StreamResponse) {
        self.stream_responses.lock().await.push_back(response);
    }

    async fn push_summary(&self, response: SummaryResponse) {
        self.summary_responses.lock().await.push_back(response);
    }

    async fn captured(&self) -> Vec<CapturedRequest> {
        self.captured.lock().await.clone()
    }

    async fn next_stream(&self) -> Option<StreamResponse> {
        let mut queue = self.stream_responses.lock().await;
        if queue.len() > 1 {
            queue.pop_front()
        } else {
            queue.front().cloned()
        }
    }

    async fn next_summary(&self) -> Option<SummaryResponse> {
        let mut queue = self.summary_responses.lock().await;
        if queue.len() > 1 {
            queue.pop_front()
        } else {
            queue.front().cloned()
        }
    }
}

async fn spawn_fake(script: Script) -> (u16, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = tokio::spawn(async move {
        loop {
            let (sock, _) = match listener.accept().await {
                Ok(accepted) => accepted,
                Err(_) => return,
            };
            let script = script.clone();
            tokio::spawn(async move {
                let _ = serve_connection(sock, script).await;
            });
        }
    });
    (port, server)
}

async fn serve_connection(mut sock: TcpStream, script: Script) -> io::Result<()> {
    let Some((path, body_json)) = read_request(&mut sock).await? else {
        return Ok(());
    };
    let is_stream = body_json
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    script.captured.lock().await.push(CapturedRequest {
        path,
        body: body_json,
        stream: is_stream,
    });

    if is_stream {
        let response = script.next_stream().await.unwrap_or_else(|| {
            StreamResponse::HttpError(500, "no scripted stream response".into())
        });
        write_stream_response(&mut sock, response).await?;
    } else {
        let response = script.next_summary().await.unwrap_or_else(|| {
            SummaryResponse::HttpError(500, "no scripted summary response".into())
        });
        write_summary_response(&mut sock, response).await?;
    }
    Ok(())
}

/// Read one request's path and JSON body, or `None` if the client hung up
/// or sent oversized headers.
async fn read_request(sock: &mut TcpStream) -> io::Result<Option<(String, Value)>> {
    let mut buf = Vec::with_capacity(4096);
    let mut header_end = None;
    while header_end.is_none() {
        let mut read_buf = [0u8; 2048];
        let n = sock.read(&mut read_buf).await?;
        if n == 0 {
            return Ok(None);
        }
        buf.extend_from_slice(&read_buf[..n]);
        if let Some(pos) = find_double_crlf(&buf) {
            header_end = Some(pos);
        }
        if buf.len() > MAX_REQUEST_HEADER_BYTES {
            return Ok(None);
        }
    }
    let header_end = header_end.unwrap();
    let headers_str = std::str::from_utf8(&buf[..header_end]).unwrap_or("");

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
        let mut read_buf = [0u8; 2048];
        let n = sock.read(&mut read_buf).await?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&read_buf[..n]);
    }
    let body_bytes = &buf[body_start..body_start + content_length.min(buf.len() - body_start)];
    let body_json = serde_json::from_slice(body_bytes).unwrap_or(Value::Null);
    Ok(Some((path, body_json)))
}

async fn write_stream_response(sock: &mut TcpStream, response: StreamResponse) -> io::Result<()> {
    match response {
        StreamResponse::Deltas(deltas) => {
            write_sse_headers(sock).await?;
            for text in deltas {
                write_delta(sock, &text).await?;
            }
            write_chunk(sock, b"data: [DONE]\n\n").await?;
            write_final_chunk(sock).await?;
        }
        StreamResponse::RawFrames(frames) => {
            write_sse_headers(sock).await?;
            for frame in frames {
                write_chunk(sock, frame.as_bytes()).await?;
            }
            write_final_chunk(sock).await?;
        }
        StreamResponse::DropAfterDeltas(deltas) => {
            write_sse_headers(sock).await?;
            for text in deltas {
                write_delta(sock, &text).await?;
            }
        }
        StreamResponse::StallAfterDeltas(deltas) => {
            write_sse_headers(sock).await?;
            for text in deltas {
                write_delta(sock, &text).await?;
            }
            tokio::time::sleep(Duration::from_secs(60)).await;
        }
        StreamResponse::PacedDeltas(deltas, gap) => {
            write_sse_headers(sock).await?;
            for text in deltas {
                tokio::time::sleep(gap).await;
                write_delta(sock, &text).await?;
            }
            write_chunk(sock, b"data: [DONE]\n\n").await?;
            write_final_chunk(sock).await?;
        }
        StreamResponse::HttpError(status, body) => {
            write_error(sock, status, &body).await?;
        }
    }
    let _ = sock.shutdown().await;
    Ok(())
}

async fn write_summary_response(sock: &mut TcpStream, response: SummaryResponse) -> io::Result<()> {
    match response {
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

async fn write_sse_headers(sock: &mut TcpStream) -> io::Result<()> {
    let headers = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n";
    sock.write_all(headers).await
}

async fn write_delta(sock: &mut TcpStream, text: &str) -> io::Result<()> {
    let frame = format!(
        "data: {{\"choices\":[{{\"delta\":{{\"content\":{}}}}}]}}\n\n",
        serde_json::to_string(text).unwrap()
    );
    write_chunk(sock, frame.as_bytes()).await
}

async fn write_chunk(sock: &mut TcpStream, payload: &[u8]) -> io::Result<()> {
    let header = format!("{:x}\r\n", payload.len());
    sock.write_all(header.as_bytes()).await?;
    sock.write_all(payload).await?;
    sock.write_all(b"\r\n").await?;
    Ok(())
}

async fn write_final_chunk(sock: &mut TcpStream) -> io::Result<()> {
    sock.write_all(b"0\r\n\r\n").await
}

async fn write_error(sock: &mut TcpStream, status: u16, body: &str) -> io::Result<()> {
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
    timeouts: TimeoutsConfig,
}

fn chat_spec(port: u16) -> ClientCfg {
    ClientCfg {
        chat: ChatConfig {
            system_prompt: "test system prompt".into(),
            max_history_tokens: nz32(10_000),
            summary_target_tokens: nz32(1000),
            preserve_recent_turns: nz32(2),
            temperature: 0.5,
            max_response_tokens: nz32(256),
            request_timeout_secs: nz64(5),
            summary_temperature: 0.3,
            top_p: None,
            top_k: None,
            min_p: None,
            presence_penalty: None,
        },
        server: LlamaServerConfig {
            binary_path: "llama-server".into(),
            host: Ipv4Addr::LOCALHOST.into(),
            port: NonZeroU16::new(port).expect("bound port is never 0"),
            gpu_layers: 9999,
            ready_timeout_secs: nz64(60),
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
            context_length: nz32(12_000),
        },
        timeouts: TimeoutsConfig::default(),
    }
}

fn delta(text: &str) -> LlmEvent {
    LlmEvent::Delta { text: text.into() }
}

fn build_client(cfg: &ClientCfg) -> LlamaChatClient {
    LlamaChatClient::new(&cfg.chat, &cfg.server, &cfg.model, &cfg.timeouts, None).unwrap()
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

    assert_eq!(
        drain(&mut rx).await,
        [delta("Hello"), delta(" "), delta("world"), LlmEvent::Done]
    );

    let captured = script.captured().await;
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].path, "/v1/chat/completions");
    assert!(captured[0].stream);
    assert_eq!(
        captured[0].body["messages"],
        json!([
            {"role": "system", "content": "test system prompt"},
            {"role": "user", "content": "hi"},
        ])
    );
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
    assert_eq!(
        captured[1].body["messages"],
        json!([
            {"role": "system", "content": "test system prompt"},
            {"role": "user", "content": "question one"},
            {"role": "assistant", "content": "first reply"},
            {"role": "user", "content": "question two"},
        ])
    );
}

#[tokio::test]
async fn connection_refused_returns_typed_error_not_panic() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);

    let mut spec = chat_spec(port);
    spec.chat.request_timeout_secs = nz64(2);
    let client = build_client(&spec);
    let (tx, mut rx) = mpsc::channel(32);
    let err = client.generate("hi".into(), tx).await.unwrap_err();
    assert!(
        matches!(err, LlmError::Chat(ChatClientError::Http(_))),
        "{err:?}"
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
    let err = client.generate("hi".into(), tx).await.unwrap_err();
    assert!(
        matches!(&err, LlmError::Chat(ChatClientError::Server { status: 500, body }) if body == "oom"),
        "{err:?}"
    );
    assert!(drain(&mut rx).await.is_empty());
}

#[tokio::test]
async fn conv_lock_is_not_held_while_streaming() {
    let script = Script::new();
    let many: Vec<String> = (0..200).map(|i| format!("d{i}")).collect();
    script.push_stream(StreamResponse::Deltas(many)).await;
    let (port, _server) = spawn_fake(script).await;

    let client = Arc::new(build_client(&chat_spec(port)));
    // One slot: once the test stops draining, the stream parks mid-flight.
    let (tx, mut rx) = mpsc::channel(1);
    let gen_client = Arc::clone(&client);
    let stream_task = tokio::spawn(async move { gen_client.generate("first".into(), tx).await });
    rx.recv().await.expect("first delta");

    timeout(
        Duration::from_secs(2),
        client.set_transient_context("hello".into()),
    )
    .await
    .expect("set_transient_context must not wait on an in-flight stream")
    .expect("set_transient_context must succeed");

    while rx.recv().await.is_some() {}
    stream_task
        .await
        .expect("stream task joined")
        .expect("generate completed");
}

#[tokio::test]
async fn stalled_stream_aborts_within_inactivity_timeout() {
    let script = Script::new();
    script
        .push_stream(StreamResponse::StallAfterDeltas(vec![
            "hello".into(),
            " ".into(),
        ]))
        .await;
    let (port, _server) = spawn_fake(script).await;

    let mut spec = chat_spec(port);
    spec.timeouts.stream_inactivity_secs = 1;
    let client = build_client(&spec);

    let (tx, mut rx) = mpsc::channel(32);
    let started = Instant::now();
    let res = timeout(Duration::from_secs(5), client.generate("hi".into(), tx))
        .await
        .expect("generate must return within outer 5s budget");
    res.expect("partial-after-emit path returns Ok");
    let elapsed = started.elapsed();
    assert!(
        elapsed < Duration::from_secs(4),
        "generate took too long ({elapsed:?}); inactivity timeout did not fire"
    );

    assert_eq!(
        drain(&mut rx).await,
        [delta("hello"), delta(" "), LlmEvent::Done]
    );
}

#[tokio::test]
async fn slow_first_token_is_not_treated_as_a_stall() {
    let script = Script::new();
    script
        .push_stream(StreamResponse::StallAfterDeltas(Vec::new()))
        .await;
    let (port, _server) = spawn_fake(script).await;

    let mut spec = chat_spec(port);
    spec.timeouts.stream_inactivity_secs = 1;
    spec.chat.request_timeout_secs = nz64(4);
    let client = build_client(&spec);

    let (tx, mut rx) = mpsc::channel(32);
    let started = Instant::now();
    let res = timeout(Duration::from_secs(15), client.generate("hi".into(), tx))
        .await
        .expect("generate must return within outer 15s budget");
    let err = res.expect_err("a first byte that never arrives is still an error");
    assert!(
        matches!(err, LlmError::Chat(ChatClientError::Sse(_))),
        "{err:?}"
    );
    let elapsed = started.elapsed();
    assert!(
        elapsed >= Duration::from_secs(3),
        "generate returned after {elapsed:?}; the inter-chunk deadline fired before the first byte"
    );
    assert!(drain(&mut rx).await.is_empty());
}

#[tokio::test]
async fn generation_longer_than_request_timeout_is_not_truncated() {
    let script = Script::new();
    let deltas: Vec<String> = ["one", " two", " three", " four", " five"]
        .map(String::from)
        .to_vec();
    script
        .push_stream(StreamResponse::PacedDeltas(
            deltas.clone(),
            Duration::from_millis(400),
        ))
        .await;
    let (port, _server) = spawn_fake(script).await;

    let mut spec = chat_spec(port);
    spec.chat.request_timeout_secs = nz64(1);
    spec.timeouts.stream_inactivity_secs = 2;
    let client = build_client(&spec);

    let (tx, mut rx) = mpsc::channel(32);
    timeout(Duration::from_secs(10), client.generate("hi".into(), tx))
        .await
        .expect("generate must return within outer 10s budget")
        .expect("generate completed");

    let texts: Vec<_> = drain(&mut rx)
        .await
        .into_iter()
        .filter_map(|e| match e {
            LlmEvent::Delta { text } => Some(text),
            _ => None,
        })
        .collect();
    assert_eq!(texts, deltas);
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
    assert_eq!(
        drain(&mut rx).await,
        [delta("partial"), delta(" reply"), LlmEvent::Done]
    );

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
    assert_eq!(drain(&mut rx).await, [delta("hi"), LlmEvent::Done]);
}

#[tokio::test]
async fn summarization_triggered_when_over_budget() {
    let script = Script::new();
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
    spec.chat.max_history_tokens = nz32(60);
    spec.chat.summary_target_tokens = nz32(15);
    spec.chat.preserve_recent_turns = nz32(1);
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
    spec.chat.max_history_tokens = nz32(50);
    spec.chat.summary_target_tokens = nz32(10);
    spec.chat.preserve_recent_turns = nz32(1);
    let client = build_client(&spec);

    for i in 0..3 {
        let (tx, mut rx) = mpsc::channel(32);
        client
            .generate(format!("turn {i}"), tx)
            .await
            .expect("generate should not fail despite summarize error");
        let events = drain(&mut rx).await;
        assert!(matches!(events.last(), Some(LlmEvent::Done)));
    }

    let captured = script.captured().await;
    assert!(
        captured.iter().any(|r| !r.stream),
        "summarize endpoint should have been attempted"
    );
}

fn tool_call_frames(call_id: &str, name: &str, arg_chunks: &[&str]) -> Vec<String> {
    tool_call_frames_finishing(call_id, name, arg_chunks, "tool_calls")
}

fn tool_call_frames_with_narration(
    narration: &str,
    call_id: &str,
    name: &str,
    arg_chunks: &[&str],
) -> Vec<String> {
    let mut frames = tool_call_frames(call_id, name, arg_chunks);
    frames.insert(
        1,
        format!(
            "data: {{\"choices\":[{{\"delta\":{{\"content\":{}}}}}]}}\n\n",
            serde_json::to_string(narration).unwrap()
        ),
    );
    frames
}

fn tool_call_frames_finishing(
    call_id: &str,
    name: &str,
    arg_chunks: &[&str],
    finish_reason: &str,
) -> Vec<String> {
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
    frames.push(format!(
        "data: {{\"choices\":[{{\"delta\":{{}},\"finish_reason\":{}}}]}}\n\n",
        serde_json::to_string(finish_reason).unwrap()
    ));
    frames.push("data: [DONE]\n\n".to_string());
    frames
}

#[tokio::test]
async fn step_with_text_reply_returns_final_and_sends_no_tools() {
    let script = Script::new();
    script
        .push_stream(StreamResponse::Deltas(vec!["answer".into()]))
        .await;
    let (port, _server) = spawn_fake(script.clone()).await;

    let client = build_client(&chat_spec(port));
    client
        .push_user("what is 2+2?".into(), Vec::new())
        .await
        .unwrap();
    let (tx, mut rx) = mpsc::channel(32);
    let outcome = client.step(Vec::new(), tx).await.unwrap();
    assert!(matches!(outcome, StepOutcome::Final));
    assert_eq!(drain(&mut rx).await, [delta("answer")]);
    let captured = script.captured().await;
    assert_eq!(captured.len(), 1);
    assert!(captured[0].body.get("tools").is_none());
    assert!(captured[0].body.get("tool_choice").is_none());
}

#[tokio::test]
async fn step_runs_tool_calls_reported_with_stop_finish_reason() {
    let script = Script::new();
    script
        .push_stream(StreamResponse::RawFrames(tool_call_frames_finishing(
            "call-7",
            "run",
            &[r#"{"command":"ls /tmp"}"#],
            "stop",
        )))
        .await;
    let (port, _server) = spawn_fake(script.clone()).await;

    let client = build_client(&chat_spec(port));
    client
        .push_user("list /tmp".into(), Vec::new())
        .await
        .unwrap();
    let (tx, _rx) = mpsc::channel(32);
    let outcome = client.step(Vec::new(), tx).await.unwrap();
    let StepOutcome::ToolCalls(calls) = outcome else {
        panic!("expected the tool call to run, got {outcome:?}");
    };
    assert_eq!(
        calls,
        [ToolCall {
            id: "call-7".into(),
            name: "run".into(),
            arguments: json!({"command": "ls /tmp"}),
        }]
    );
}

#[tokio::test]
async fn step_truncated_tool_call_arguments_error_rather_than_vanish() {
    let script = Script::new();
    script
        .push_stream(StreamResponse::RawFrames(tool_call_frames_finishing(
            "call-8",
            "run",
            &[r#"{"command":"ls /t"#],
            "length",
        )))
        .await;
    let (port, _server) = spawn_fake(script.clone()).await;

    let client = build_client(&chat_spec(port));
    client
        .push_user("list /tmp".into(), Vec::new())
        .await
        .unwrap();
    let (tx, _rx) = mpsc::channel(32);
    let err = client
        .step(Vec::new(), tx)
        .await
        .expect_err("truncated arguments must not be swallowed");
    assert!(
        matches!(err, LlmError::ToolCallParse(_)),
        "expected a tool-call parse error, got {err:?}"
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
    client
        .push_user("list /tmp".into(), Vec::new())
        .await
        .unwrap();
    let (tx, mut rx) = mpsc::channel(32);
    let tools = vec![json!({
        "type": "function",
        "function": {
            "name": "run",
            "description": "run a command",
            "parameters": {"type":"object","properties":{"command":{"type":"string"}}},
            "strict": true
        }
    })];
    let outcome = client.step(tools.clone(), tx).await.unwrap();
    let StepOutcome::ToolCalls(calls) = outcome else {
        panic!("expected ToolCalls, got {outcome:?}");
    };
    assert_eq!(
        calls,
        [ToolCall {
            id: "call-42".into(),
            name: "run".into(),
            arguments: json!({"command": "ls /tmp"}),
        }]
    );
    assert!(drain(&mut rx).await.is_empty());

    let captured = script.captured().await;
    assert_eq!(captured[0].body["tool_choice"], "auto");
    assert_eq!(captured[0].body["tools"], json!(tools));
}

#[tokio::test]
async fn agent_round_trip_commits_tool_calls_and_result_to_history() {
    let script = Script::new();
    script
        .push_stream(StreamResponse::RawFrames(tool_call_frames(
            "call-7",
            "run",
            &[r#"{"command":"echo hi"}"#],
        )))
        .await;
    script
        .push_stream(StreamResponse::Deltas(vec!["done".into()]))
        .await;
    let (port, _server) = spawn_fake(script.clone()).await;

    let client = build_client(&chat_spec(port));
    client
        .push_user("please echo hi".into(), Vec::new())
        .await
        .unwrap();

    let tools = vec![json!({
        "type": "function",
        "function": {"name":"run","parameters":{"type":"object"},"strict":true}
    })];
    let (tx1, mut rx1) = mpsc::channel(32);
    let outcome1 = client.step(tools.clone(), tx1).await.unwrap();
    let StepOutcome::ToolCalls(calls) = outcome1 else {
        panic!("expected ToolCalls, got {outcome1:?}");
    };
    drain(&mut rx1).await;

    let result = ToolResultPayload {
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

    let captured = script.captured().await;
    assert_eq!(captured.len(), 2);
    assert_eq!(
        captured[1].body["messages"],
        json!([
            {"role": "system", "content": "test system prompt"},
            {"role": "user", "content": "please echo hi"},
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call-7",
                    "type": "function",
                    "function": {"name": "run", "arguments": r#"{"command":"echo hi"}"#},
                }],
            },
            {"role": "tool", "content": "hi\n[exit:0 | 2ms]", "tool_call_id": "call-7"},
        ]),
        "a text-only result must ride the tool role with the id of its call"
    );
}

#[tokio::test]
async fn narration_before_a_tool_call_stays_in_history() {
    let script = Script::new();
    script
        .push_stream(StreamResponse::RawFrames(tool_call_frames_with_narration(
            "Got it, I'll run a test.",
            "call-9",
            "run",
            &[r#"{"command":"echo hi"}"#],
        )))
        .await;
    script
        .push_stream(StreamResponse::Deltas(vec!["done".into()]))
        .await;
    let (port, _server) = spawn_fake(script.clone()).await;

    let client = build_client(&chat_spec(port));
    client
        .push_user("please echo hi".into(), Vec::new())
        .await
        .unwrap();

    let tools = vec![json!({
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

    client
        .push_tool_results(vec![ToolResultPayload {
            call_id: calls[0].id.clone(),
            name: "run".into(),
            content: "hi\n[exit:0 | 2ms]".into(),
            attachments: Vec::new(),
        }])
        .await
        .unwrap();

    let (tx2, mut rx2) = mpsc::channel(32);
    client.step(tools, tx2).await.unwrap();
    drain(&mut rx2).await;

    let captured = script.captured().await;
    let messages = captured[1].body["messages"].as_array().unwrap();
    let assistant_with_calls = messages
        .iter()
        .find(|m| m["role"] == "assistant" && m.get("tool_calls").is_some())
        .expect("assistant message with tool_calls");
    assert_eq!(
        assistant_with_calls["content"], "Got it, I'll run a test.",
        "pre-tool-call narration must survive into the next request"
    );
}

#[tokio::test]
async fn request_timeout_surfaces_as_error() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        loop {
            if let Ok((sock, _)) = listener.accept().await {
                tokio::time::sleep(Duration::from_secs(30)).await;
                drop(sock);
            }
        }
    });

    let mut spec = chat_spec(port);
    spec.chat.request_timeout_secs = nz64(1);
    let client = build_client(&spec);
    let (tx, _rx) = mpsc::channel(32);
    let err = client.generate("hi".into(), tx).await.unwrap_err();
    assert!(
        matches!(err, LlmError::Chat(ChatClientError::Sse(_))),
        "{err:?}"
    );
}

#[tokio::test]
async fn complete_oneshot_survives_a_response_larger_than_the_channel() {
    let script = Script::new();
    let deltas: Vec<String> = (0..500).map(|i| format!("tok{i} ")).collect();
    script.push_stream(StreamResponse::Deltas(deltas)).await;
    let (port, _server) = spawn_fake(script.clone()).await;

    let client = build_client(&chat_spec(port));
    let text = timeout(
        Duration::from_secs(10),
        client.complete_oneshot("title?".into(), Thinking::Enabled),
    )
    .await
    .expect("complete_oneshot must not deadlock")
    .expect("stream completes");
    let expected: String = (0..500).map(|i| format!("tok{i} ")).collect();
    assert_eq!(text, expected);
}

#[tokio::test]
async fn complete_oneshot_sends_a_lone_prompt_on_the_summary_budget() {
    let script = Script::new();
    script
        .push_stream(StreamResponse::Deltas(vec!["A Short Title".into()]))
        .await;
    script
        .push_stream(StreamResponse::Deltas(vec!["Another Title".into()]))
        .await;
    let (port, _server) = spawn_fake(script.clone()).await;

    let cfg = chat_spec(port);
    let client = build_client(&cfg);
    client
        .push_user("earlier".into(), Vec::new())
        .await
        .unwrap();
    let title = client
        .complete_oneshot("title?".into(), Thinking::Disabled)
        .await
        .unwrap();
    assert_eq!(title, "A Short Title");
    client
        .complete_oneshot("title?".into(), Thinking::Enabled)
        .await
        .unwrap();

    let captured = script.captured().await;
    assert_eq!(
        captured[0].body["messages"],
        json!([{"role": "user", "content": "title?"}])
    );
    assert_eq!(
        captured[0].body["max_tokens"],
        cfg.chat.max_summary_tokens()
    );
    assert_eq!(
        captured[0].body["chat_template_kwargs"],
        json!({"enable_thinking": false}),
        "Thinking::Disabled must ask the chat template to skip reasoning"
    );
    assert!(
        captured[1].body.get("chat_template_kwargs").is_none(),
        "Thinking::Enabled must leave the request untouched"
    );
}

#[tokio::test]
async fn transient_note_is_the_last_wire_message_for_exactly_one_step() {
    let script = Script::new();
    for reply in ["first", "second"] {
        script
            .push_stream(StreamResponse::Deltas(vec![reply.into()]))
            .await;
    }
    let (port, _server) = spawn_fake(script.clone()).await;

    let client = build_client(&chat_spec(port));
    client.push_user("go".into(), Vec::new()).await.unwrap();
    client
        .set_transient_note("answer now".into())
        .await
        .unwrap();
    for _ in 0..2 {
        let (tx, mut rx) = mpsc::channel(32);
        client.step(Vec::new(), tx).await.unwrap();
        drain(&mut rx).await;
    }

    let captured = script.captured().await;
    let first = captured[0].body["messages"].as_array().unwrap();
    let last = first.last().unwrap();
    assert_eq!(last["role"], "user");
    assert_eq!(last["content"], "answer now");

    let second = captured[1].body["messages"].as_array().unwrap();
    assert!(
        second.iter().all(|m| m["content"] != "answer now"),
        "note must not outlive the step it was set for: {second:?}"
    );
}

#[tokio::test]
async fn reasoning_rides_along_with_its_tool_call_until_the_next_user_turn() {
    let script = Script::new();
    let mut frames = tool_call_frames("call-1", "run", &[r#"{"command":"ls"}"#]);
    frames.insert(
        1,
        "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"list it, \"}}]}\n\n".into(),
    );
    frames.insert(
        2,
        "data: {\"choices\":[{\"delta\":{\"content\":\"<think>then read</think>\"}}]}\n\n".into(),
    );
    script.push_stream(StreamResponse::RawFrames(frames)).await;
    for reply in ["done", "sure"] {
        script
            .push_stream(StreamResponse::Deltas(vec![reply.into()]))
            .await;
    }
    let (port, _server) = spawn_fake(script.clone()).await;

    let client = build_client(&chat_spec(port));
    client.push_user("look".into(), Vec::new()).await.unwrap();
    let (tx, mut rx) = mpsc::channel(32);
    let outcome = client.step(Vec::new(), tx).await.unwrap();
    assert!(matches!(outcome, StepOutcome::ToolCalls(_)));
    drain(&mut rx).await;
    client
        .push_tool_results(vec![ToolResultPayload {
            call_id: "call-1".into(),
            name: "run".into(),
            content: "a.txt".into(),
            attachments: Vec::new(),
        }])
        .await
        .unwrap();
    let (tx, mut rx) = mpsc::channel(32);
    client.step(Vec::new(), tx).await.unwrap();
    drain(&mut rx).await;

    client.push_user("thanks".into(), Vec::new()).await.unwrap();
    let (tx, mut rx) = mpsc::channel(32);
    client.step(Vec::new(), tx).await.unwrap();
    drain(&mut rx).await;

    let captured = script.captured().await;
    let calling = |request: &CapturedRequest| {
        request.body["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m.get("tool_calls").is_some())
            .cloned()
            .expect("assistant tool-call message")
    };
    assert_eq!(
        calling(&captured[1])["reasoning_content"],
        "list it, then read",
        "both reasoning channels must reach the next step of the loop"
    );
    assert!(
        calling(&captured[2]).get("reasoning_content").is_none(),
        "a new user turn ends the loop the reasoning belonged to"
    );
}
