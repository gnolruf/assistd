//! Test-only fake llama-server. Only compiled under the `test-support`
//! feature; used by `crates/assistd-llm/tests/supervisor.rs`,
//! `crates/assistd-llm/tests/presence.rs`, and the cross-crate stress /
//! latency tests to exercise lifecycle, presence, and chat-completion
//! paths without needing a real llama-server binary.
//!
//! Modes (via `--mode` or `FAKE_LLAMA_MODE`):
//!   normal              - bind, serve 200 on /health, block until SIGTERM
//!   never-ready         - bind, serve 503 on /health forever
//!   crash-after=<secs>  - bind, serve 200 OK, then `exit(0)` after N seconds
//!   bind-fail           - immediately exit(1) without binding
//!   load-failure        - /models/load returns 500 (for wake-failure tests)
//!
//! Endpoints:
//!   GET  /health                - 200 {"status":"ok"} (or 503 in never-ready)
//!   POST /models/load           - 200, records hit, marks model loaded
//!   POST /models/unload         - 200, records hit, marks model unloaded
//!   GET  /models                - JSON list with current load state
//!   POST /v1/chat/completions   - SSE chunked stream OR JSON summary
//!   POST /test/script           - push a scripted chat reply (X-Test-Control: 1)
//!   POST /test/reset            - clear queue + counters (X-Test-Control: 1)
//!   GET  /debug/counters        - test introspection (PID, hits, last prompt)
//!
//! Default chat reply (when no script pushed): one delta `"hello"` + [DONE].
//!
//! Flags: --host <addr> --port <port> --mode <mode>

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stdout,
    clippy::print_stderr
)]

use std::collections::VecDeque;
use std::env;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;

#[derive(Debug, Clone)]
enum Mode {
    Normal,
    NeverReady,
    CrashAfter(u64),
    BindFail,
    LoadFailure,
}

fn parse_mode(s: &str) -> Option<Mode> {
    if let Some(rest) = s.strip_prefix("crash-after=") {
        let secs: u64 = rest.parse().ok()?;
        return Some(Mode::CrashAfter(secs));
    }
    match s {
        "normal" => Some(Mode::Normal),
        "never-ready" => Some(Mode::NeverReady),
        "bind-fail" => Some(Mode::BindFail),
        "load-failure" => Some(Mode::LoadFailure),
        _ => None,
    }
}

struct Args {
    host: String,
    port: u16,
    mode: Mode,
}

fn parse_args() -> Args {
    let mut host = "127.0.0.1".to_string();
    let mut port: u16 = 0;
    let mut mode = env::var("FAKE_LLAMA_MODE")
        .ok()
        .and_then(|s| parse_mode(&s))
        .unwrap_or(Mode::Normal);

    let argv: Vec<String> = env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--host" => {
                host = argv[i + 1].clone();
                i += 2;
            }
            "--port" => {
                port = argv[i + 1].parse().expect("--port must be u16");
                i += 2;
            }
            "--mode" => {
                mode = parse_mode(&argv[i + 1]).expect("invalid --mode");
                i += 2;
            }
            // Ignore llama-server args we don't implement but real
            // router-mode spawns pass through.
            "--jinja" => i += 1,
            "-ngl" | "-c" => i += 2,
            _ => i += 1,
        }
    }
    Args { host, port, mode }
}

#[derive(Clone, Default)]
struct ChatScript {
    deltas: Vec<String>,
    delay_ms_between: u64,
}

#[derive(Default)]
struct ServerState {
    loaded_model: Option<String>,
    load_count: u32,
    unload_count: u32,
    chat_completions_count: u32,
    last_prompt: Option<String>,
    chat_scripts: VecDeque<ChatScript>,
}

#[tokio::main]
async fn main() -> ExitCode {
    let args = parse_args();

    if matches!(args.mode, Mode::BindFail) {
        eprintln!("fake_llama_server: bind-fail mode; exiting");
        return ExitCode::from(1);
    }

    let listener = match TcpListener::bind((args.host.as_str(), args.port)).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("fake_llama_server: bind failed: {e}");
            return ExitCode::from(2);
        }
    };
    eprintln!(
        "fake_llama_server: listening on {}:{} mode={:?}",
        args.host, args.port, args.mode
    );

    let state = Arc::new(Mutex::new(ServerState::default()));

    if let Mode::CrashAfter(secs) = args.mode {
        let state = state.clone();
        tokio::spawn(async move {
            serve_loop(listener, Mode::Normal, state).await;
        });
        tokio::time::sleep(std::time::Duration::from_secs(secs)).await;
        eprintln!("fake_llama_server: crash-after elapsed; exiting 0");
        return ExitCode::SUCCESS;
    }

    serve_loop(listener, args.mode, state).await;
    ExitCode::SUCCESS
}

async fn serve_loop(listener: TcpListener, mode: Mode, state: Arc<Mutex<ServerState>>) {
    loop {
        let (sock, _) = match listener.accept().await {
            Ok(x) => x,
            Err(e) => {
                eprintln!("fake_llama_server: accept error: {e}");
                continue;
            }
        };
        let mode = mode.clone();
        let state = state.clone();
        tokio::spawn(async move {
            let _ = handle_request(sock, mode, state).await;
        });
    }
}

async fn handle_request(
    mut sock: TcpStream,
    mode: Mode,
    state: Arc<Mutex<ServerState>>,
) -> std::io::Result<()> {
    let (head, body) = read_request(&mut sock).await?;
    let (method, path) = parse_request_line(&head);

    match (method.as_str(), path.as_str()) {
        // Streaming chat: written directly to the socket so SSE chunks
        // flow as they're produced instead of being buffered into a
        // single Content-Length response.
        ("POST", "/v1/chat/completions") => {
            handle_chat_completion(&mut sock, &state, &body).await?;
            return Ok(());
        }
        ("POST", "/test/script") | ("POST", "/test/reset") => {
            let resp = if has_test_control_header(&head) {
                if path == "/test/script" {
                    handle_test_script(&state, &body).await
                } else {
                    handle_test_reset(&state).await
                }
            } else {
                (
                    "HTTP/1.1 403 Forbidden",
                    "application/json",
                    "{\"error\":\"missing X-Test-Control: 1 header\"}".to_string(),
                )
            };
            write_one_shot(&mut sock, resp).await?;
            return Ok(());
        }
        _ => {}
    }

    let resp = match (method.as_str(), path.as_str()) {
        ("GET", "/health") => health_response(&mode),
        ("POST", "/models/load") => load_response(&mode, &state, &body).await,
        ("POST", "/models/unload") => unload_response(&state, &body).await,
        ("GET", "/models") => list_models_response(&state).await,
        ("GET", "/debug/counters") => counters_response(&state).await,
        _ => (
            "HTTP/1.1 404 Not Found",
            "text/plain",
            format!("unknown route: {method} {path}"),
        ),
    };
    write_one_shot(&mut sock, resp).await?;
    Ok(())
}

async fn write_one_shot(
    sock: &mut TcpStream,
    (status_line, content_type, body): (&'static str, &'static str, String),
) -> std::io::Result<()> {
    let resp = format!(
        "{status_line}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    sock.write_all(resp.as_bytes()).await?;
    let _ = sock.shutdown().await;
    Ok(())
}

/// Reads a full HTTP request: headers until `\r\n\r\n`, then `Content-Length`
/// bytes of body if indicated. Returns (head_string, body_string).
async fn read_request(sock: &mut TcpStream) -> std::io::Result<(String, String)> {
    let mut buf = Vec::with_capacity(2048);
    let mut tmp = [0u8; 1024];
    let header_end;
    loop {
        let n = sock.read(&mut tmp).await?;
        if n == 0 {
            return Ok((String::from_utf8_lossy(&buf).into_owned(), String::new()));
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(idx) = find_header_end(&buf) {
            header_end = idx;
            break;
        }
        if buf.len() > 16 * 1024 {
            // Prevent unbounded growth in case a client keeps sending.
            header_end = buf.len();
            break;
        }
    }
    let head = String::from_utf8_lossy(&buf[..header_end]).into_owned();
    let content_length = parse_content_length(&head);
    let already_in_buf = buf.len().saturating_sub(header_end + 4);
    let mut body_bytes: Vec<u8> = buf[(header_end + 4).min(buf.len())..].to_vec();

    if content_length > already_in_buf {
        let remaining = content_length - already_in_buf;
        let mut body_tail = vec![0u8; remaining];
        sock.read_exact(&mut body_tail).await?;
        body_bytes.extend_from_slice(&body_tail);
    } else {
        body_bytes.truncate(content_length);
    }

    Ok((head, String::from_utf8_lossy(&body_bytes).into_owned()))
}

fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

fn parse_content_length(head: &str) -> usize {
    for line in head.lines() {
        let Some(colon) = line.find(':') else {
            continue;
        };
        let name = line[..colon].trim();
        if name.eq_ignore_ascii_case("content-length")
            && let Ok(n) = line[colon + 1..].trim().parse::<usize>()
        {
            return n;
        }
    }
    0
}

fn parse_request_line(head: &str) -> (String, String) {
    let first = head.lines().next().unwrap_or("");
    let mut parts = first.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();
    (method, path)
}

/// Header-gate for test-control endpoints. Production-shaped clients can't
/// hit `/test/*` without explicitly setting this header, which prevents a
/// misbehaving-client test from accidentally reconfiguring scripted state.
fn has_test_control_header(head: &str) -> bool {
    for line in head.lines() {
        if let Some(colon) = line.find(':') {
            let name = line[..colon].trim();
            let value = line[colon + 1..].trim();
            if name.eq_ignore_ascii_case("x-test-control") && value == "1" {
                return true;
            }
        }
    }
    false
}

fn health_response(mode: &Mode) -> (&'static str, &'static str, String) {
    if matches!(mode, Mode::NeverReady) {
        (
            "HTTP/1.1 503 Service Unavailable",
            "application/json",
            "{\"status\":\"loading\"}".to_string(),
        )
    } else {
        (
            "HTTP/1.1 200 OK",
            "application/json",
            "{\"status\":\"ok\"}".to_string(),
        )
    }
}

async fn load_response(
    mode: &Mode,
    state: &Arc<Mutex<ServerState>>,
    body: &str,
) -> (&'static str, &'static str, String) {
    if matches!(mode, Mode::LoadFailure) {
        return (
            "HTTP/1.1 500 Internal Server Error",
            "application/json",
            "{\"error\":\"load failed\"}".to_string(),
        );
    }
    let model = extract_model_field(body);
    let mut s = state.lock().await;
    s.load_count += 1;
    s.loaded_model = Some(model.clone());
    (
        "HTTP/1.1 200 OK",
        "application/json",
        format!(
            "{{\"model\":\"{}\",\"status\":\"loaded\"}}",
            escape_json(&model)
        ),
    )
}

async fn unload_response(
    state: &Arc<Mutex<ServerState>>,
    body: &str,
) -> (&'static str, &'static str, String) {
    let model = extract_model_field(body);
    let mut s = state.lock().await;
    s.unload_count += 1;
    s.loaded_model = None;
    (
        "HTTP/1.1 200 OK",
        "application/json",
        format!(
            "{{\"model\":\"{}\",\"status\":\"unloaded\"}}",
            escape_json(&model)
        ),
    )
}

async fn list_models_response(
    state: &Arc<Mutex<ServerState>>,
) -> (&'static str, &'static str, String) {
    let s = state.lock().await;
    let body = match &s.loaded_model {
        Some(name) => format!(
            "{{\"data\":[{{\"id\":\"{}\",\"status\":\"loaded\"}}]}}",
            escape_json(name)
        ),
        None => "{\"data\":[]}".to_string(),
    };
    ("HTTP/1.1 200 OK", "application/json", body)
}

async fn counters_response(
    state: &Arc<Mutex<ServerState>>,
) -> (&'static str, &'static str, String) {
    let s = state.lock().await;
    let pid = std::process::id();
    let loaded = match &s.loaded_model {
        Some(n) => format!("\"{}\"", escape_json(n)),
        None => "null".to_string(),
    };
    let last_prompt = match &s.last_prompt {
        Some(p) => format!("\"{}\"", escape_json(p)),
        None => "null".to_string(),
    };
    let body = format!(
        "{{\"pid\":{},\"load_count\":{},\"unload_count\":{},\"loaded_model\":{},\
         \"chat_completions_count\":{},\"last_prompt\":{}}}",
        pid, s.load_count, s.unload_count, loaded, s.chat_completions_count, last_prompt
    );
    ("HTTP/1.1 200 OK", "application/json", body)
}

async fn handle_test_script(
    state: &Arc<Mutex<ServerState>>,
    body: &str,
) -> (&'static str, &'static str, String) {
    let parsed: Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(e) => {
            return (
                "HTTP/1.1 400 Bad Request",
                "application/json",
                format!(
                    "{{\"error\":\"invalid json: {}\"}}",
                    escape_json(&e.to_string())
                ),
            );
        }
    };
    let deltas = parsed
        .get("deltas")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let delay_ms_between = parsed
        .get("delay_ms_between")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let mut s = state.lock().await;
    s.chat_scripts.push_back(ChatScript {
        deltas,
        delay_ms_between,
    });
    (
        "HTTP/1.1 200 OK",
        "application/json",
        "{\"status\":\"queued\"}".to_string(),
    )
}

async fn handle_test_reset(
    state: &Arc<Mutex<ServerState>>,
) -> (&'static str, &'static str, String) {
    let mut s = state.lock().await;
    s.chat_scripts.clear();
    s.chat_completions_count = 0;
    s.last_prompt = None;
    (
        "HTTP/1.1 200 OK",
        "application/json",
        "{\"status\":\"reset\"}".to_string(),
    )
}

/// Handle `POST /v1/chat/completions`. Streaming requests get a chunked SSE
/// body terminated by `data: [DONE]`; non-streaming requests get a single
/// JSON object with `choices[0].message.content`. Pops one entry off the
/// scripted-reply queue per request; falls back to a single `"hello"` delta
/// when the queue is empty so cross-crate tests can run without scripting.
async fn handle_chat_completion(
    sock: &mut TcpStream,
    state: &Arc<Mutex<ServerState>>,
    body: &str,
) -> std::io::Result<()> {
    let parsed: Value = serde_json::from_str(body).unwrap_or(Value::Null);
    let stream = parsed
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let last_user = parsed
        .get("messages")
        .and_then(|m| m.as_array())
        .and_then(|arr| {
            arr.iter().rev().find_map(|msg| {
                if msg.get("role").and_then(|r| r.as_str()) == Some("user") {
                    msg.get("content")
                        .and_then(|c| c.as_str())
                        .map(|s| s.to_string())
                } else {
                    None
                }
            })
        });

    let script = {
        let mut s = state.lock().await;
        s.chat_completions_count += 1;
        if let Some(p) = last_user {
            s.last_prompt = Some(p);
        }
        s.chat_scripts.pop_front().unwrap_or(ChatScript {
            deltas: vec!["hello".to_string()],
            delay_ms_between: 0,
        })
    };

    if stream {
        write_sse_stream(sock, &script).await?;
    } else {
        let combined: String = script.deltas.join("");
        let body = format!(
            "{{\"choices\":[{{\"message\":{{\"role\":\"assistant\",\"content\":{}}}}}]}}",
            serde_json::to_string(&combined).unwrap_or_else(|_| "\"\"".to_string())
        );
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        sock.write_all(resp.as_bytes()).await?;
        let _ = sock.shutdown().await;
    }
    Ok(())
}

async fn write_sse_stream(sock: &mut TcpStream, script: &ChatScript) -> std::io::Result<()> {
    let headers = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n";
    sock.write_all(headers).await?;

    for (i, delta) in script.deltas.iter().enumerate() {
        if i > 0 && script.delay_ms_between > 0 {
            tokio::time::sleep(Duration::from_millis(script.delay_ms_between)).await;
        }
        let frame = format!(
            "data: {{\"choices\":[{{\"delta\":{{\"content\":{}}}}}]}}\n\n",
            serde_json::to_string(&delta).unwrap_or_else(|_| "\"\"".to_string())
        );
        write_chunk(sock, frame.as_bytes()).await?;
    }
    write_chunk(sock, b"data: [DONE]\n\n").await?;
    write_final_chunk(sock).await?;
    let _ = sock.shutdown().await;
    Ok(())
}

async fn write_chunk(sock: &mut TcpStream, payload: &[u8]) -> std::io::Result<()> {
    let header = format!("{:x}\r\n", payload.len());
    sock.write_all(header.as_bytes()).await?;
    sock.write_all(payload).await?;
    sock.write_all(b"\r\n").await?;
    Ok(())
}

async fn write_final_chunk(sock: &mut TcpStream) -> std::io::Result<()> {
    sock.write_all(b"0\r\n\r\n").await
}

/// Pulls the `"model": "..."` field out of a JSON body without dragging in
/// a full JSON dep. Good enough for the single-field requests we receive.
fn extract_model_field(body: &str) -> String {
    let key = "\"model\"";
    let Some(start) = body.find(key) else {
        return String::new();
    };
    let after_key = &body[start + key.len()..];
    let Some(colon) = after_key.find(':') else {
        return String::new();
    };
    let after_colon = &after_key[colon + 1..];
    let Some(quote_start) = after_colon.find('"') else {
        return String::new();
    };
    let value_start = &after_colon[quote_start + 1..];
    let Some(quote_end) = value_start.find('"') else {
        return String::new();
    };
    value_start[..quote_end].to_string()
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}
