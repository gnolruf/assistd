//! Test-only fake llama-server, compiled under the `test-support` feature,
//! for exercising lifecycle, presence, and chat-completion paths without a
//! real llama-server binary.

use std::collections::VecDeque;
use std::env;
use std::io;
use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::signal::unix::{SignalKind, signal};
use tokio::sync::Mutex;

const MAX_HEADER_BYTES: usize = 16 * 1024;

/// Server behaviour, from `--mode`, else a `mode` file beside the binary,
/// else `normal`.
#[derive(Debug, Clone)]
enum Mode {
    /// `normal`: serve 200 on `/health` until SIGTERM.
    Normal,
    /// `never-ready`: serve 503 on `/health` forever.
    NeverReady,
    /// `crash-after=<secs>`: serve normally, then exit 0 after that long.
    CrashAfter(u64),
    /// `bind-fail`: exit 1 without binding.
    BindFail,
    /// `load-failure`: fail `POST /models/load` with 500.
    LoadFailure,
    /// `slow-term=<secs>`: serve normally, then exit that long after SIGTERM.
    SlowTerm(u64),
}

struct Args {
    host: String,
    port: u16,
    mode: Mode,
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

fn parse_mode(s: &str) -> Option<Mode> {
    if let Some(rest) = s.strip_prefix("crash-after=") {
        let secs: u64 = rest.parse().ok()?;
        return Some(Mode::CrashAfter(secs));
    }
    if let Some(rest) = s.strip_prefix("slow-term=") {
        let secs: u64 = rest.parse().ok()?;
        return Some(Mode::SlowTerm(secs));
    }
    match s {
        "normal" => Some(Mode::Normal),
        "never-ready" => Some(Mode::NeverReady),
        "bind-fail" => Some(Mode::BindFail),
        "load-failure" => Some(Mode::LoadFailure),
        _ => None,
    }
}

/// Mode named by a file called `mode` in the directory of `program`
/// (`argv[0]`), so a symlink to this binary carries its own mode.
fn mode_beside(program: &str) -> Option<Mode> {
    let text = std::fs::read_to_string(Path::new(program).with_file_name("mode")).ok()?;
    Some(parse_mode(text.trim()).expect("invalid mode file"))
}

/// Parses `--host <addr> --port <port> --mode <mode>`, skipping the other
/// flags a router-mode spawn passes.
fn parse_args() -> Args {
    let mut host = "127.0.0.1".to_string();
    let mut port: u16 = 0;
    let argv: Vec<String> = env::args().collect();
    let mut mode = argv
        .first()
        .and_then(|program| mode_beside(program))
        .unwrap_or(Mode::Normal);

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
            "--jinja" => i += 1,
            "-ngl" | "-c" => i += 2,
            _ => i += 1,
        }
    }
    Args { host, port, mode }
}

#[tokio::main]
async fn main() -> ExitCode {
    let args = parse_args();

    if matches!(args.mode, Mode::BindFail) {
        eprintln!("fake_llama_server: bind-fail mode; exiting");
        return ExitCode::from(1);
    }

    let listener = match TcpListener::bind((args.host.as_str(), args.port)).await {
        Ok(listener) => listener,
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
        tokio::time::sleep(Duration::from_secs(secs)).await;
        eprintln!("fake_llama_server: crash-after elapsed; exiting 0");
        return ExitCode::SUCCESS;
    }

    if let Mode::SlowTerm(secs) = args.mode {
        let mut term = signal(SignalKind::terminate()).expect("install SIGTERM handler");
        let state = state.clone();
        tokio::spawn(async move {
            serve_loop(listener, Mode::Normal, state).await;
        });
        term.recv().await;
        eprintln!("fake_llama_server: SIGTERM received; exiting in {secs}s");
        tokio::time::sleep(Duration::from_secs(secs)).await;
        return ExitCode::SUCCESS;
    }

    serve_loop(listener, args.mode, state).await;
    ExitCode::SUCCESS
}

async fn serve_loop(listener: TcpListener, mode: Mode, state: Arc<Mutex<ServerState>>) {
    loop {
        let (sock, _) = match listener.accept().await {
            Ok(accepted) => accepted,
            Err(e) => {
                eprintln!("fake_llama_server: accept error: {e}");
                continue;
            }
        };
        let mode = mode.clone();
        let state = state.clone();
        tokio::spawn(async move {
            let _ = serve_connection(sock, mode, state).await;
        });
    }
}

/// Serves one request:
/// - `GET /health`: 200 `{"status":"ok"}`, or 503 in never-ready mode.
/// - `POST /models/load`, `POST /models/unload`: 200; counts the hit and
///   updates the load state.
/// - `GET /models`: the current load state.
/// - `POST /v1/chat/completions`: an SSE stream or a JSON summary.
/// - `POST /test/script`, `POST /test/reset`: queue a scripted chat reply,
///   or clear the queue and counters. Both require `X-Test-Control: 1`.
/// - `GET /debug/counters`: PID, hit counts and last prompt.
async fn serve_connection(
    mut sock: TcpStream,
    mode: Mode,
    state: Arc<Mutex<ServerState>>,
) -> io::Result<()> {
    let (head, body) = read_request(&mut sock).await?;
    let (method, path) = parse_request_line(&head);

    match (method.as_str(), path.as_str()) {
        ("POST", "/v1/chat/completions") => {
            serve_chat_completion(&mut sock, &state, &body).await?;
            return Ok(());
        }
        ("POST", "/test/script") | ("POST", "/test/reset") => {
            let resp = if has_test_control_header(&head) {
                if path == "/test/script" {
                    queue_script_response(&state, &body).await
                } else {
                    reset_response(&state).await
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
) -> io::Result<()> {
    let resp = format!(
        "{status_line}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    sock.write_all(resp.as_bytes()).await?;
    let _ = sock.shutdown().await;
    Ok(())
}

/// Reads a full HTTP request: headers until `\r\n\r\n`, then `Content-Length`
/// bytes of body if indicated.
async fn read_request(sock: &mut TcpStream) -> io::Result<(String, String)> {
    let mut buf = Vec::with_capacity(2048);
    let mut read_buf = [0u8; 1024];
    let header_end;
    loop {
        let n = sock.read(&mut read_buf).await?;
        if n == 0 {
            return Ok((String::from_utf8_lossy(&buf).into_owned(), String::new()));
        }
        buf.extend_from_slice(&read_buf[..n]);
        if let Some(idx) = find_header_end(&buf) {
            header_end = idx;
            break;
        }
        if buf.len() > MAX_HEADER_BYTES {
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

/// Whether the request carries `X-Test-Control: 1`, which gates `/test/*`.
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
    let mut server = state.lock().await;
    server.load_count += 1;
    server.loaded_model = Some(model.clone());
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
    let mut server = state.lock().await;
    server.unload_count += 1;
    server.loaded_model = None;
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
    let server = state.lock().await;
    let body = match &server.loaded_model {
        Some(name) => format!(
            "{{\"data\":[{{\"id\":\"{}\",\"status\":{{\"value\":\"loaded\",\"args\":[]}}}}]}}",
            escape_json(name)
        ),
        None => "{\"data\":[]}".to_string(),
    };
    ("HTTP/1.1 200 OK", "application/json", body)
}

async fn counters_response(
    state: &Arc<Mutex<ServerState>>,
) -> (&'static str, &'static str, String) {
    let server = state.lock().await;
    let pid = std::process::id();
    let loaded = match &server.loaded_model {
        Some(name) => format!("\"{}\"", escape_json(name)),
        None => "null".to_string(),
    };
    let last_prompt = match &server.last_prompt {
        Some(prompt) => format!("\"{}\"", escape_json(prompt)),
        None => "null".to_string(),
    };
    let body = format!(
        "{{\"pid\":{},\"load_count\":{},\"unload_count\":{},\"loaded_model\":{},\
         \"chat_completions_count\":{},\"last_prompt\":{}}}",
        pid,
        server.load_count,
        server.unload_count,
        loaded,
        server.chat_completions_count,
        last_prompt
    );
    ("HTTP/1.1 200 OK", "application/json", body)
}

async fn queue_script_response(
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
    let mut server = state.lock().await;
    server.chat_scripts.push_back(ChatScript {
        deltas,
        delay_ms_between,
    });
    (
        "HTTP/1.1 200 OK",
        "application/json",
        "{\"status\":\"queued\"}".to_string(),
    )
}

async fn reset_response(state: &Arc<Mutex<ServerState>>) -> (&'static str, &'static str, String) {
    let mut server = state.lock().await;
    server.chat_scripts.clear();
    server.chat_completions_count = 0;
    server.last_prompt = None;
    (
        "HTTP/1.1 200 OK",
        "application/json",
        "{\"status\":\"reset\"}".to_string(),
    )
}

/// Answer `POST /v1/chat/completions` with the next scripted reply (default
/// `"hello"`): a chunked SSE stream written as produced, or one JSON body.
async fn serve_chat_completion(
    sock: &mut TcpStream,
    state: &Arc<Mutex<ServerState>>,
    body: &str,
) -> io::Result<()> {
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
        let mut server = state.lock().await;
        server.chat_completions_count += 1;
        if let Some(prompt) = last_user {
            server.last_prompt = Some(prompt);
        }
        server.chat_scripts.pop_front().unwrap_or(ChatScript {
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

async fn write_sse_stream(sock: &mut TcpStream, script: &ChatScript) -> io::Result<()> {
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

/// The `"model": "..."` string from a flat JSON body, found by text search;
/// empty when absent.
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
