//! Test-only fake llama-server. Only compiled under the `test-support`
//! feature; used by `crates/assistd-llm/tests/supervisor.rs` and
//! `crates/assistd-core/tests/presence.rs` to exercise lifecycle and
//! presence paths without needing a real llama-server binary.
//!
//! Modes (via `--mode` or `FAKE_LLAMA_MODE`):
//!   normal              - bind, serve 200 on /health, block until SIGTERM
//!   never-ready         - bind, serve 503 on /health forever
//!   crash-after=<secs>  - bind, serve 200 OK, then `exit(0)` after N seconds
//!   bind-fail           - immediately exit(1) without binding
//!   load-failure        - /models/load returns 500 (for wake-failure tests)
//!
//! Endpoints:
//!   GET  /health            - 200 {"status":"ok"} (or 503 in never-ready)
//!   POST /models/load       - 200, records hit, marks model loaded
//!   POST /models/unload     - 200, records hit, marks model unloaded
//!   GET  /models            - JSON list with current load state
//!   GET  /debug/counters    - test introspection: load/unload counts, PID
//!
//! Flags: --host <addr> --port <port> --mode <mode>

use std::env;
use std::process::ExitCode;
use std::sync::Arc;

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
            // Ignore llama-server args we don't implement. `--hf-repo` is no
            // longer passed in router mode, but accept it anyway for forward
            // compat in case an old test path invokes us that way.
            "--jinja" => i += 1,
            "--hf-repo" | "-ngl" | "-c" => i += 2,
            _ => i += 1,
        }
    }
    Args { host, port, mode }
}

#[derive(Default)]
struct ServerState {
    loaded_model: Option<String>,
    load_count: u32,
    unload_count: u32,
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

    let (status_line, content_type, body) = match (method.as_str(), path.as_str()) {
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

    let resp = format!(
        "{status_line}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = sock.write_all(resp.as_bytes()).await;
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
    let body = format!(
        "{{\"pid\":{},\"load_count\":{},\"unload_count\":{},\"loaded_model\":{}}}",
        pid,
        s.load_count,
        s.unload_count,
        match &s.loaded_model {
            Some(n) => format!("\"{}\"", escape_json(n)),
            None => "null".to_string(),
        }
    );
    ("HTTP/1.1 200 OK", "application/json", body)
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
