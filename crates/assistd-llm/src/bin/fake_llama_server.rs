//! Test-only fake llama-server. Only compiled under the `test-support`
//! feature; used by `crates/assistd-llm/tests/supervisor.rs` to exercise the
//! supervisor's lifecycle paths without needing a real llama-server binary.
//!
//! Modes (via `--mode`):
//!   normal              - bind, serve 200 OK on /health, block until SIGTERM
//!   never-ready         - bind, serve 503 on /health forever
//!   crash-after=<secs>  - bind, serve 200 OK, then `exit(0)` after N seconds
//!   bind-fail           - immediately exit(1) without binding
//!
//! Flags: --host <addr> --port <port> --mode <mode>
//!
//! Uses tokio + a hand-rolled minimal HTTP/1.1 response writer so we don't
//! pull in a whole HTTP server crate just for tests.

use std::env;
use std::process::ExitCode;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

#[derive(Debug, Clone)]
enum Mode {
    Normal,
    NeverReady,
    CrashAfter(u64),
    BindFail,
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
    // Mode is driven by FAKE_LLAMA_MODE so integration tests can set it per
    // test without needing an `extra_args` escape hatch on ServerSpec.
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
            // Ignore llama-server args we don't implement: --jinja, --hf <id>,
            // -ngl <n>, -c <n>, etc.
            "--jinja" => {
                i += 1;
            }
            "--hf" | "-ngl" | "-c" => {
                i += 2;
            }
            _ => {
                i += 1;
            }
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

    if let Mode::CrashAfter(secs) = args.mode {
        let listener = listener;
        let mode = Mode::Normal;
        tokio::spawn(async move {
            serve_loop(listener, mode).await;
        });
        tokio::time::sleep(std::time::Duration::from_secs(secs)).await;
        eprintln!("fake_llama_server: crash-after elapsed; exiting 0");
        return ExitCode::SUCCESS;
    }

    serve_loop(listener, args.mode).await;
    ExitCode::SUCCESS
}

async fn serve_loop(listener: TcpListener, mode: Mode) {
    loop {
        let (mut sock, _) = match listener.accept().await {
            Ok(x) => x,
            Err(e) => {
                eprintln!("fake_llama_server: accept error: {e}");
                continue;
            }
        };
        let mode = mode.clone();
        tokio::spawn(async move {
            // Read at most 2 KiB of the request — we only need the request line.
            let mut buf = [0u8; 2048];
            let _ = match sock.read(&mut buf).await {
                Ok(n) => n,
                Err(_) => return,
            };

            let ready = !matches!(mode, Mode::NeverReady);
            let (status_line, body) = if ready {
                ("HTTP/1.1 200 OK", "{\"status\":\"ok\"}")
            } else {
                (
                    "HTTP/1.1 503 Service Unavailable",
                    "{\"status\":\"loading\"}",
                )
            };
            let resp = format!(
                "{status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = sock.write_all(resp.as_bytes()).await;
            let _ = sock.shutdown().await;
        });
    }
}
