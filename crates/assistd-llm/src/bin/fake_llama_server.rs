//! Test-only stand-in for the llama-server process: binds the port and answers `/health`,
//! so the supervisor's lifecycle paths run without a real llama-server.

use std::env;
use std::io::{self, Write};
use std::path::Path;
use std::process::{Command, ExitCode, Stdio};
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::signal::unix::{SignalKind, signal};

const MAX_REQUEST_LINE_BYTES: usize = 16 * 1024;

const HEALTHY: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 15\r\nConnection: close\r\n\r\n{\"status\":\"ok\"}";

const LOADING: &[u8] = b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nContent-Length: 20\r\nConnection: close\r\n\r\n{\"status\":\"loading\"}";

/// Process behaviour, from a `mode` file beside the binary, else `normal`.
#[derive(Debug, Clone, Copy)]
enum Mode {
    /// `normal`: serve 200 on `/health` until SIGTERM.
    Normal,
    /// `never-ready`: serve 503 on `/health` forever.
    NeverReady,
    /// `bind-fail`: exit 1 without binding.
    BindFail,
    /// `slow-term=<secs>`: serve normally, then exit that long after SIGTERM.
    SlowTerm(u64),
    /// `with-orphan`: first spawn a grandchild that ignores SIGTERM and write
    /// its pid to `orphan.pid` beside the binary, then serve normally.
    WithOrphan,
}

struct Args {
    program: String,
    host: String,
    port: u16,
    mode: Mode,
}

fn parse_mode(text: &str) -> Option<Mode> {
    if let Some(secs) = text.strip_prefix("slow-term=") {
        return Some(Mode::SlowTerm(secs.parse().ok()?));
    }
    match text {
        "normal" => Some(Mode::Normal),
        "never-ready" => Some(Mode::NeverReady),
        "bind-fail" => Some(Mode::BindFail),
        "with-orphan" => Some(Mode::WithOrphan),
        _ => None,
    }
}

/// Mode named by a file called `mode` in the directory of `program`
/// (`argv[0]`), so a symlink to this binary carries its own mode.
fn mode_beside(program: &str) -> Option<Mode> {
    let text = std::fs::read_to_string(Path::new(program).with_file_name("mode")).ok()?;
    Some(parse_mode(text.trim()).expect("invalid mode file"))
}

/// Reads `--host <addr>` and `--port <port>`, ignoring the other flags a
/// router-mode spawn passes.
fn parse_args() -> Args {
    let argv: Vec<String> = env::args().collect();
    let program = argv.first().cloned().unwrap_or_default();
    let value_of = |flag: &str| {
        argv.windows(2)
            .find(|pair| pair[0] == flag)
            .map(|pair| pair[1].clone())
    };
    let mode = mode_beside(&program).unwrap_or(Mode::Normal);
    Args {
        host: value_of("--host").unwrap_or_else(|| "127.0.0.1".to_string()),
        port: value_of("--port").map_or(0, |port| port.parse().expect("--port must be u16")),
        mode,
        program,
    }
}

/// Spawn a `sleep` that ignores SIGTERM and record its pid in `orphan.pid`
/// beside `program`.
fn spawn_orphan_beside(program: &str) {
    let orphan = Command::new("sh")
        .args(["-c", "trap '' TERM; exec sleep 300"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn orphan")
        .id();
    std::fs::write(
        Path::new(program).with_file_name("orphan.pid"),
        orphan.to_string(),
    )
    .expect("write orphan pid file");
}

#[tokio::main]
async fn main() -> ExitCode {
    let args = parse_args();

    if matches!(args.mode, Mode::BindFail) {
        let _ = writeln!(io::stderr(), "fake_llama_server: bind-fail mode; exiting");
        return ExitCode::from(1);
    }
    if matches!(args.mode, Mode::WithOrphan) {
        spawn_orphan_beside(&args.program);
    }

    let listener = match TcpListener::bind((args.host.as_str(), args.port)).await {
        Ok(listener) => listener,
        Err(e) => {
            let _ = writeln!(io::stderr(), "fake_llama_server: bind failed: {e}");
            return ExitCode::from(2);
        }
    };
    let _ = writeln!(
        io::stderr(),
        "fake_llama_server: listening on {}:{} mode={:?}",
        args.host,
        args.port,
        args.mode
    );

    let health = if matches!(args.mode, Mode::NeverReady) {
        LOADING
    } else {
        HEALTHY
    };
    let serving = serve_loop(listener, health);
    let Mode::SlowTerm(secs) = args.mode else {
        serving.await;
        return ExitCode::SUCCESS;
    };
    let mut term = signal(SignalKind::terminate()).expect("install SIGTERM handler");
    tokio::spawn(serving);
    term.recv().await;
    let _ = writeln!(
        io::stderr(),
        "fake_llama_server: SIGTERM received; exiting in {secs}s"
    );
    tokio::time::sleep(Duration::from_secs(secs)).await;
    ExitCode::SUCCESS
}

async fn serve_loop(listener: TcpListener, health: &'static [u8]) {
    loop {
        let (client, _) = match listener.accept().await {
            Ok(accepted) => accepted,
            Err(e) => {
                let _ = writeln!(io::stderr(), "fake_llama_server: accept error: {e}");
                continue;
            }
        };
        tokio::spawn(async move {
            let _ = serve_connection(client, health).await;
        });
    }
}

/// Answer `GET /health` with `health` and close any other connection unanswered.
async fn serve_connection(mut client: TcpStream, health: &'static [u8]) -> io::Result<()> {
    let received = read_request_line(&mut client).await?;
    if received.starts_with(b"GET /health ") {
        client.write_all(health).await?;
        client.shutdown().await?;
    }
    Ok(())
}

/// Everything read from `client` up to and past its request line.
async fn read_request_line(client: &mut TcpStream) -> io::Result<Vec<u8>> {
    let mut received = Vec::with_capacity(2048);
    let mut chunk = [0u8; 1024];
    while !received.windows(2).any(|pair| pair == b"\r\n")
        && received.len() < MAX_REQUEST_LINE_BYTES
    {
        let read = client.read(&mut chunk).await?;
        if read == 0 {
            break;
        }
        received.extend_from_slice(&chunk[..read]);
    }
    Ok(received)
}
