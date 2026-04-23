//! Client for the continuous-listen CLI subcommands: `listen-start`,
//! `listen-stop`, `listen-toggle`, `listen-state`. Each sends one IPC
//! request over the daemon's Unix socket, prints the response, and
//! exits on `Event::Done` or `Event::Error`.

use anyhow::{Context, Result};
use assistd_ipc::{Event, Request, socket_path};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use uuid::Uuid;

/// Which continuous-listen command the CLI is dispatching.
#[derive(Debug, Clone, Copy)]
pub enum ListenAction {
    Start,
    Stop,
    Toggle,
    State,
}

impl ListenAction {
    fn to_request(self, id: String) -> Request {
        match self {
            ListenAction::Start => Request::ListenStart { id },
            ListenAction::Stop => Request::ListenStop { id },
            ListenAction::Toggle => Request::ListenToggle { id },
            ListenAction::State => Request::GetListenState { id },
        }
    }
}

pub async fn run(action: ListenAction) -> Result<()> {
    let path = socket_path();
    let stream = UnixStream::connect(&path).await.with_context(|| {
        format!(
            "assistd daemon is not running (could not connect to {})",
            path.display()
        )
    })?;

    let (read_half, mut write_half) = stream.into_split();
    let req = action.to_request(Uuid::new_v4().to_string());
    let mut body = serde_json::to_string(&req)?;
    body.push('\n');
    write_half.write_all(body.as_bytes()).await?;
    write_half.shutdown().await?;

    let mut reader = BufReader::new(read_half);
    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line).await?;
        if n == 0 {
            anyhow::bail!("daemon closed the connection without sending a terminal event");
        }
        let event: Event = serde_json::from_str(line.trim())
            .with_context(|| format!("invalid JSON from daemon: {}", line.trim()))?;

        match event {
            Event::ListenState { active, .. } => {
                println!("listen: {}", if active { "on" } else { "off" });
            }
            Event::Done { .. } => return Ok(()),
            Event::Error { message, .. } => {
                eprintln!("daemon error: {message}");
                std::process::exit(1);
            }
            // Other event types are not expected for listen commands —
            // ignore defensively so a future event addition doesn't
            // break the client.
            _ => {}
        }
    }
}
