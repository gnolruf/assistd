//! Client for the manual presence-control subcommands: `sleep`, `wake`,
//! `drowse`, `cycle`. Each sends exactly one IPC request over the daemon's
//! Unix socket, prints the resulting presence state, and exits on
//! `Event::Done` or `Event::Error`.

use anyhow::{Context, Result};
use assistd_ipc::{Event, PresenceState, Request, socket_path};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use uuid::Uuid;

/// Which transition the CLI is asking the daemon to run.
#[derive(Debug, Clone, Copy)]
pub enum PresenceAction {
    Sleep,
    Drowse,
    Wake,
    Cycle,
}

impl PresenceAction {
    fn to_request(self, id: String) -> Request {
        match self {
            PresenceAction::Sleep => Request::SetPresence {
                id,
                target: PresenceState::Sleeping,
            },
            PresenceAction::Drowse => Request::SetPresence {
                id,
                target: PresenceState::Drowsy,
            },
            PresenceAction::Wake => Request::SetPresence {
                id,
                target: PresenceState::Active,
            },
            PresenceAction::Cycle => Request::Cycle { id },
        }
    }
}

pub async fn run(action: PresenceAction) -> Result<()> {
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
            Event::Presence { state, .. } => {
                println!("presence: {}", presence_label(state));
            }
            Event::Done { .. } => return Ok(()),
            Event::Error { message, .. } => {
                eprintln!("daemon error: {message}");
                std::process::exit(1);
            }
            // Any other event types (Delta, ToolCall, …) shouldn't appear on a
            // presence request — ignore so a future protocol addition doesn't
            // break this CLI.
            _ => {}
        }
    }
}

fn presence_label(state: PresenceState) -> &'static str {
    match state {
        PresenceState::Active => "active",
        PresenceState::Drowsy => "drowsy",
        PresenceState::Sleeping => "sleeping",
    }
}
