//! Client for the voice-output CLI subcommands: `voice-toggle`,
//! `voice-skip`, `voice-state`. Each sends one IPC request over the
//! daemon's Unix socket, prints the response, and exits on `Event::Done`
//! or `Event::Error`.

use anyhow::{Context, Result};
use assistd_ipc::{Event, Request, socket_path};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use uuid::Uuid;

/// Which voice-output command the CLI is dispatching.
#[derive(Debug, Clone, Copy)]
pub enum VoiceCtlAction {
    Toggle,
    Skip,
    State,
}

impl VoiceCtlAction {
    fn to_request(self, id: String) -> Request {
        match self {
            VoiceCtlAction::Toggle => Request::VoiceToggle { id },
            VoiceCtlAction::Skip => Request::VoiceSkip { id },
            VoiceCtlAction::State => Request::GetVoiceState { id },
        }
    }
}

pub async fn run(action: VoiceCtlAction) -> Result<()> {
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
            Event::VoiceOutputState { enabled, .. } => {
                println!("voice-output: {}", if enabled { "on" } else { "off" });
            }
            Event::Done { .. } => return Ok(()),
            Event::Error { message, .. } => {
                eprintln!("daemon error: {message}");
                std::process::exit(1);
            }
            _ => {}
        }
    }
}
