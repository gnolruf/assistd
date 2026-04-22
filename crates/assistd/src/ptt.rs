//! Client for the push-to-talk CLI subcommands: `ptt-start`,
//! `ptt-stop`. Each sends exactly one IPC request over the daemon's
//! Unix socket, prints event lines as they arrive (voice state, the
//! final transcription, and — for `ptt-stop` — the streaming LLM
//! response), and exits on `Event::Done` or `Event::Error`.
//!
//! Typical i3 binding:
//!
//! ```text
//! bindsym           $mod+space exec --no-startup-id assistd ptt-start
//! bindsym --release $mod+space exec --no-startup-id assistd ptt-stop
//! ```

use std::io::Write;

use anyhow::{Context, Result};
use assistd_ipc::{Event, Request, VoiceCaptureState, socket_path};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use uuid::Uuid;

/// Which phase of the PTT cycle the CLI is asking the daemon to run.
#[derive(Debug, Clone, Copy)]
pub enum PttAction {
    Start,
    Stop,
}

impl PttAction {
    fn to_request(self, id: String) -> Request {
        match self {
            PttAction::Start => Request::PttStart { id },
            PttAction::Stop => Request::PttStop { id },
        }
    }
}

pub async fn run(action: PttAction) -> Result<()> {
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
    let mut stdout = std::io::stdout().lock();
    let mut wrote_delta = false;
    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line).await?;
        if n == 0 {
            anyhow::bail!("daemon closed the connection without sending a terminal event");
        }
        let event: Event = serde_json::from_str(line.trim())
            .with_context(|| format!("invalid JSON from daemon: {}", line.trim()))?;

        match event {
            Event::VoiceState { state, .. } => {
                eprintln!("[voice: {}]", voice_state_label(state));
            }
            Event::Transcription { text, .. } => {
                if text.trim().is_empty() {
                    eprintln!("[transcription: (no speech detected)]");
                } else {
                    eprintln!("[transcription: {text}]");
                }
            }
            Event::Delta { text, .. } => {
                stdout.write_all(text.as_bytes())?;
                stdout.flush()?;
                wrote_delta = wrote_delta || !text.is_empty();
            }
            Event::ToolCall { name, args, .. } => {
                let preview = args.get("command").and_then(|v| v.as_str()).unwrap_or("");
                if preview.is_empty() {
                    eprintln!("\n[tool call: {name}]");
                } else {
                    eprintln!("\n[tool call: {name} {preview}]");
                }
            }
            Event::ToolResult { name, result, .. } => {
                let exit = result
                    .get("exit_code")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                eprintln!("[tool result: {name} exit:{exit}]");
            }
            Event::Presence { .. } => {}
            Event::Done { .. } => {
                if wrote_delta {
                    writeln!(stdout)?;
                }
                return Ok(());
            }
            Event::Error { message, .. } => {
                if wrote_delta {
                    writeln!(stdout)?;
                }
                eprintln!("daemon error: {message}");
                std::process::exit(1);
            }
        }
    }
}

fn voice_state_label(s: VoiceCaptureState) -> &'static str {
    match s {
        VoiceCaptureState::Idle => "idle",
        VoiceCaptureState::Recording => "recording",
        VoiceCaptureState::Transcribing => "transcribing",
    }
}
