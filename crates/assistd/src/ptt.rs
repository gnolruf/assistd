//! `ptt-start` and `ptt-stop` subcommands.

use std::io::Write;

use anyhow::Result;
use assistd_ipc::{Event, Request, VoiceCaptureState};
use uuid::Uuid;

use crate::ipc_helper::run_one_shot;

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
    let req = action.to_request(Uuid::new_v4().to_string());
    let mut stdout = std::io::stdout().lock();
    let mut wrote_delta = false;
    run_one_shot(req, |event| {
        match event {
            Event::VoiceState { state, .. } => {
                eprintln!("[voice: {}]", voice_state_label(*state));
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
            Event::Status {
                severity,
                component,
                message,
                ..
            } => {
                eprintln!("[{severity} {component}: {message}]");
            }
            Event::ConfirmRequest { .. } => {
                eprintln!(
                    "[daemon asked for destructive-command confirmation; denying \
                     (non-interactive ptt)]"
                );
            }
            Event::Done { .. } | Event::Error { .. } if wrote_delta => {
                writeln!(stdout)?;
            }
            _ => {}
        }
        Ok(())
    })
    .await
}

fn voice_state_label(s: VoiceCaptureState) -> &'static str {
    match s {
        VoiceCaptureState::Idle => "idle",
        VoiceCaptureState::Queued => "queued",
        VoiceCaptureState::Recording => "recording",
        VoiceCaptureState::Transcribing => "transcribing",
    }
}
