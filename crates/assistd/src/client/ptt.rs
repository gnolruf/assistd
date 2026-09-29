//! `ptt-start` and `ptt-stop` subcommands.

use std::io::{self, Write};

use anyhow::Result;
use assistd_ipc::{Event, Request, VoiceCaptureState};
use uuid::Uuid;

use super::run_one_shot;
use super::terminal_text::{escape_controls, escape_controls_single_line};

#[derive(Debug, Clone, Copy)]
pub(crate) enum PttAction {
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

pub(crate) async fn run(action: PttAction) -> Result<()> {
    let req = action.to_request(Uuid::new_v4().to_string());
    let mut stdout = io::stdout().lock();
    let mut wrote_delta = false;
    run_one_shot(req, |event| {
        match event {
            Event::VoiceState { state, .. } => {
                writeln!(io::stderr(), "[voice: {}]", voice_state_label(*state))?;
            }
            Event::Transcription { text, .. } => {
                if text.trim().is_empty() {
                    writeln!(io::stderr(), "[transcription: (no speech detected)]")?;
                } else {
                    writeln!(
                        io::stderr(),
                        "[transcription: {}]",
                        escape_controls_single_line(text)
                    )?;
                }
            }
            Event::Delta { text, .. } => {
                stdout.write_all(escape_controls(text).as_bytes())?;
                stdout.flush()?;
                wrote_delta = wrote_delta || !text.is_empty();
            }
            Event::ToolCall { name, args, .. } => {
                let name = escape_controls_single_line(name);
                let preview = escape_controls_single_line(
                    args.get("command").and_then(|v| v.as_str()).unwrap_or(""),
                );
                if preview.is_empty() {
                    writeln!(io::stderr(), "\n[tool call: {name}]")?;
                } else {
                    writeln!(io::stderr(), "\n[tool call: {name} {preview}]")?;
                }
            }
            Event::ToolResult { name, result, .. } => {
                let name = escape_controls_single_line(name);
                let exit = result
                    .get("exit_code")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                writeln!(io::stderr(), "[tool result: {name} exit:{exit}]")?;
            }
            Event::Status {
                severity,
                component,
                message,
                ..
            } => {
                writeln!(
                    io::stderr(),
                    "[{severity} {component}: {}]",
                    escape_controls_single_line(message)
                )?;
            }
            Event::ConfirmRequest { .. } => {
                writeln!(
                    io::stderr(),
                    "[daemon asked for destructive-command confirmation; denying \
                     (non-interactive ptt)]"
                )?;
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

fn voice_state_label(state: VoiceCaptureState) -> &'static str {
    match state {
        VoiceCaptureState::Idle => "idle",
        VoiceCaptureState::Queued => "queued",
        VoiceCaptureState::Recording => "recording",
        VoiceCaptureState::Transcribing => "transcribing",
    }
}
