//! `ptt-start` and `ptt-stop` subcommands.

use std::io::{self, Write};

use anyhow::Result;
use assistd_ipc::{EventBody, Request, StatusKind, VoiceCaptureState};
use uuid::Uuid;

use super::run_one_shot;
use super::terminal_text::{escape_controls, escape_controls_single_line};

/// Sends the request `with_id` builds and prints the daemon's answer.
pub(crate) async fn run(with_id: fn(String) -> Request) -> Result<()> {
    let req = with_id(Uuid::new_v4().to_string());
    let mut stdout = io::stdout().lock();
    let mut wrote_delta = false;
    run_one_shot(req, |event| {
        match &event.body {
            EventBody::VoiceState { state } => {
                writeln!(io::stderr(), "[voice: {}]", voice_state_label(*state))?;
            }
            EventBody::Transcription { text } => {
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
            EventBody::Delta { text } => {
                stdout.write_all(escape_controls(text).as_bytes())?;
                stdout.flush()?;
                wrote_delta = wrote_delta || !text.is_empty();
            }
            EventBody::ToolCall { name, args } => {
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
            EventBody::ToolResult { name, result } => {
                let name = escape_controls_single_line(name);
                let exit = result.exit_code;
                writeln!(io::stderr(), "[tool result: {name} exit:{exit}]")?;
            }
            EventBody::Status {
                event: StatusKind::CompactingSoon,
                ..
            } => {}
            EventBody::Status {
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
            EventBody::Done | EventBody::Error { .. } if wrote_delta => {
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
        VoiceCaptureState::Recording => "recording",
        VoiceCaptureState::Transcribing => "transcribing",
    }
}
