//! `query` subcommand.

use std::io::Write;
use std::path::PathBuf;

use anyhow::Result;
use assistd_ipc::attachment::{LoadImageError, LoadedImage, MAX_IMAGE_BYTES, load_image};
use assistd_ipc::{Event, ImageAttachment, Request};
use clap::Args;
use uuid::Uuid;

use crate::ipc_helper::run_one_shot;

const PREVIEW_MAX_CHARS: usize = 80;

#[derive(Args)]
pub struct QueryArgs {
    /// Text to send to the daemon.
    pub text: String,
    /// Attach one or more images as vision inputs for this turn. Repeat
    /// the flag to attach multiple. Each path must point to a PNG, JPEG,
    /// or WebP file under 32 MiB.
    #[arg(long = "image", value_name = "PATH")]
    pub images: Vec<PathBuf>,
}

pub async fn run(args: QueryArgs) -> Result<()> {
    let attachments = load_attachments(&args.images).await?;
    let req = if attachments.is_empty() {
        Request::query(Uuid::new_v4().to_string(), args.text)
    } else {
        Request::query_with_attachments(Uuid::new_v4().to_string(), args.text, attachments)
    };

    let mut stdout = std::io::stdout().lock();
    let mut wrote_anything = false;
    run_one_shot(req, |event| {
        print_event(&mut stdout, event, &mut wrote_anything)
    })
    .await
}

async fn load_attachments(paths: &[PathBuf]) -> Result<Vec<ImageAttachment>> {
    let mut attachments = Vec::with_capacity(paths.len());
    for path in paths {
        match load_image(path).await {
            Ok(LoadedImage { mime, bytes }) => {
                attachments.push(ImageAttachment::from_bytes(mime, &bytes));
            }
            Err(e) => {
                let kind = match &e {
                    LoadImageError::TooLarge { .. } => {
                        format!("(max {} MiB)", MAX_IMAGE_BYTES / (1024 * 1024))
                    }
                    _ => String::new(),
                };
                anyhow::bail!("--image {}: {} {kind}", path.display(), e.user_message());
            }
        }
    }
    Ok(attachments)
}

/// `wrote_anything` records whether reply text was printed, which owes a
/// trailing newline on `Done`/`Error`.
fn print_event(out: &mut impl Write, event: &Event, wrote_anything: &mut bool) -> Result<()> {
    match event {
        Event::Delta { text, .. } => {
            out.write_all(text.as_bytes())?;
            out.flush()?;
            *wrote_anything = *wrote_anything || !text.is_empty();
        }
        Event::ToolCall { name, args, .. } => {
            let preview = args
                .get("command")
                .and_then(|v| v.as_str())
                .map(truncate_preview)
                .unwrap_or_default();
            if preview.is_empty() {
                writeln!(out, "\n[tool call: {name}]")?;
            } else {
                writeln!(out, "\n[tool call: {name} {preview}]")?;
            }
        }
        Event::ToolResult { name, result, .. } => {
            let exit = result
                .get("exit_code")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            writeln!(out, "[tool result: {name} exit:{exit}]")?;
        }
        Event::Presence { state, .. } => {
            writeln!(out, "[presence: {state:?}]")?;
        }
        Event::VoiceState { state, .. } => {
            writeln!(out, "[voice: {state:?}]")?;
        }
        Event::Transcription { text, .. } => {
            if !text.is_empty() {
                writeln!(out, "[transcription: {text}]")?;
            }
        }
        Event::ListenState { active, .. } => {
            writeln!(out, "[listen: {}]", if *active { "on" } else { "off" })?;
        }
        Event::VoiceOutputState { enabled, .. } => {
            writeln!(
                out,
                "[voice-output: {}]",
                if *enabled { "on" } else { "off" }
            )?;
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
                 (non-interactive query)]"
            );
        }
        Event::Done { .. } | Event::Error { .. } if *wrote_anything => {
            writeln!(out)?;
        }
        _ => {}
    }
    Ok(())
}

fn truncate_preview(s: &str) -> String {
    let mut out: String = s.chars().take(PREVIEW_MAX_CHARS).collect();
    if s.chars().count() > PREVIEW_MAX_CHARS {
        out.push('…');
    }
    out
}
