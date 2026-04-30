use anyhow::Result;
use assistd_ipc::{Event, ImageAttachment, IpcClient, IpcClientError, Request};
use assistd_tools::attachment::{LoadImageError, MAX_IMAGE_BYTES};
use clap::Args;
use std::io::Write;
use std::path::PathBuf;
use uuid::Uuid;

/// Tool-call preview cap: long commands are truncated to keep the status
/// line readable while still showing the shape of the call.
const PREVIEW_MAX_CHARS: usize = 80;

fn truncate_preview(s: &str) -> String {
    let mut out = String::with_capacity(PREVIEW_MAX_CHARS + 1);
    for ch in s.chars().take(PREVIEW_MAX_CHARS) {
        out.push(ch);
    }
    if s.chars().count() > PREVIEW_MAX_CHARS {
        out.push('…');
    }
    out
}

#[derive(Args)]
pub struct QueryArgs {
    /// Text to send to the daemon
    pub text: String,
    /// Attach one or more images as vision inputs for this turn. Repeat
    /// the flag to attach multiple. Each path must point to a PNG, JPEG,
    /// or WebP file under 32 MiB.
    #[arg(long = "image", value_name = "PATH")]
    pub images: Vec<PathBuf>,
}

pub async fn run(args: QueryArgs) -> Result<()> {
    let mut wire_attachments = Vec::with_capacity(args.images.len());
    for path in &args.images {
        // Reject the bad path before opening the daemon socket so the
        // user sees the error in their shell, not as a daemon Event.
        match assistd_tools::load_image_attachment(path).await {
            Ok((assistd_tools::Attachment::Image { mime, bytes }, _)) => {
                wire_attachments.push(ImageAttachment::from_bytes(mime, &bytes));
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

    let client = IpcClient::new();
    let req = if wire_attachments.is_empty() {
        Request::query(Uuid::new_v4().to_string(), args.text)
    } else {
        Request::query_with_attachments(Uuid::new_v4().to_string(), args.text, wire_attachments)
    };
    let mut stream = client.one_shot(req).await.map_err(|e| match e {
        IpcClientError::NotReachable { path, source } => anyhow::anyhow!(
            "assistd daemon is not running (could not connect to {}): {source}",
            path.display()
        ),
        other => anyhow::Error::from(other),
    })?;

    let stdout = std::io::stdout();
    let mut stdout = stdout.lock();
    let mut wrote_anything = false;

    loop {
        let event = match stream.next_event().await? {
            Some(ev) => ev,
            None => anyhow::bail!("daemon closed the connection without sending a terminal event"),
        };

        match event {
            Event::Delta { text, .. } => {
                stdout.write_all(text.as_bytes())?;
                stdout.flush()?;
                wrote_anything = wrote_anything || !text.is_empty();
            }
            Event::ToolCall { name, args, .. } => {
                // Prefix tool calls with a line separator so the stream
                // remains readable when they land mid-response. Extract
                // the command preview so the user sees what the model
                // is running, not just the tool name.
                let preview = args
                    .get("command")
                    .and_then(|v| v.as_str())
                    .map(truncate_preview)
                    .unwrap_or_default();
                if preview.is_empty() {
                    writeln!(stdout, "\n[tool call: {name}]")?;
                } else {
                    writeln!(stdout, "\n[tool call: {name} {preview}]")?;
                }
            }
            Event::ToolResult { name, result, .. } => {
                let exit = result
                    .get("exit_code")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                writeln!(stdout, "[tool result: {name} exit:{exit}]")?;
            }
            Event::Presence { state, .. } => {
                writeln!(stdout, "[presence: {state:?}]")?;
            }
            Event::VoiceState { state, .. } => {
                writeln!(stdout, "[voice: {state:?}]")?;
            }
            Event::Transcription { text, .. } => {
                if !text.is_empty() {
                    writeln!(stdout, "[transcription: {text}]")?;
                }
            }
            Event::ListenState { active, .. } => {
                writeln!(stdout, "[listen: {}]", if active { "on" } else { "off" })?;
            }
            Event::VoiceOutputState { enabled, .. } => {
                writeln!(
                    stdout,
                    "[voice-output: {}]",
                    if enabled { "on" } else { "off" }
                )?;
            }
            // Memory* events shouldn't appear on a Query stream — the
            // daemon only emits them on `Request::Memory*`. Tolerate
            // them silently in case a future feature reuses the wire.
            Event::SemanticHit { .. }
            | Event::MemoryValue { .. }
            | Event::MemoryKeys { .. }
            | Event::MemoryRow { .. }
            | Event::MemoryForgetResult { .. }
            | Event::ReindexProgress { .. }
            | Event::Capabilities { .. } => {}
            // `assistd query` is one-shot non-interactive: it can't
            // answer a confirm prompt. The daemon-side gate denies
            // when the connection drops, so we just note the event
            // and keep reading until the agent emits its denial Done.
            // The interactive flow lives in the TUI via DialogConnection.
            Event::ConfirmRequest { .. } => {
                eprintln!(
                    "[daemon asked for destructive-command confirmation; denying \
                     (non-interactive query)]"
                );
            }
            Event::Done { .. } => {
                if wrote_anything {
                    writeln!(stdout)?;
                }
                return Ok(());
            }
            Event::Error { message, .. } => {
                if wrote_anything {
                    writeln!(stdout)?;
                }
                eprintln!("daemon error: {message}");
                std::process::exit(1);
            }
        }
    }
}
