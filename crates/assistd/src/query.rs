use anyhow::{Context, Result};
use assistd_ipc::{Event, Request, socket_path};
use clap::Args;
use std::io::Write;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use uuid::Uuid;

#[derive(Args)]
pub struct QueryArgs {
    /// Text to send to the daemon
    pub text: String,
}

pub async fn run(args: QueryArgs) -> Result<()> {
    let path = socket_path();

    let stream = UnixStream::connect(&path).await.with_context(|| {
        format!(
            "assistd daemon is not running (could not connect to {})",
            path.display()
        )
    })?;

    let (read_half, mut write_half) = stream.into_split();
    let req = Request::Query {
        id: Uuid::new_v4().to_string(),
        text: args.text,
    };
    let mut body = serde_json::to_string(&req)?;
    body.push('\n');
    write_half.write_all(body.as_bytes()).await?;
    write_half.shutdown().await?;

    let mut reader = BufReader::new(read_half);
    let stdout = std::io::stdout();
    let mut stdout = stdout.lock();
    let mut wrote_anything = false;

    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line).await?;
        if n == 0 {
            anyhow::bail!("daemon closed the connection without sending a terminal event");
        }
        let event: Event = serde_json::from_str(line.trim())
            .with_context(|| format!("invalid JSON from daemon: {}", line.trim()))?;

        match event {
            Event::Delta { text, .. } => {
                stdout.write_all(text.as_bytes())?;
                stdout.flush()?;
                wrote_anything = wrote_anything || !text.is_empty();
            }
            Event::ToolCall { name, .. } => {
                // Tool use is reserved for a future milestone; surface to
                // the user so the stream is readable when it lands.
                writeln!(stdout, "\n[tool call: {name}]")?;
            }
            Event::ToolResult { name, .. } => {
                writeln!(stdout, "[tool result: {name}]")?;
            }
            Event::Presence { state, .. } => {
                writeln!(stdout, "[presence: {state:?}]")?;
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
