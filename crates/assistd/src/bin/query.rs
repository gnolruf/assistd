use anyhow::{Context, Result};
use clap::Parser;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

use assistd_core::ipc::{self, Request, Response};

#[derive(Parser)]
#[command(
    name = "assistd-query",
    version,
    about = "Send a one-shot query to the running assistd daemon"
)]
struct Cli {
    /// Text to send to the daemon
    text: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let path = ipc::socket_path();

    let stream = UnixStream::connect(&path).await.with_context(|| {
        format!(
            "assistd daemon is not running (could not connect to {})",
            path.display()
        )
    })?;

    let (read_half, mut write_half) = stream.into_split();
    let req = Request::Query { text: cli.text };
    let mut body = serde_json::to_string(&req)?;
    body.push('\n');
    write_half.write_all(body.as_bytes()).await?;
    write_half.shutdown().await?;

    let mut reader = BufReader::new(read_half);
    let mut line = String::new();
    let n = reader.read_line(&mut line).await?;
    if n == 0 {
        anyhow::bail!("daemon closed the connection without sending a response");
    }

    match serde_json::from_str::<Response>(line.trim())
        .with_context(|| format!("invalid JSON from daemon: {}", line.trim()))?
    {
        Response::Response { text } => {
            println!("{text}");
            Ok(())
        }
        Response::Error { message } => {
            eprintln!("daemon error: {message}");
            std::process::exit(1);
        }
    }
}
