//! The worker side: answer the daemon's requests over stdin and stdout, one
//! file at a time, until stdin closes.

use std::io;
use std::path::PathBuf;

use assistd_utils::text::panic_message;
use thiserror::Error;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Stdout};
use tracing::{error, warn};

use crate::extract::extract_text;
use crate::protocol::{Extraction, FromWorker, Request, SkipReason, decode, encode};

/// Why the worker stopped before stdin closed.
#[derive(Debug, Error)]
#[error("extraction worker stdio: {0}")]
pub struct ServeError(#[from] io::Error);

/// Run the extraction worker over this process's stdin and stdout; returns
/// once stdin closes.
pub async fn serve() -> Result<(), ServeError> {
    let mut stdout = tokio::io::stdout();
    send(&mut stdout, &FromWorker::Ready).await?;
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    while let Some(line) = lines.next_line().await? {
        match decode::<Request>(line.as_bytes()) {
            Ok(request) => {
                let reply = FromWorker::Reply {
                    id: request.id,
                    extraction: extract_off_runtime(request).await,
                };
                send(&mut stdout, &reply).await?;
            }
            Err(e) => warn!("extraction worker: {e}; line ignored"),
        }
    }
    Ok(())
}

/// Extract `request`'s file on the blocking pool, turning a panic in the
/// extractor into a skipped file.
async fn extract_off_runtime(request: Request) -> Extraction {
    let Request {
        path,
        max_file_bytes,
        ..
    } = request;
    let extracting =
        tokio::task::spawn_blocking(move || extract_text(&PathBuf::from(path), max_file_bytes));
    extracting.await.unwrap_or_else(|failure| {
        let message = match failure.try_into_panic() {
            Ok(panic) => format!("the extractor panicked: {}", panic_message(panic.as_ref())),
            Err(cancelled) => cancelled.to_string(),
        };
        error!("extraction worker: {message}");
        Extraction::Skipped {
            reason: SkipReason::ExtractorFailed { message },
        }
    })
}

async fn send(stdout: &mut Stdout, message: &FromWorker) -> io::Result<()> {
    stdout.write_all(encode(message).as_bytes()).await?;
    stdout.flush().await
}
