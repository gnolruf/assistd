//! The worker side: answer the daemon's requests over stdin and stdout, one
//! at a time, until stdin closes.

use std::io;
use std::path::PathBuf;

use assistd_utils::text::panic_message;
use thiserror::Error;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Stdout};
use tokio::task::JoinError;
use tracing::{error, warn};

use crate::extract::{extract_text, render_page};
use crate::protocol::{
    Extraction, FromWorker, RenderedPage, Request, SkipReason, Task, decode, encode,
};

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
            Ok(request) => send(&mut stdout, &answer(request).await).await?,
            Err(e) => warn!("extraction worker: {e}; line ignored"),
        }
    }
    Ok(())
}

/// Do `request`'s task on the blocking pool, turning a panic in the extractor
/// into a skipped file or a failed page.
async fn answer(request: Request) -> FromWorker {
    let Request {
        id,
        path,
        max_file_bytes,
        task,
    } = request;
    let path = PathBuf::from(path);
    match task {
        Task::Extract => {
            let extracting =
                tokio::task::spawn_blocking(move || extract_text(&path, max_file_bytes));
            let extraction = extracting
                .await
                .unwrap_or_else(|failure| Extraction::Skipped {
                    reason: extractor_failure(failure),
                });
            FromWorker::Reply { id, extraction }
        }
        Task::RenderPage { page } => {
            let rendering =
                tokio::task::spawn_blocking(move || render_page(&path, max_file_bytes, page));
            let page = rendering
                .await
                .unwrap_or_else(|failure| RenderedPage::Failed {
                    reason: extractor_failure(failure),
                });
            FromWorker::Rendered { id, page }
        }
    }
}

fn extractor_failure(failure: JoinError) -> SkipReason {
    let message = match failure.try_into_panic() {
        Ok(panic) => format!("the extractor panicked: {}", panic_message(panic.as_ref())),
        Err(cancelled) => cancelled.to_string(),
    };
    error!("extraction worker: {message}");
    SkipReason::ExtractorFailed { message }
}

async fn send(stdout: &mut Stdout, message: &FromWorker) -> io::Result<()> {
    stdout.write_all(encode(message).as_bytes()).await?;
    stdout.flush().await
}
