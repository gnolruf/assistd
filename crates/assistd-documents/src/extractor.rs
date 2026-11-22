//! The daemon's end of the pipe: the worker's readiness, and the one request
//! it is answering.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use assistd_utils::pipe_worker::{CallError, ExitBlame, PendingReplies};
use assistd_utils::readiness::NotReady;
use parking_lot::Mutex;
use thiserror::Error;
use tokio::sync::{mpsc, oneshot, watch};

use crate::protocol::{Extraction, FromWorker, RenderedPage, Request, SkipReason, Task};

/// How long the worker has to answer one request before it is restarted.
const REQUEST_DEADLINE: Duration = Duration::from_secs(120);

/// A request the worker did not get to answer.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ExtractorError {
    #[error("document extraction is {0}")]
    NotReady(#[from] NotReady),
    #[error("the extraction worker exited before it answered")]
    Exited,
}

/// Where the answer to the pending request goes, by the message that carries it.
#[derive(Debug)]
enum Awaited {
    Extraction(oneshot::Sender<Extraction>),
    Page(oneshot::Sender<RenderedPage>),
}

/// The file the worker is reading, and the one a worker last exited over.
#[derive(Debug, Default)]
struct FilesAtExit {
    reading: Option<String>,
    exited_over: Option<String>,
}

impl FilesAtExit {
    /// Account for the worker's exit. The second in a row over one file is held
    /// against the file, and the one after against the worker again.
    fn blame_exit(&mut self) -> ExitBlame {
        let Some(reading) = self.reading.take() else {
            return ExitBlame::Worker;
        };
        if self.exited_over.as_ref() == Some(&reading) {
            self.exited_over = None;
            ExitBlame::Work
        } else {
            self.exited_over = Some(reading);
            ExitBlame::Worker
        }
    }
}

/// The extraction worker as the daemon sees it for its whole lifetime: how far
/// the worker has started, and its answers while one runs.
#[derive(Debug)]
pub struct Extractor {
    readiness: watch::Sender<Result<(), NotReady>>,
    replies: PendingReplies<Request, Awaited>,
    files: Mutex<FilesAtExit>,
    /// Held for a whole request, so the worker reads one file at a time.
    turn: tokio::sync::Mutex<()>,
}

impl Extractor {
    /// An extractor whose worker has yet to start.
    pub fn starting() -> Self {
        Self::with_readiness(Err(NotReady::Starting))
    }

    /// An extractor that gets no worker, for `reason`.
    pub fn unavailable(reason: impl Into<Arc<str>>) -> Self {
        Self::with_readiness(Err(NotReady::Unavailable(reason.into())))
    }

    fn with_readiness(readiness: Result<(), NotReady>) -> Self {
        Self {
            readiness: watch::channel(readiness).0,
            replies: PendingReplies::new("extraction worker"),
            files: Mutex::default(),
            turn: tokio::sync::Mutex::new(()),
        }
    }

    /// Whether a worker is up to take requests, or why none is.
    pub fn readiness(&self) -> Result<(), NotReady> {
        self.readiness.borrow().clone()
    }

    /// Follows [`Self::readiness`] as the worker starts, dies and restarts.
    pub fn subscribe_readiness(&self) -> watch::Receiver<Result<(), NotReady>> {
        self.readiness.subscribe()
    }

    /// The text of the file at `path` page by page, or why it was skipped; a file
    /// larger than `max_file_bytes` is. Requests are answered one at a time. A
    /// worker that misses the deadline skips the file as timed out and is restarted.
    pub async fn extract(
        &self,
        path: &Path,
        max_file_bytes: u64,
    ) -> Result<Extraction, ExtractorError> {
        let skipped = |reason| Extraction::Skipped { reason };
        let Some(path) = path.to_str() else {
            return Ok(skipped(path_not_utf8()));
        };
        let (reply_tx, reply) = oneshot::channel();
        let request = PendingRequest {
            path: path.to_owned(),
            max_file_bytes,
            task: Task::Extract,
            awaited: Awaited::Extraction(reply_tx),
        };
        let answered = self.answer(request, reply).await?;
        Ok(answered.unwrap_or_else(|| skipped(SkipReason::TimedOut)))
    }

    /// Page `page` of the PDF at `path`, counted from zero, as an image, under the
    /// limits and the deadline of [`Self::extract`]; a page the worker does not
    /// render in time is a failed one.
    pub async fn render_page(
        &self,
        path: &Path,
        max_file_bytes: u64,
        page: u32,
    ) -> Result<RenderedPage, ExtractorError> {
        let failed = |reason| RenderedPage::Failed { reason };
        let Some(path) = path.to_str() else {
            return Ok(failed(path_not_utf8()));
        };
        let (reply_tx, reply) = oneshot::channel();
        let request = PendingRequest {
            path: path.to_owned(),
            max_file_bytes,
            task: Task::RenderPage { page },
            awaited: Awaited::Page(reply_tx),
        };
        let answered = self.answer(request, reply).await?;
        Ok(answered.unwrap_or_else(|| failed(SkipReason::TimedOut)))
    }

    /// Send `request` once the worker is free and wait for `reply`; `None` when
    /// the worker missed the deadline, which gets it restarted.
    async fn answer<T>(
        &self,
        request: PendingRequest,
        reply: oneshot::Receiver<T>,
    ) -> Result<Option<T>, ExtractorError> {
        let _turn = self.turn.lock().await;
        self.readiness()?;
        let PendingRequest {
            path,
            max_file_bytes,
            task,
            awaited,
        } = request;
        self.files.lock().reading = Some(path.clone());
        let request = |id| Request {
            id,
            path,
            max_file_bytes,
            task,
        };
        let answered = self
            .replies
            .call(request, awaited, reply, Some(REQUEST_DEADLINE))
            .await;
        self.files.lock().reading = None;
        match answered {
            Ok(answered) => Ok(Some(answered)),
            Err(CallError::Unresponsive { .. }) => Ok(None),
            Err(CallError::NotRunning | CallError::Exited) => Err(ExtractorError::Exited),
        }
    }

    pub(crate) fn set_readiness(&self, readiness: Result<(), NotReady>) {
        self.readiness.send_if_modified(|current| {
            let changed = *current != readiness;
            *current = readiness;
            changed
        });
    }

    /// Route requests to a newly spawned worker through `outgoing`; the returned
    /// receiver resolves once that worker misses a request's deadline.
    pub(crate) fn connect(
        &self,
        outgoing: mpsc::UnboundedSender<Request>,
    ) -> oneshot::Receiver<()> {
        self.replies.connect(outgoing)
    }

    /// The worker is gone: no worker is ready, and its pending request fails.
    /// Returns whether it exited over the file the worker before it exited over.
    pub(crate) fn disconnect(&self) -> ExitBlame {
        self.set_readiness(Err(NotReady::Starting));
        let blame = self.files.lock().blame_exit();
        self.replies.disconnect();
        blame
    }

    /// Act on one message from the worker: it is ready, or it answered. An answer
    /// the pending request did not ask for is ignored.
    pub(crate) fn receive(&self, message: FromWorker) {
        match message {
            FromWorker::Ready => self.set_readiness(Ok(())),
            FromWorker::Reply { id, extraction } => {
                let awaited = self
                    .replies
                    .take_if(id, |awaited| matches!(awaited, Awaited::Extraction(_)));
                if let Some(Awaited::Extraction(reply)) = awaited {
                    let _ = reply.send(extraction.capped());
                }
            }
            FromWorker::Rendered { id, page } => {
                let awaited = self
                    .replies
                    .take_if(id, |awaited| matches!(awaited, Awaited::Page(_)));
                if let Some(Awaited::Page(reply)) = awaited {
                    let _ = reply.send(page.capped());
                }
            }
        }
    }
}

/// One request on its way to the worker, and where its answer goes.
struct PendingRequest {
    path: String,
    max_file_bytes: u64,
    task: Task,
    awaited: Awaited,
}

fn path_not_utf8() -> SkipReason {
    SkipReason::Unreadable {
        message: "its path is not valid UTF-8".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What the exit of a worker that dies as soon as it is asked for `path` is
    /// held against.
    async fn exit_while_reading(extractor: &Extractor, path: &str) -> ExitBlame {
        let (outgoing, mut requests) = mpsc::unbounded_channel();
        let _unresponsive = extractor.connect(outgoing);
        extractor.set_readiness(Ok(()));
        let dying = async {
            requests.recv().await.expect("the request is sent");
            extractor.disconnect()
        };
        let (extraction, blame) = tokio::join!(extractor.extract(Path::new(path), 1), dying);
        assert_eq!(extraction, Err(ExtractorError::Exited));
        blame
    }

    #[tokio::test]
    async fn only_the_second_exit_in_a_row_over_one_file_is_held_against_the_file() {
        let extractor = Extractor::starting();
        let blames = [
            exit_while_reading(&extractor, "/notes/fatal.pdf").await,
            exit_while_reading(&extractor, "/notes/fatal.pdf").await,
            exit_while_reading(&extractor, "/notes/fatal.pdf").await,
            exit_while_reading(&extractor, "/notes/other.pdf").await,
            extractor.disconnect(),
        ];
        assert_eq!(
            blames,
            [
                ExitBlame::Worker,
                ExitBlame::Work,
                ExitBlame::Worker,
                ExitBlame::Worker,
                ExitBlame::Worker,
            ]
        );
    }
}
