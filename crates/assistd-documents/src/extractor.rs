//! The daemon's end of the pipe: the worker's readiness, and the one request
//! it is answering.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use assistd_utils::pipe_worker::{CallError, PendingReplies};
use assistd_utils::readiness::NotReady;
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

/// The extraction worker as the daemon sees it for its whole lifetime: how far
/// the worker has started, and its answers while one runs.
#[derive(Debug)]
pub struct Extractor {
    readiness: watch::Sender<Result<(), NotReady>>,
    replies: PendingReplies<Request, Awaited>,
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
    pub(crate) fn disconnect(&self) {
        self.set_readiness(Err(NotReady::Starting));
        self.replies.disconnect();
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
