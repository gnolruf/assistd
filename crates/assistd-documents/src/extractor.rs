//! The daemon's end of the pipe: the worker's readiness, and the one request
//! it is answering.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use assistd_utils::readiness::NotReady;
use thiserror::Error;
use tokio::sync::{mpsc, oneshot, watch};
use tracing::warn;

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

/// The running worker's input queue and its request awaiting a reply.
/// Dropping it fails that request with [`ExtractorError::Exited`].
#[derive(Debug)]
struct Connection {
    outgoing: mpsc::UnboundedSender<Request>,
    pending: Option<(u64, Awaited)>,
    unresponsive: Option<oneshot::Sender<()>>,
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
    connection: parking_lot::Mutex<Option<Connection>>,
    /// Held for a whole request, so the worker reads one file at a time.
    turn: tokio::sync::Mutex<()>,
    next_id: AtomicU64,
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
            connection: parking_lot::Mutex::new(None),
            turn: tokio::sync::Mutex::new(()),
            next_id: AtomicU64::new(0),
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

    /// The text of the file at `path`, or why it was skipped; a file larger than
    /// `max_file_bytes` is. Requests are answered one at a time. A worker that
    /// misses the deadline skips the file as timed out and is restarted.
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
        mut reply: oneshot::Receiver<T>,
    ) -> Result<Option<T>, ExtractorError> {
        let _turn = self.turn.lock().await;
        self.readiness()?;
        let id = self.send_request(request)?;
        if let Ok(answered) = tokio::time::timeout(REQUEST_DEADLINE, &mut reply).await {
            return answered.map(Some).map_err(|_| ExtractorError::Exited);
        }
        if self.abandon(id) {
            return Ok(None);
        }
        reply.await.map(Some).map_err(|_| ExtractorError::Exited)
    }

    fn send_request(&self, request: PendingRequest) -> Result<u64, ExtractorError> {
        let PendingRequest {
            path,
            max_file_bytes,
            task,
            awaited,
        } = request;
        let mut connection = self.connection.lock();
        let connection = connection.as_mut().ok_or(ExtractorError::Exited)?;
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        connection.pending = Some((id, awaited));
        let request = Request {
            id,
            path,
            max_file_bytes,
            task,
        };
        connection
            .outgoing
            .send(request)
            .map_err(|_| ExtractorError::Exited)?;
        Ok(id)
    }

    /// Give up on request `id` and flag its worker unresponsive; false when the
    /// request was already answered or its worker already gone.
    fn abandon(&self, id: u64) -> bool {
        let mut connection = self.connection.lock();
        let Some(connection) = connection.as_mut() else {
            return false;
        };
        if connection
            .pending
            .take_if(|(pending, _)| *pending == id)
            .is_none()
        {
            return false;
        }
        warn!(target: "assistd::documents::worker", "extraction worker missed a request's deadline");
        if let Some(unresponsive) = connection.unresponsive.take() {
            let _ = unresponsive.send(());
        }
        true
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
        let (unresponsive_tx, unresponsive_rx) = oneshot::channel();
        *self.connection.lock() = Some(Connection {
            outgoing,
            pending: None,
            unresponsive: Some(unresponsive_tx),
        });
        unresponsive_rx
    }

    /// The worker is gone: fail its pending request.
    pub(crate) fn disconnect(&self) {
        self.connection.lock().take();
    }

    /// Act on one message from the worker: it is ready, or it answered. An answer
    /// the pending request did not ask for is ignored.
    pub(crate) fn receive(&self, message: FromWorker) {
        match message {
            FromWorker::Ready => self.set_readiness(Ok(())),
            FromWorker::Reply { id, extraction } => {
                let awaited =
                    self.take_pending(id, |awaited| matches!(awaited, Awaited::Extraction(_)));
                if let Some(Awaited::Extraction(reply)) = awaited {
                    let _ = reply.send(extraction.capped());
                }
            }
            FromWorker::Rendered { id, page } => {
                let awaited = self.take_pending(id, |awaited| matches!(awaited, Awaited::Page(_)));
                if let Some(Awaited::Page(reply)) = awaited {
                    let _ = reply.send(page.capped());
                }
            }
        }
    }

    /// The pending request when it is `id` and `expects` the answer that arrived.
    fn take_pending(&self, id: u64, expects: impl Fn(&Awaited) -> bool) -> Option<Awaited> {
        let mut connection = self.connection.lock();
        let pending = &mut connection.as_mut()?.pending;
        pending
            .take_if(|(pending, awaited)| *pending == id && expects(awaited))
            .map(|(_, awaited)| awaited)
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
