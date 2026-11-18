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

use crate::protocol::{Extraction, FromWorker, Request, SkipReason};

/// How long the worker has to answer one request before it is restarted.
const EXTRACTION_DEADLINE: Duration = Duration::from_secs(120);

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
    pending: Option<(u64, oneshot::Sender<Extraction>)>,
    unresponsive: Option<oneshot::Sender<()>>,
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
        let Some(path) = path.to_str() else {
            return Ok(Extraction::Skipped {
                reason: SkipReason::Unreadable {
                    message: "its path is not valid UTF-8".into(),
                },
            });
        };
        let _turn = self.turn.lock().await;
        self.readiness()?;
        let (id, mut reply) = self.send_request(path.to_owned(), max_file_bytes)?;
        if let Ok(answered) = tokio::time::timeout(EXTRACTION_DEADLINE, &mut reply).await {
            return answered.map_err(|_| ExtractorError::Exited);
        }
        if self.abandon(id) {
            return Ok(Extraction::Skipped {
                reason: SkipReason::TimedOut,
            });
        }
        reply.await.map_err(|_| ExtractorError::Exited)
    }

    fn send_request(
        &self,
        path: String,
        max_file_bytes: u64,
    ) -> Result<(u64, oneshot::Receiver<Extraction>), ExtractorError> {
        let mut connection = self.connection.lock();
        let connection = connection.as_mut().ok_or(ExtractorError::Exited)?;
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (reply_tx, reply_rx) = oneshot::channel();
        connection.pending = Some((id, reply_tx));
        let request = Request {
            id,
            path,
            max_file_bytes,
        };
        connection
            .outgoing
            .send(request)
            .map_err(|_| ExtractorError::Exited)?;
        Ok((id, reply_rx))
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

    /// Act on one message from the worker: it is ready, or it answered.
    pub(crate) fn receive(&self, message: FromWorker) {
        match message {
            FromWorker::Ready => self.set_readiness(Ok(())),
            FromWorker::Reply { id, extraction } => {
                let reply = self.connection.lock().as_mut().and_then(|connection| {
                    connection.pending.take_if(|(pending, _)| *pending == id)
                });
                if let Some((_, reply)) = reply {
                    let _ = reply.send(extraction.capped());
                }
            }
        }
    }
}
