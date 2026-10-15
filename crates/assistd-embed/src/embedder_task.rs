//! Background worker that embeds queued rows in batches and stores each batch in the
//! semantic store. A row whose embed fails is logged and left unindexed.

use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use assistd_memory::{EmbeddingTarget, SemanticStore};
use tokio::sync::mpsc::error::TrySendError;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;

use crate::{BATCH_SIZE, Embedder, embed_jobs};

/// Shortest gap between two "embed queue full" warnings.
const QUEUE_FULL_WARN_INTERVAL: Duration = Duration::from_secs(60);

static QUEUE_FULL_WARNINGS: Mutex<DropWarnLimiter> = Mutex::new(DropWarnLimiter::new());

/// One row to embed: `text` is the content of `target`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbedJob {
    pub target: EmbeddingTarget,
    pub text: String,
}

/// Rate limit for queue-full warnings, counting the drops it holds back.
#[derive(Debug)]
struct DropWarnLimiter {
    last_warn: Option<Instant>,
    suppressed: u64,
}

impl DropWarnLimiter {
    const fn new() -> Self {
        Self {
            last_warn: None,
            suppressed: 0,
        }
    }

    /// Record a drop at `now`; when a warning is due, the drops held back since the last one.
    fn record_drop(&mut self, now: Instant) -> Option<u64> {
        let due = self
            .last_warn
            .is_none_or(|last| now.duration_since(last) >= QUEUE_FULL_WARN_INTERVAL);
        if !due {
            self.suppressed += 1;
            return None;
        }
        self.last_warn = Some(now);
        Some(std::mem::take(&mut self.suppressed))
    }
}

/// Queue `job` without waiting. A full queue drops it with a rate-limited warning; a
/// closed queue (embedding disabled) drops it quietly. A dropped row stays unindexed.
pub fn enqueue_embed_job(tx: &mpsc::Sender<EmbedJob>, job: EmbedJob) {
    match tx.try_send(job) {
        Ok(()) => {}
        Err(TrySendError::Full(job)) => warn_queue_full(&job),
        Err(TrySendError::Closed(job)) => tracing::debug!(
            target: "assistd::embed",
            row = %job.target,
            "embed queue closed; row stays unindexed"
        ),
    }
}

fn warn_queue_full(job: &EmbedJob) {
    let due = QUEUE_FULL_WARNINGS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .record_drop(Instant::now());
    match due {
        Some(suppressed) => tracing::warn!(
            target: "assistd::embed",
            row = %job.target,
            suppressed,
            "embed queue full; dropping job, row stays unindexed until `assistd memory reindex`"
        ),
        None => tracing::debug!(
            target: "assistd::embed",
            row = %job.target,
            "embed queue full; dropping job"
        ),
    }
}

/// Spawn the worker, embedding queued jobs in batches of up to [`BATCH_SIZE`]. It exits
/// when the job channel closes, or on `shutdown` once queued jobs are stored; the memory
/// writer behind `store` must outlive it.
pub fn spawn_embedder_task(
    embedder: Arc<dyn Embedder>,
    store: Arc<dyn SemanticStore>,
    mut rx: mpsc::Receiver<EmbedJob>,
    mut shutdown: watch::Receiver<bool>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut batch = Vec::with_capacity(BATCH_SIZE);
        loop {
            tokio::select! {
                biased;
                received = rx.recv_many(&mut batch, BATCH_SIZE) => {
                    if received == 0 {
                        tracing::debug!(
                            target: "assistd::embed",
                            "embed channel closed; worker exiting"
                        );
                        break;
                    }
                    embed_and_store_batch(&*embedder, &*store, &mut batch).await;
                }
                _ = shutdown.changed() => {
                    if *shutdown.borrow() {
                        drain_queue(&*embedder, &*store, &mut rx, &mut batch).await;
                        break;
                    }
                }
            }
        }
    })
}

async fn drain_queue(
    embedder: &dyn Embedder,
    store: &dyn SemanticStore,
    rx: &mut mpsc::Receiver<EmbedJob>,
    batch: &mut Vec<EmbedJob>,
) {
    tracing::debug!(
        target: "assistd::embed",
        "shutdown received; draining embed queue"
    );
    loop {
        batch.extend(std::iter::from_fn(|| rx.try_recv().ok()).take(BATCH_SIZE));
        if batch.is_empty() {
            break;
        }
        embed_and_store_batch(embedder, store, batch).await;
    }
}

/// Embed and store every job in `batch`, leaving it empty.
async fn embed_and_store_batch(
    embedder: &dyn Embedder,
    store: &dyn SemanticStore,
    batch: &mut Vec<EmbedJob>,
) {
    let embeddings = embed_jobs(embedder, batch).await;
    batch.clear();
    let rows = embeddings.len();
    if let Err(err) = store.store_embeddings(embedder.model(), embeddings).await {
        tracing::warn!(
            target: "assistd::embed",
            rows,
            error = %err,
            "storing embeddings failed; rows stay unindexed until `assistd memory reindex`"
        );
    }
}

#[cfg(test)]
mod tests;
