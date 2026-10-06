//! Background worker that embeds queued rows in batches and stores the vectors through
//! the memory writer. A row whose embed fails is logged and left unindexed.

use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use assistd_memory::{WriteOp, vector_to_blob};
use tokio::sync::mpsc::error::TrySendError;
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::JoinHandle;

use crate::{BATCH_SIZE, EmbedError, Embedder, embed_each};

/// Shortest gap between two "embed queue full" warnings.
const QUEUE_FULL_WARN_INTERVAL: Duration = Duration::from_secs(60);

static QUEUE_FULL_WARNINGS: Mutex<DropWarnLimiter> = Mutex::new(DropWarnLimiter::new());

/// One row to embed.
#[derive(Debug)]
pub enum EmbedJob {
    Chunk { chunk_id: i64, text: String },
    Memory { memory_id: i64, text: String },
}

impl EmbedJob {
    fn text(&self) -> &str {
        match self {
            Self::Chunk { text, .. } | Self::Memory { text, .. } => text,
        }
    }

    fn target(&self) -> (&'static str, i64) {
        match *self {
            Self::Chunk { chunk_id, .. } => ("chunk", chunk_id),
            Self::Memory { memory_id, .. } => ("memory", memory_id),
        }
    }
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
        Err(TrySendError::Closed(job)) => {
            let (kind, rowid) = job.target();
            tracing::debug!(
                target: "assistd::embed",
                kind,
                rowid,
                "embed queue closed; row stays unindexed"
            );
        }
    }
}

fn warn_queue_full(job: &EmbedJob) {
    let (kind, rowid) = job.target();
    let due = QUEUE_FULL_WARNINGS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .record_drop(Instant::now());
    match due {
        Some(suppressed) => tracing::warn!(
            target: "assistd::embed",
            kind,
            rowid,
            suppressed,
            "embed queue full; dropping job, row stays unindexed until `assistd memory reindex`"
        ),
        None => tracing::debug!(
            target: "assistd::embed",
            kind,
            rowid,
            "embed queue full; dropping job"
        ),
    }
}

/// Spawn the worker, embedding queued jobs in batches of up to [`BATCH_SIZE`]. It exits
/// when the job channel closes, or on `shutdown` once queued jobs are stored; the memory
/// writer behind `writer_tx` must outlive it.
pub fn spawn_embedder_task(
    embedder: Arc<dyn Embedder>,
    writer_tx: Arc<mpsc::Sender<WriteOp>>,
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
                    embed_and_store_batch(&*embedder, &writer_tx, &mut batch).await;
                }
                _ = shutdown.changed() => {
                    if *shutdown.borrow() {
                        drain_queue(&*embedder, &writer_tx, &mut rx, &mut batch).await;
                        break;
                    }
                }
            }
        }
    })
}

async fn drain_queue(
    embedder: &dyn Embedder,
    writer_tx: &mpsc::Sender<WriteOp>,
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
        embed_and_store_batch(embedder, writer_tx, batch).await;
    }
}

/// Embed and store every job in `batch`, leaving it empty.
async fn embed_and_store_batch(
    embedder: &dyn Embedder,
    writer_tx: &mpsc::Sender<WriteOp>,
    batch: &mut Vec<EmbedJob>,
) {
    let texts: Vec<&str> = batch.iter().map(EmbedJob::text).collect();
    let results = embed_each(embedder, &texts).await;
    for (job, result) in batch.drain(..).zip(results) {
        store_embedding(embedder, writer_tx, job, result).await;
    }
}

async fn store_embedding(
    embedder: &dyn Embedder,
    writer_tx: &mpsc::Sender<WriteOp>,
    job: EmbedJob,
    result: Result<Vec<f32>, EmbedError>,
) {
    let (kind, rowid) = job.target();
    let embedding = match result {
        Ok(embedding) => embedding,
        Err(err) => {
            tracing::warn!(
                target: "assistd::embed",
                kind,
                rowid,
                error = %err,
                "embed failed; row stays unindexed (backfill can recover)"
            );
            return;
        }
    };
    let model = embedder.model().to_string();
    let dim = i64::try_from(embedder.dim()).unwrap_or(i64::MAX);
    let vector = vector_to_blob(&embedding);
    let (ack_tx, ack_rx) = oneshot::channel();
    let op = match job {
        EmbedJob::Chunk { chunk_id, .. } => WriteOp::StoreChunkEmbedding {
            chunk_id,
            model,
            dim,
            vector,
            ack: ack_tx,
        },
        EmbedJob::Memory { memory_id, .. } => WriteOp::StoreMemoryEmbedding {
            memory_id,
            model,
            dim,
            vector,
            ack: ack_tx,
        },
    };
    if writer_tx.send(op).await.is_err() {
        tracing::warn!(
            target: "assistd::embed",
            kind,
            rowid,
            "memory writer task is gone; dropping embedding"
        );
        return;
    }
    match ack_rx.await {
        Ok(Ok(())) => {}
        Ok(Err(err)) => tracing::warn!(
            target: "assistd::embed",
            kind,
            rowid,
            error = %err,
            "memory writer rejected embedding"
        ),
        Err(_) => tracing::warn!(
            target: "assistd::embed",
            kind,
            rowid,
            "memory writer dropped ack channel before responding"
        ),
    }
}

#[cfg(test)]
mod tests;
