//! Background worker that embeds queued rows in batches and stores the
//! vectors through the memory writer. A row whose embed fails is logged
//! and dropped, never retried; it stays unindexed until a reindex.

use std::sync::Arc;

use assistd_memory::{WriteOp, vector_to_blob};
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::JoinHandle;

use crate::{BATCH_SIZE, EmbedError, Embedder, embed_each};

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

/// Spawn the worker. Jobs already queued together are embedded in one
/// request of up to [`BATCH_SIZE`] inputs. The task exits when the job
/// channel closes, or once `shutdown` flips to `true` and the jobs
/// already queued are stored. Vectors reach the database only through
/// `writer_tx`, so the memory writer must outlive the task.
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
                    handle_batch(&*embedder, &writer_tx, &mut batch).await;
                }
                _ = shutdown.changed() => {
                    if *shutdown.borrow() {
                        tracing::debug!(
                            target: "assistd::embed",
                            "shutdown received; draining embed queue"
                        );
                        loop {
                            batch.extend(
                                std::iter::from_fn(|| rx.try_recv().ok()).take(BATCH_SIZE),
                            );
                            if batch.is_empty() {
                                break;
                            }
                            handle_batch(&*embedder, &writer_tx, &mut batch).await;
                        }
                        break;
                    }
                }
            }
        }
    })
}

/// Embed and store every job in `batch`, leaving it empty.
async fn handle_batch(
    embedder: &dyn Embedder,
    writer_tx: &mpsc::Sender<WriteOp>,
    batch: &mut Vec<EmbedJob>,
) {
    let texts: Vec<&str> = batch.iter().map(EmbedJob::text).collect();
    let results = embed_each(embedder, &texts).await;
    for (job, result) in batch.drain(..).zip(results) {
        store(embedder, writer_tx, job, result).await;
    }
}

async fn store(
    embedder: &dyn Embedder,
    writer_tx: &mpsc::Sender<WriteOp>,
    job: EmbedJob,
    result: Result<Vec<f32>, EmbedError>,
) {
    let (kind, rowid) = job.target();
    let vec = match result {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(
                target: "assistd::embed",
                kind,
                rowid,
                error = %e,
                "embed failed; row stays unindexed (backfill can recover)"
            );
            return;
        }
    };
    let model = embedder.model().to_string();
    let dim = embedder.dim() as i64;
    let vector = vector_to_blob(&vec);
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
        Ok(Err(e)) => tracing::warn!(
            target: "assistd::embed",
            kind,
            rowid,
            error = %e,
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
