//! Background worker that consumes [`EmbedJob`]s, calls the embedder,
//! and dispatches the resulting vector into the memory writer.
//!
//! Mirrors `assistd-memory`'s writer-task pattern:
//! `tokio::select! { biased; op = rx.recv() ...; _ = shutdown.changed() => drain }`
//! so any in-flight jobs land before the daemon exits.
//!
//! Each job carries the rowid that the embedding will FK against. Two
//! variants (chunks FK to `conversation_chunks.id`, memories FK to
//! `memories.id`) share one queue, one worker, and one HTTP
//! connection: there's no semantic value in segregating them and a
//! single queue gives the writer cleaner backpressure shape.
//!
//! Failures are logged and dropped. The chunk / memory row stays
//! without an embedding; a future `assistd memory backfill-embeddings`
//! pass can pick it up. We deliberately avoid retries here; if the
//! embed server is wedged, retrying just amplifies the wedge.

use std::sync::Arc;

use assistd_memory::{WriteOp, vector_to_blob};
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::JoinHandle;

use crate::Embedder;

/// One job for the embedder worker. Variants share one queue so the
/// worker stays simple and ordering is FIFO across both kinds.
#[derive(Debug)]
pub enum EmbedJob {
    /// Embed a `conversation_chunks` row's text. The resulting vector
    /// is committed via [`WriteOp::StoreChunkEmbedding`].
    Chunk { chunk_id: i64, text: String },
    /// Embed a `memories` row's value. The resulting vector is
    /// committed via [`WriteOp::StoreMemoryEmbedding`].
    Memory { memory_id: i64, text: String },
}

/// Spawn the worker. The daemon holds the returned [`JoinHandle`] and
/// awaits it on shutdown so any in-flight embeddings land before the
/// memory writer task drains.
pub fn spawn_embedder_task(
    embedder: Arc<dyn Embedder>,
    writer_tx: Arc<mpsc::Sender<WriteOp>>,
    mut rx: mpsc::Receiver<EmbedJob>,
    mut shutdown: watch::Receiver<bool>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            tokio::select! {
                biased;
                op = rx.recv() => {
                    match op {
                        Some(job) => handle_job(&*embedder, &writer_tx, job).await,
                        None => {
                            tracing::debug!(
                                target: "assistd::embed",
                                "embed channel closed; worker exiting"
                            );
                            break;
                        }
                    }
                }
                _ = shutdown.changed() => {
                    if *shutdown.borrow() {
                        tracing::debug!(
                            target: "assistd::embed",
                            "shutdown received; draining embed queue"
                        );
                        while let Ok(job) = rx.try_recv() {
                            handle_job(&*embedder, &writer_tx, job).await;
                        }
                        break;
                    }
                }
            }
        }
    })
}

async fn handle_job(embedder: &dyn Embedder, writer_tx: &mpsc::Sender<WriteOp>, job: EmbedJob) {
    let model = embedder.model().to_string();
    let dim = embedder.dim() as i64;

    let (rowid, kind, text) = match &job {
        EmbedJob::Chunk { chunk_id, text } => (*chunk_id, "chunk", text.clone()),
        EmbedJob::Memory { memory_id, text } => (*memory_id, "memory", text.clone()),
    };

    let vec = match embedder.embed(text).await {
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
mod tests {
    use super::*;
    use anyhow::Result;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Mock that returns a fixed vector and counts calls.
    struct MockEmbedder {
        calls: AtomicUsize,
        vec: Vec<f32>,
    }

    #[async_trait]
    impl Embedder for MockEmbedder {
        async fn embed(&self, _text: String) -> Result<Vec<f32>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(self.vec.clone())
        }
        fn model(&self) -> &str {
            "mock"
        }
        fn dim(&self) -> usize {
            self.vec.len()
        }
    }

    #[tokio::test]
    async fn worker_routes_chunk_job_to_storechunkembedding() {
        let embedder = Arc::new(MockEmbedder {
            calls: AtomicUsize::new(0),
            vec: vec![1.0, 0.0],
        });
        let (write_tx, mut write_rx) = mpsc::channel::<WriteOp>(8);
        let (job_tx, job_rx) = mpsc::channel::<EmbedJob>(8);
        let (sd_tx, sd_rx) = watch::channel(false);
        let task = spawn_embedder_task(embedder.clone(), Arc::new(write_tx), job_rx, sd_rx);

        job_tx
            .send(EmbedJob::Chunk {
                chunk_id: 42,
                text: "hello".into(),
            })
            .await
            .unwrap();

        match write_rx.recv().await.unwrap() {
            WriteOp::StoreChunkEmbedding {
                chunk_id,
                model,
                dim,
                vector,
                ack,
            } => {
                assert_eq!(chunk_id, 42);
                assert_eq!(model, "mock");
                assert_eq!(dim, 2);
                assert_eq!(vector.len(), 8); // 2 f32s = 8 bytes
                let _ = ack.send(Ok(()));
            }
            _ => panic!("expected StoreChunkEmbedding"),
        }

        // Shutdown so the task exits cleanly.
        sd_tx.send(true).unwrap();
        drop(job_tx);
        let _ = tokio::time::timeout(std::time::Duration::from_secs(2), task).await;
        assert_eq!(embedder.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn worker_routes_memory_job_to_storememoryembedding() {
        let embedder = Arc::new(MockEmbedder {
            calls: AtomicUsize::new(0),
            vec: vec![0.0, 1.0],
        });
        let (write_tx, mut write_rx) = mpsc::channel::<WriteOp>(8);
        let (job_tx, job_rx) = mpsc::channel::<EmbedJob>(8);
        let (sd_tx, sd_rx) = watch::channel(false);
        let task = spawn_embedder_task(embedder.clone(), Arc::new(write_tx), job_rx, sd_rx);

        job_tx
            .send(EmbedJob::Memory {
                memory_id: 7,
                text: "vim".into(),
            })
            .await
            .unwrap();

        match write_rx.recv().await.unwrap() {
            WriteOp::StoreMemoryEmbedding { memory_id, ack, .. } => {
                assert_eq!(memory_id, 7);
                let _ = ack.send(Ok(()));
            }
            _ => panic!("expected StoreMemoryEmbedding"),
        }
        sd_tx.send(true).unwrap();
        drop(job_tx);
        let _ = tokio::time::timeout(std::time::Duration::from_secs(2), task).await;
    }

    #[tokio::test]
    async fn worker_drains_queue_on_shutdown() {
        let embedder = Arc::new(MockEmbedder {
            calls: AtomicUsize::new(0),
            vec: vec![1.0],
        });
        let (write_tx, mut write_rx) = mpsc::channel::<WriteOp>(8);
        let (job_tx, job_rx) = mpsc::channel::<EmbedJob>(8);
        let (sd_tx, sd_rx) = watch::channel(false);

        // Pre-load 3 jobs before the worker starts.
        for i in 0..3 {
            job_tx
                .send(EmbedJob::Chunk {
                    chunk_id: i,
                    text: format!("t{i}"),
                })
                .await
                .unwrap();
        }
        let task = spawn_embedder_task(embedder.clone(), Arc::new(write_tx), job_rx, sd_rx);
        // Trip shutdown immediately. The drain branch should still
        // process the queued jobs.
        sd_tx.send(true).unwrap();
        drop(job_tx);

        // Collect 3 writes, ack each.
        for _ in 0..3 {
            let op = tokio::time::timeout(std::time::Duration::from_secs(2), write_rx.recv())
                .await
                .expect("ack arrived in time")
                .expect("write_rx open");
            if let WriteOp::StoreChunkEmbedding { ack, .. } = op {
                let _ = ack.send(Ok(()));
            } else {
                panic!("expected StoreChunkEmbedding");
            }
        }
        let _ = tokio::time::timeout(std::time::Duration::from_secs(2), task).await;
        assert_eq!(embedder.calls.load(Ordering::SeqCst), 3);
    }
}
