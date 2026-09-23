//! Background worker that embeds queued rows and stores the vectors
//! through the memory writer. A failed embed is logged and dropped,
//! never retried; the row stays unindexed until a reindex.

use std::sync::Arc;

use assistd_memory::{WriteOp, vector_to_blob};
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::JoinHandle;

use crate::Embedder;

/// One row to embed.
#[derive(Debug)]
pub enum EmbedJob {
    Chunk { chunk_id: i64, text: String },
    Memory { memory_id: i64, text: String },
}

/// Spawn the worker. The caller awaits the returned handle on shutdown
/// so in-flight embeddings land before the memory writer drains.
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
    use crate::EmbedError;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    struct MockEmbedder {
        calls: AtomicUsize,
        vec: Vec<f32>,
    }

    #[async_trait]
    impl Embedder for MockEmbedder {
        async fn embed(&self, _text: String) -> Result<Vec<f32>, EmbedError> {
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

    struct Harness {
        embedder: Arc<MockEmbedder>,
        jobs: mpsc::Sender<EmbedJob>,
        writes: mpsc::Receiver<WriteOp>,
        shutdown: watch::Sender<bool>,
        task: JoinHandle<()>,
    }

    impl Harness {
        fn spawn(vec: Vec<f32>) -> Self {
            let embedder = Arc::new(MockEmbedder {
                calls: AtomicUsize::new(0),
                vec,
            });
            let (write_tx, writes) = mpsc::channel(8);
            let (jobs, job_rx) = mpsc::channel(8);
            let (shutdown, sd_rx) = watch::channel(false);
            let task = spawn_embedder_task(embedder.clone(), Arc::new(write_tx), job_rx, sd_rx);
            Self {
                embedder,
                jobs,
                writes,
                shutdown,
                task,
            }
        }

        async fn next_write(&mut self) -> WriteOp {
            tokio::time::timeout(Duration::from_secs(2), self.writes.recv())
                .await
                .expect("write op arrived in time")
                .expect("writer channel open")
        }

        /// Signal shutdown while keeping the job sender alive, so only
        /// the shutdown path can end the worker.
        async fn shut_down(self) -> usize {
            self.shutdown.send_replace(true);
            tokio::time::timeout(Duration::from_secs(2), self.task)
                .await
                .expect("worker exited after shutdown")
                .expect("worker did not panic");
            self.embedder.calls.load(Ordering::SeqCst)
        }
    }

    #[tokio::test]
    async fn worker_routes_chunk_job_to_storechunkembedding() {
        let mut h = Harness::spawn(vec![1.0, 0.0]);
        h.jobs
            .send(EmbedJob::Chunk {
                chunk_id: 42,
                text: "hello".into(),
            })
            .await
            .unwrap();

        match h.next_write().await {
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
                assert_eq!(vector, vector_to_blob(&[1.0, 0.0]));
                ack.send(Ok(())).unwrap();
            }
            _ => panic!("expected StoreChunkEmbedding"),
        }
        assert_eq!(h.shut_down().await, 1);
    }

    #[tokio::test]
    async fn worker_routes_memory_job_to_storememoryembedding() {
        let mut h = Harness::spawn(vec![0.0, 1.0]);
        h.jobs
            .send(EmbedJob::Memory {
                memory_id: 7,
                text: "vim".into(),
            })
            .await
            .unwrap();

        match h.next_write().await {
            WriteOp::StoreMemoryEmbedding {
                memory_id,
                model,
                dim,
                vector,
                ack,
            } => {
                assert_eq!(memory_id, 7);
                assert_eq!(model, "mock");
                assert_eq!(dim, 2);
                assert_eq!(vector, vector_to_blob(&[0.0, 1.0]));
                ack.send(Ok(())).unwrap();
            }
            _ => panic!("expected StoreMemoryEmbedding"),
        }
        assert_eq!(h.shut_down().await, 1);
    }

    #[tokio::test]
    async fn worker_embeds_queued_jobs_before_honouring_shutdown() {
        let mut h = Harness::spawn(vec![1.0]);
        for i in 0..3 {
            h.jobs
                .send(EmbedJob::Chunk {
                    chunk_id: i,
                    text: format!("t{i}"),
                })
                .await
                .unwrap();
        }
        h.shutdown.send(true).unwrap();

        for i in 0..3 {
            match h.next_write().await {
                WriteOp::StoreChunkEmbedding { chunk_id, ack, .. } => {
                    assert_eq!(chunk_id, i);
                    ack.send(Ok(())).unwrap();
                }
                _ => panic!("expected StoreChunkEmbedding"),
            }
        }
        assert_eq!(h.shut_down().await, 3);
    }
}
