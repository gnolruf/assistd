use std::sync::Mutex;
use std::time::Duration;

use async_trait::async_trait;

use assistd_memory::{
    ChunkId, Embedding, EmbeddingHit, MemoryHit, MemoryId, Result as MemoryResult, SessionId,
    StaleEmbeddings, UnindexedRows,
};

use super::*;
use crate::EmbedError;

/// Returns `embedding` for every input and records the inputs of every call.
#[derive(Debug)]
struct MockEmbedder {
    calls: Mutex<Vec<Vec<String>>>,
    embedding: Vec<f32>,
}

#[async_trait]
impl Embedder for MockEmbedder {
    async fn embed(&self, text: String) -> Result<Vec<f32>, EmbedError> {
        self.embed_batch(&[text.as_str()])
            .await?
            .pop()
            .ok_or(EmbedError::Disabled)
    }
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        self.calls
            .lock()
            .unwrap()
            .push(texts.iter().map(|&t| t.to_owned()).collect());
        Ok(vec![self.embedding.clone(); texts.len()])
    }
    fn model(&self) -> &'static str {
        "mock"
    }
    fn dim(&self) -> usize {
        self.embedding.len()
    }
}

/// Forwards every `store_embeddings` call to the test; nothing else is used by the worker.
#[derive(Debug)]
struct RecordingStore {
    stored: mpsc::UnboundedSender<(String, Vec<Embedding>)>,
}

#[async_trait]
impl SemanticStore for RecordingStore {
    async fn nearest_chunks(
        &self,
        _query_vector: Vec<f32>,
        _top_k: usize,
        _model: &str,
        _exclude_session: Option<&SessionId>,
    ) -> MemoryResult<Vec<EmbeddingHit>> {
        Ok(Vec::new())
    }
    async fn nearest_memories(
        &self,
        _query_vector: Vec<f32>,
        _top_k: usize,
        _model: &str,
    ) -> MemoryResult<Vec<MemoryHit>> {
        Ok(Vec::new())
    }
    async fn count_stale(&self, _current: &str) -> MemoryResult<StaleEmbeddings> {
        Ok(StaleEmbeddings::default())
    }
    async fn count_missing(&self, _current: &str) -> MemoryResult<UnindexedRows> {
        Ok(UnindexedRows::default())
    }
    async fn memories_missing_embedding(
        &self,
        _current: &str,
        _after: Option<MemoryId>,
        _limit: usize,
    ) -> MemoryResult<Vec<(MemoryId, String)>> {
        Ok(Vec::new())
    }
    async fn chunks_missing_embedding(
        &self,
        _current: &str,
        _after: Option<ChunkId>,
        _limit: usize,
    ) -> MemoryResult<Vec<(ChunkId, String)>> {
        Ok(Vec::new())
    }
    async fn store_embeddings(&self, model: &str, embeddings: Vec<Embedding>) -> MemoryResult<()> {
        self.stored.send((model.to_owned(), embeddings)).unwrap();
        Ok(())
    }
}

struct Harness {
    embedder: Arc<MockEmbedder>,
    jobs: mpsc::Sender<EmbedJob>,
    stored: mpsc::UnboundedReceiver<(String, Vec<Embedding>)>,
    shutdown: watch::Sender<bool>,
    task: JoinHandle<()>,
}

impl Harness {
    fn spawn(embedding: Vec<f32>) -> Self {
        let embedder = Arc::new(MockEmbedder {
            calls: Mutex::new(Vec::new()),
            embedding,
        });
        let (stored_tx, stored) = mpsc::unbounded_channel();
        let store = Arc::new(RecordingStore { stored: stored_tx });
        let (jobs, job_rx) = mpsc::channel(8);
        let (shutdown, shutdown_rx) = watch::channel(false);
        let task = spawn_embedder_task(embedder.clone(), store, job_rx, shutdown_rx);
        Self {
            embedder,
            jobs,
            stored,
            shutdown,
            task,
        }
    }

    async fn send_chunks(&self, texts: &[&str]) {
        for (chunk_id, text) in (0..).zip(texts) {
            self.jobs.send(chunk_job(chunk_id, text)).await.unwrap();
        }
    }

    /// The targets of every embedding stored until `count` have arrived.
    async fn stored_targets(&mut self, count: usize) -> Vec<EmbeddingTarget> {
        let mut targets = Vec::new();
        while targets.len() < count {
            let (_, embeddings) = tokio::time::timeout(Duration::from_secs(2), self.stored.recv())
                .await
                .expect("embeddings stored in time")
                .expect("store still open");
            targets.extend(embeddings.iter().map(|embedding| embedding.target));
        }
        targets
    }

    /// Signal shutdown with the job sender still alive, so only the shutdown path can end
    /// the worker. Returns the inputs of every embed call.
    async fn shut_down(mut self) -> Vec<Vec<String>> {
        self.shutdown.send_replace(true);
        tokio::time::timeout(Duration::from_secs(2), self.task)
            .await
            .expect("worker exited after shutdown")
            .expect("worker did not panic");
        assert!(self.stored.try_recv().is_err(), "unexpected extra store");
        std::mem::take(&mut *self.embedder.calls.lock().unwrap())
    }
}

fn chunk_job(chunk_id: i64, text: &str) -> EmbedJob {
    EmbedJob {
        target: EmbeddingTarget::Chunk(ChunkId(chunk_id)),
        text: text.to_owned(),
    }
}

fn chunk_targets(ids: &[i64]) -> Vec<EmbeddingTarget> {
    ids.iter()
        .map(|&id| EmbeddingTarget::Chunk(ChunkId(id)))
        .collect()
}

#[tokio::test]
async fn worker_stores_embedded_job_under_the_embedder_model() {
    let mut harness = Harness::spawn(vec![1.0, 0.0]);
    harness.jobs.send(chunk_job(42, "hello")).await.unwrap();

    let (model, embeddings) = harness.stored.recv().await.unwrap();
    assert_eq!(model, "mock");
    assert_eq!(
        embeddings,
        [Embedding {
            target: EmbeddingTarget::Chunk(ChunkId(42)),
            vector: vec![1.0, 0.0],
        }]
    );
    assert_eq!(harness.shut_down().await, vec![vec!["hello"]]);
}

#[tokio::test]
async fn worker_coalesces_queued_jobs_into_one_embed_call() {
    let mut harness = Harness::spawn(vec![1.0]);
    harness.send_chunks(&["t0", "t1", "t2"]).await;

    assert_eq!(harness.stored_targets(3).await, chunk_targets(&[0, 1, 2]));
    assert_eq!(harness.shut_down().await, vec![vec!["t0", "t1", "t2"]]);
}

#[tokio::test]
async fn worker_embeds_queued_jobs_before_honouring_shutdown() {
    let mut harness = Harness::spawn(vec![1.0]);
    harness.send_chunks(&["t0", "t1", "t2"]).await;
    harness.shutdown.send(true).unwrap();

    assert_eq!(harness.stored_targets(3).await, chunk_targets(&[0, 1, 2]));
    assert_eq!(harness.shut_down().await.concat(), vec!["t0", "t1", "t2"]);
}
