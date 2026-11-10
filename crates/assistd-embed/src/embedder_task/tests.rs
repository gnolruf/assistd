use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;

use assistd_memory::{
    AdjacentChunks, ChunkId, ChunkLocator, ChunkScope, Embedding, EmbeddingHit, MemoryHit,
    MemoryId, Result as MemoryResult, UnindexedRows,
};

use super::*;

/// Embeds every text as `[1.0]`, except that `"bad"` fails for its own content and
/// every call fails with a server error while `down`. Records the inputs of every call.
#[derive(Debug, Default)]
struct ScriptedEmbedder {
    down: AtomicBool,
    calls: Mutex<Vec<Vec<String>>>,
}

#[async_trait]
impl Embedder for ScriptedEmbedder {
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
        let failure = if self.down.load(Ordering::SeqCst) {
            Some(reqwest::StatusCode::SERVICE_UNAVAILABLE)
        } else if texts.contains(&"bad") {
            Some(reqwest::StatusCode::BAD_REQUEST)
        } else {
            None
        };
        match failure {
            Some(status) => Err(EmbedError::Status {
                status,
                body: String::new(),
            }),
            None => Ok(vec![vec![1.0]; texts.len()]),
        }
    }
    fn model(&self) -> &'static str {
        "scripted"
    }
    fn dim(&self) -> usize {
        1
    }
}

/// Lists the chunks and memories still in `missing`, and takes out each one stored.
#[derive(Debug, Default)]
struct PendingStore {
    missing_chunks: Mutex<BTreeMap<ChunkId, String>>,
    missing_memories: Mutex<BTreeMap<MemoryId, String>>,
    stored: Mutex<Vec<(String, Embedding)>>,
}

fn page_after<Id: Copy + Ord>(
    missing: &Mutex<BTreeMap<Id, String>>,
    after: Option<Id>,
    limit: usize,
) -> Vec<(Id, String)> {
    missing
        .lock()
        .unwrap()
        .iter()
        .filter(|&(&id, _)| after.is_none_or(|after| id > after))
        .take(limit)
        .map(|(&id, text)| (id, text.clone()))
        .collect()
}

#[async_trait]
impl SemanticStore for PendingStore {
    async fn nearest_chunks(
        &self,
        _query_vector: Vec<f32>,
        _top_k: usize,
        _scope: ChunkScope,
    ) -> MemoryResult<Vec<EmbeddingHit>> {
        Ok(Vec::new())
    }
    async fn adjacent_chunks(&self, _locator: &ChunkLocator) -> MemoryResult<AdjacentChunks> {
        Ok(AdjacentChunks::default())
    }
    async fn nearest_memories(
        &self,
        _query_vector: Vec<f32>,
        _top_k: usize,
    ) -> MemoryResult<Vec<MemoryHit>> {
        Ok(Vec::new())
    }
    async fn drop_other_models(&self, _model: &str) -> MemoryResult<u64> {
        Ok(0)
    }
    async fn count_missing(&self) -> MemoryResult<UnindexedRows> {
        Ok(UnindexedRows::default())
    }
    async fn memories_missing_embedding(
        &self,
        after: Option<MemoryId>,
        limit: usize,
    ) -> MemoryResult<Vec<(MemoryId, String)>> {
        Ok(page_after(&self.missing_memories, after, limit))
    }
    async fn chunks_missing_embedding(
        &self,
        after: Option<ChunkId>,
        limit: usize,
    ) -> MemoryResult<Vec<(ChunkId, String)>> {
        Ok(page_after(&self.missing_chunks, after, limit))
    }
    async fn store_embeddings(&self, model: &str, embeddings: Vec<Embedding>) -> MemoryResult<()> {
        for embedding in embeddings {
            match embedding.target {
                EmbeddingTarget::Chunk(id) => self.missing_chunks.lock().unwrap().remove(&id),
                EmbeddingTarget::Memory(id) => self.missing_memories.lock().unwrap().remove(&id),
            };
            self.stored
                .lock()
                .unwrap()
                .push((model.to_owned(), embedding));
        }
        Ok(())
    }
    async fn clear_unembeddable(&self) -> MemoryResult<u64> {
        Ok(0)
    }
}

struct Harness {
    embedder: Arc<ScriptedEmbedder>,
    store: Arc<PendingStore>,
    wake: Arc<Notify>,
    shutdown: watch::Sender<bool>,
    task: JoinHandle<()>,
}

impl Harness {
    /// Start the worker over a store already missing embeddings for `chunks`.
    fn spawn(chunks: &[&str], embedder_down: bool) -> Self {
        let embedder = Arc::new(ScriptedEmbedder::default());
        embedder.down.store(embedder_down, Ordering::SeqCst);
        let store = Arc::new(PendingStore::default());
        let wake = Arc::new(Notify::new());
        let (shutdown, shutdown_rx) = watch::channel(false);
        let harness = Self {
            task: spawn_embedder_task(embedder.clone(), store.clone(), wake.clone(), shutdown_rx),
            embedder,
            store,
            wake,
            shutdown,
        };
        harness.add_chunks(chunks);
        harness
    }

    fn add_chunks(&self, texts: &[&str]) {
        let mut missing = self.store.missing_chunks.lock().unwrap();
        let first_id = missing.last_key_value().map_or(0, |(id, _)| id.0 + 1);
        missing.extend(
            (first_id..)
                .zip(texts)
                .map(|(id, text)| (ChunkId(id), (*text).to_owned())),
        );
    }

    fn missing_chunks(&self) -> usize {
        self.store.missing_chunks.lock().unwrap().len()
    }

    /// Give the worker `wait` of paused time, then the vector stored for each chunk id.
    async fn stored_after(&self, wait: Duration) -> Vec<(i64, Option<Vec<f32>>)> {
        tokio::time::sleep(wait).await;
        self.store
            .stored
            .lock()
            .unwrap()
            .iter()
            .map(|(_, embedding)| match embedding.target {
                EmbeddingTarget::Chunk(id) => (id.0, embedding.vector.clone()),
                EmbeddingTarget::Memory(id) => panic!("unexpected memory {id}"),
            })
            .collect()
    }

    /// Signal shutdown and return the inputs of every embed call.
    async fn shut_down(self) -> Vec<Vec<String>> {
        self.shutdown.send_replace(true);
        tokio::time::timeout(Duration::from_secs(2), self.task)
            .await
            .expect("worker exited after shutdown")
            .expect("worker did not panic");
        std::mem::take(&mut *self.embedder.calls.lock().unwrap())
    }
}

#[tokio::test(start_paused = true)]
async fn worker_embeds_rows_already_missing_and_those_it_is_woken_for() {
    let backlog: Vec<String> = (0..BATCH_SIZE + 1).map(|n| format!("t{n}")).collect();
    let backlog: Vec<&str> = backlog.iter().map(String::as_str).collect();
    let harness = Harness::spawn(&backlog, false);
    harness
        .store
        .missing_memories
        .lock()
        .unwrap()
        .insert(MemoryId(7), "fact".to_owned());

    tokio::time::sleep(Duration::from_millis(10)).await;
    assert_eq!(harness.missing_chunks(), 0);
    assert!(harness.store.missing_memories.lock().unwrap().is_empty());
    assert!(
        harness
            .store
            .stored
            .lock()
            .unwrap()
            .iter()
            .all(|(model, embedding)| model == "scripted" && embedding.vector.is_some())
    );

    harness.add_chunks(&["late"]);
    tokio::time::sleep(Duration::from_secs(600)).await;
    assert_eq!(harness.missing_chunks(), 1, "nothing woke the worker");
    harness.wake.notify_one();
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert_eq!(harness.missing_chunks(), 0);

    let calls = harness.shut_down().await;
    let batch_sizes: Vec<usize> = calls.iter().map(Vec::len).collect();
    assert_eq!(batch_sizes, [BATCH_SIZE, 1, 1, 1]);
}

#[tokio::test(start_paused = true)]
async fn worker_retries_rows_left_by_a_server_failure_without_being_woken() {
    let harness = Harness::spawn(&["a", "b"], true);

    assert_eq!(harness.stored_after(Duration::from_millis(10)).await, []);
    assert_eq!(harness.missing_chunks(), 2);

    harness.embedder.down.store(false, Ordering::SeqCst);
    assert_eq!(
        harness.stored_after(Duration::from_secs(120)).await,
        [(0, Some(vec![1.0])), (1, Some(vec![1.0]))]
    );
    harness.shut_down().await;
}

#[tokio::test(start_paused = true)]
async fn worker_marks_a_row_its_content_fails_and_does_not_retry_it() {
    let harness = Harness::spawn(&["a", "bad", "c"], false);

    assert_eq!(
        harness.stored_after(Duration::from_secs(600)).await,
        [(0, Some(vec![1.0])), (1, None), (2, Some(vec![1.0]))]
    );
    assert_eq!(
        harness.shut_down().await,
        [vec!["a", "bad", "c"], vec!["a"], vec!["bad"], vec!["c"]]
    );
}
