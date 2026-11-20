//! Background worker that embeds the rows the semantic store lists as missing an
//! embedding, a batch at a time, each time it is woken: message chunks and memories
//! at once, document chunks only while a gate is open.

use std::future::{Future, pending};
use std::sync::Arc;

use assistd_memory::{ChunkSource, EmbeddingTarget, MemoryError, SemanticStore};
use assistd_utils::backoff::backoff_delay;
use thiserror::Error;
use tokio::sync::{Notify, watch};
use tokio::task::JoinHandle;

use crate::{BATCH_SIZE, EmbedError, Embedder, embed_rows};

/// Why a pass over the missing rows stopped before reaching the end.
#[derive(Debug, Error)]
enum PassError {
    #[error(transparent)]
    Embed(#[from] EmbedError),
    #[error(transparent)]
    Store(#[from] MemoryError),
}

/// How a pass over the missing rows ended without failing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PassEnd {
    /// No row is left to embed, or shutdown was signalled.
    Finished,
    /// The worker was woken while document chunks were still left or held back.
    Woken,
}

/// What the worker embeds with and into, and the signals it runs on.
struct EmbedWorker {
    embedder: Arc<dyn Embedder>,
    store: Arc<dyn SemanticStore>,
    wake: Arc<Notify>,
    documents_gate: watch::Receiver<bool>,
    shutdown: watch::Receiver<bool>,
}

impl EmbedWorker {
    async fn run(mut self) {
        let mut failed_passes = 0u32;
        loop {
            let retry_in = match self.embed_missing_rows().await {
                Ok(PassEnd::Woken) => {
                    failed_passes = 0;
                    continue;
                }
                Ok(PassEnd::Finished) => {
                    failed_passes = 0;
                    None
                }
                Err(err) => {
                    let delay = backoff_delay(failed_passes);
                    log_stopped_pass(&err, failed_passes);
                    failed_passes = failed_passes.saturating_add(1);
                    Some(delay)
                }
            };
            tokio::select! {
                biased;
                _ = self.shutdown.wait_for(|stop| *stop) => break,
                () = self.wake.notified() => {}
                () = tokio::time::sleep(retry_in.unwrap_or_default()), if retry_in.is_some() => {}
            }
        }
        tracing::debug!(target: "assistd::embed", "embed worker exiting");
    }

    /// Embed and store every message chunk, then every memory, then the document
    /// chunks, missing an embedding.
    async fn embed_missing_rows(&mut self) -> Result<PassEnd, PassError> {
        let store = &*self.store;
        embed_missing_of_kind(
            &*self.embedder,
            store,
            &self.shutdown,
            |after, limit| store.chunks_missing_embedding(ChunkSource::Message, after, limit),
            EmbeddingTarget::Chunk,
        )
        .await?;
        embed_missing_of_kind(
            &*self.embedder,
            store,
            &self.shutdown,
            |after, limit| store.memories_missing_embedding(after, limit),
            EmbeddingTarget::Memory,
        )
        .await?;
        self.embed_missing_document_chunks().await
    }

    /// Page through the document chunks missing an embedding in id order, waiting
    /// before each page for the gate to be open, until they run out, the worker is
    /// woken, shutdown flips, or a page has rows left for a later attempt.
    async fn embed_missing_document_chunks(&mut self) -> Result<PassEnd, PassError> {
        let mut after = None;
        loop {
            tokio::select! {
                biased;
                _ = self.shutdown.wait_for(|stop| *stop) => return Ok(PassEnd::Finished),
                () = self.wake.notified() => return Ok(PassEnd::Woken),
                () = gate_open(&mut self.documents_gate) => {}
            }
            let page = self
                .store
                .chunks_missing_embedding(ChunkSource::Document, after, BATCH_SIZE)
                .await?;
            let Some(&(last_id, _)) = page.last() else {
                return Ok(PassEnd::Finished);
            };
            after = Some(last_id);
            embed_page(&*self.embedder, &*self.store, page, EmbeddingTarget::Chunk).await?;
        }
    }
}

/// Spawn the worker. It embeds the rows missing an embedding at once and again after each
/// `wake`, document chunks last and only while `documents_gate` reads `true`, retrying a failed
/// pass with backoff. It exits on `shutdown` between batches; `store`'s database must outlive it.
pub fn spawn_embedder_task(
    embedder: Arc<dyn Embedder>,
    store: Arc<dyn SemanticStore>,
    wake: Arc<Notify>,
    documents_gate: watch::Receiver<bool>,
    shutdown: watch::Receiver<bool>,
) -> JoinHandle<()> {
    let worker = EmbedWorker {
        embedder,
        store,
        wake,
        documents_gate,
        shutdown,
    };
    tokio::spawn(worker.run())
}

fn log_stopped_pass(err: &PassError, earlier_failures: u32) {
    if earlier_failures == 0 {
        tracing::warn!(
            target: "assistd::embed",
            error = %err,
            "embedding pass stopped early; rows stay unindexed until a retry succeeds"
        );
    } else {
        tracing::debug!(
            target: "assistd::embed",
            error = %err,
            earlier_failures,
            "embedding pass stopped early again"
        );
    }
}

/// Wait until `gate` reads `true`; never returns once it can no longer do so.
async fn gate_open(gate: &mut watch::Receiver<bool>) {
    if gate.wait_for(|open| *open).await.is_err() {
        pending::<()>().await;
    }
}

/// Page through `list` in id order, embedding and storing each page, until it runs
/// out, `shutdown` flips, or a page has rows left for a later attempt.
async fn embed_missing_of_kind<Id, L, LFut>(
    embedder: &dyn Embedder,
    store: &dyn SemanticStore,
    shutdown: &watch::Receiver<bool>,
    list: L,
    to_target: fn(Id) -> EmbeddingTarget,
) -> Result<(), PassError>
where
    Id: Copy,
    L: Fn(Option<Id>, usize) -> LFut,
    LFut: Future<Output = Result<Vec<(Id, String)>, MemoryError>>,
{
    let mut after = None;
    while !*shutdown.borrow() {
        let page = list(after, BATCH_SIZE).await?;
        let Some(&(last_id, _)) = page.last() else {
            break;
        };
        after = Some(last_id);
        embed_page(embedder, store, page, to_target).await?;
    }
    Ok(())
}

/// Embed `page` in one batch and store what came back; fails when rows of it are
/// left for a later attempt.
async fn embed_page<Id>(
    embedder: &dyn Embedder,
    store: &dyn SemanticStore,
    page: Vec<(Id, String)>,
    to_target: fn(Id) -> EmbeddingTarget,
) -> Result<(), PassError> {
    let rows = page
        .into_iter()
        .map(|(id, text)| (to_target(id), text))
        .collect();
    let embedded = embed_rows(embedder, rows).await;
    store.store_embeddings(embedded.embeddings).await?;
    embedded.deferred.map_or(Ok(()), |err| Err(err.into()))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::ops::Range;
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    use async_trait::async_trait;

    use assistd_memory::{
        AdjacentChunks, ChunkId, ChunkLocator, DocumentHit, DocumentText, Embedding, EmbeddingHit,
        EmbeddingSpace, EmptyPages, FileStamp, IndexPass, IndexRootCounts, MemoryHit, MemoryId,
        NoSemanticStore, PageReading, Result as MemoryResult, SessionScope, TranscriptionCounts,
        UnindexedRows, WaitingPage,
    };

    /// The id of the first document chunk a [`Harness`] adds, above every message chunk's.
    const FIRST_DOCUMENT_CHUNK: i64 = 1_000_000;

    use super::*;

    /// Embeds every text as `[1.0]` after `batch_time`, except that `"bad"` fails for its
    /// own content and every call fails with a server error while `down`. Records the
    /// inputs of every call.
    #[derive(Debug, Default)]
    struct ScriptedEmbedder {
        down: AtomicBool,
        batch_time: Duration,
        calls: Mutex<Vec<Vec<String>>>,
    }

    #[async_trait]
    impl Embedder for ScriptedEmbedder {
        async fn embed_query(&self, query: &str) -> Result<Vec<f32>, EmbedError> {
            self.embed_documents(&[query])
                .await?
                .pop()
                .ok_or(EmbedError::CountMismatch {
                    got: 0,
                    expected: 1,
                })
        }
        async fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
            self.calls
                .lock()
                .unwrap()
                .push(texts.iter().map(|&t| t.to_owned()).collect());
            tokio::time::sleep(self.batch_time).await;
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
        missing_document_chunks: Mutex<BTreeMap<ChunkId, String>>,
        missing_memories: Mutex<BTreeMap<MemoryId, String>>,
        stored: Mutex<Vec<Embedding>>,
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
            _scope: SessionScope,
        ) -> MemoryResult<Vec<EmbeddingHit>> {
            Ok(Vec::new())
        }
        async fn search_documents(
            &self,
            _query: String,
            _query_vector: Vec<f32>,
            _top_k: usize,
        ) -> MemoryResult<Vec<DocumentHit>> {
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
        async fn adopt_embedding_space(
            &self,
            _space: &EmbeddingSpace,
        ) -> MemoryResult<Option<EmbeddingSpace>> {
            Ok(None)
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
            source: ChunkSource,
            after: Option<ChunkId>,
            limit: usize,
        ) -> MemoryResult<Vec<(ChunkId, String)>> {
            let missing = match source {
                ChunkSource::Message => &self.missing_chunks,
                ChunkSource::Document => &self.missing_document_chunks,
            };
            Ok(page_after(missing, after, limit))
        }
        async fn store_embeddings(&self, embeddings: Vec<Embedding>) -> MemoryResult<()> {
            for embedding in embeddings {
                match embedding.target {
                    EmbeddingTarget::Chunk(id) if id.0 < FIRST_DOCUMENT_CHUNK => {
                        self.missing_chunks.lock().unwrap().remove(&id)
                    }
                    EmbeddingTarget::Chunk(id) => {
                        self.missing_document_chunks.lock().unwrap().remove(&id)
                    }
                    EmbeddingTarget::Memory(id) => {
                        self.missing_memories.lock().unwrap().remove(&id)
                    }
                };
                self.stored.lock().unwrap().push(embedding);
            }
            Ok(())
        }
        async fn clear_unembeddable(&self) -> MemoryResult<u64> {
            Ok(0)
        }
        async fn begin_index_pass(&self, root: &Path) -> MemoryResult<IndexPass> {
            NoSemanticStore.begin_index_pass(root).await
        }
        async fn changed_documents(
            &self,
            pass: IndexPass,
            files: Vec<(PathBuf, FileStamp)>,
        ) -> MemoryResult<Vec<(PathBuf, FileStamp)>> {
            NoSemanticStore.changed_documents(pass, files).await
        }
        async fn index_document(
            &self,
            pass: IndexPass,
            path: PathBuf,
            stamp: FileStamp,
            pages: Vec<String>,
            empty_pages: EmptyPages,
        ) -> MemoryResult<Vec<ChunkId>> {
            NoSemanticStore
                .index_document(pass, path, stamp, pages, empty_pages)
                .await
        }
        async fn next_waiting_page(&self) -> MemoryResult<Option<WaitingPage>> {
            NoSemanticStore.next_waiting_page().await
        }
        async fn store_page_reading(
            &self,
            page: WaitingPage,
            reading: PageReading,
        ) -> MemoryResult<Vec<ChunkId>> {
            NoSemanticStore.store_page_reading(page, reading).await
        }
        async fn count_transcriptions(&self) -> MemoryResult<TranscriptionCounts> {
            NoSemanticStore.count_transcriptions().await
        }
        async fn retry_failed_transcriptions(&self) -> MemoryResult<u64> {
            NoSemanticStore.retry_failed_transcriptions().await
        }
        async fn finish_index_pass(&self, pass: IndexPass) -> MemoryResult<u64> {
            NoSemanticStore.finish_index_pass(pass).await
        }
        async fn finish_index_pass_under(
            &self,
            pass: IndexPass,
            paths: Vec<PathBuf>,
        ) -> MemoryResult<u64> {
            NoSemanticStore.finish_index_pass_under(pass, paths).await
        }
        async fn remove_document(&self, path: &Path) -> MemoryResult<bool> {
            NoSemanticStore.remove_document(path).await
        }
        async fn index_roots(&self) -> MemoryResult<Vec<PathBuf>> {
            NoSemanticStore.index_roots().await
        }
        async fn index_root_counts(&self) -> MemoryResult<Vec<IndexRootCounts>> {
            NoSemanticStore.index_root_counts().await
        }
        async fn remove_index_root(&self, root: &Path) -> MemoryResult<u64> {
            NoSemanticStore.remove_index_root(root).await
        }
        async fn count_document_chunks(&self) -> MemoryResult<u64> {
            NoSemanticStore.count_document_chunks().await
        }
        async fn document_text(
            &self,
            path: &Path,
            page: usize,
            bytes: Range<usize>,
        ) -> MemoryResult<Option<DocumentText>> {
            NoSemanticStore.document_text(path, page, bytes).await
        }
    }

    struct Harness {
        embedder: Arc<ScriptedEmbedder>,
        store: Arc<PendingStore>,
        wake: Arc<Notify>,
        documents_gate: watch::Sender<bool>,
        shutdown: watch::Sender<bool>,
        task: JoinHandle<()>,
    }

    impl Harness {
        /// Start the worker over a store already missing embeddings for `chunks`.
        fn spawn(chunks: &[&str], embedder_down: bool) -> Self {
            let embedder = ScriptedEmbedder::default();
            embedder.down.store(embedder_down, Ordering::SeqCst);
            let harness = Self::spawn_with(embedder);
            harness.add_chunks(chunks);
            harness
        }

        /// Start the worker on `embedder` over an empty store, the documents gate open.
        fn spawn_with(embedder: ScriptedEmbedder) -> Self {
            let embedder = Arc::new(embedder);
            let store = Arc::new(PendingStore::default());
            let wake = Arc::new(Notify::new());
            let (documents_gate, documents_gate_rx) = watch::channel(true);
            let (shutdown, shutdown_rx) = watch::channel(false);
            Self {
                task: spawn_embedder_task(
                    embedder.clone(),
                    store.clone(),
                    wake.clone(),
                    documents_gate_rx,
                    shutdown_rx,
                ),
                embedder,
                store,
                wake,
                documents_gate,
                shutdown,
            }
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

        fn add_document_chunks(&self, count: usize) {
            let mut missing = self.store.missing_document_chunks.lock().unwrap();
            let first_id = missing
                .last_key_value()
                .map_or(FIRST_DOCUMENT_CHUNK, |(id, _)| id.0 + 1);
            missing.extend(
                (first_id..)
                    .take(count)
                    .map(|id| (ChunkId(id), format!("d{id}"))),
            );
        }

        fn missing_chunks(&self) -> usize {
            self.store.missing_chunks.lock().unwrap().len()
        }

        fn missing_document_chunks(&self) -> usize {
            self.store.missing_document_chunks.lock().unwrap().len()
        }

        /// Give the worker `wait` of paused time, then the vector stored for each chunk id.
        async fn stored_after(&self, wait: Duration) -> Vec<(i64, Option<Vec<f32>>)> {
            tokio::time::sleep(wait).await;
            self.store
                .stored
                .lock()
                .unwrap()
                .iter()
                .map(|embedding| match embedding.target {
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
                .all(|embedding| embedding.vector.is_some())
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

    #[tokio::test(start_paused = true)]
    async fn worker_embeds_message_chunks_but_no_document_chunk_while_the_gate_is_closed() {
        let harness = Harness::spawn(&["said"], false);
        harness.documents_gate.send_replace(false);
        harness.add_document_chunks(1);

        tokio::time::sleep(Duration::from_secs(600)).await;
        assert_eq!(harness.missing_chunks(), 0);
        assert_eq!(harness.missing_document_chunks(), 1);

        harness.add_chunks(&["said later"]);
        harness.wake.notify_one();
        tokio::time::sleep(Duration::from_secs(600)).await;
        assert_eq!(harness.missing_chunks(), 0);
        assert_eq!(harness.missing_document_chunks(), 1);

        assert_eq!(
            harness.shut_down().await,
            [vec!["said"], vec!["said later"]]
        );
    }

    #[tokio::test(start_paused = true)]
    async fn worker_stops_after_the_batch_in_flight_when_the_gate_closes_and_resumes_when_it_opens()
    {
        let harness = Harness::spawn_with(ScriptedEmbedder {
            batch_time: Duration::from_secs(1),
            ..ScriptedEmbedder::default()
        });
        harness.add_document_chunks(2 * BATCH_SIZE + 1);

        tokio::time::sleep(Duration::from_millis(1500)).await;
        assert_eq!(harness.missing_document_chunks(), BATCH_SIZE + 1);
        harness.documents_gate.send_replace(false);
        tokio::time::sleep(Duration::from_secs(600)).await;
        assert_eq!(
            harness.missing_document_chunks(),
            1,
            "the batch in flight was dropped, or another was sent"
        );

        harness.add_chunks(&["said"]);
        harness.wake.notify_one();
        harness.documents_gate.send_replace(true);
        tokio::time::sleep(Duration::from_secs(600)).await;
        assert_eq!(harness.missing_document_chunks(), 0);

        let calls = harness.shut_down().await;
        assert_eq!(calls.len(), 4);
        assert_eq!(
            calls[2],
            ["said"],
            "a message chunk waited behind documents"
        );
    }
}
