//! Background worker that embeds the rows the semantic store lists as missing an
//! embedding, a batch at a time, each time it is woken: message chunks and memories
//! at once, document chunks only while a gate is open.

use std::collections::HashMap;
use std::sync::Arc;

use assistd_memory::{
    ChunkId, ChunkSource, Embedding, EmbeddingQueue, EmbeddingTarget, MemoryError, MemoryId,
};
use assistd_utils::backlog::{Backlog, DrainEnd, Pass, Signals, drain, keep_drained};
use assistd_utils::backoff::backoff_delay;
use thiserror::Error;
use tokio::sync::watch;
use tokio::task::JoinHandle;

use crate::{BATCH_SIZE, EmbedError, Embedder};

/// Times a row may fail alone, the server answering other inputs, before it is
/// marked as refused.
const FAILURES_BEFORE_REFUSAL: u32 = 3;

/// The text a server is asked to embed to show that it still works.
const SERVER_CHECK_TEXT: &str = "ping";

/// Why a pass over the missing rows stopped before reaching the end.
#[derive(Debug, Error)]
enum PassError {
    #[error(transparent)]
    Embed(#[from] EmbedError),
    #[error(transparent)]
    Store(#[from] MemoryError),
}

/// The rows one [`EmbeddingBacklog`] lists.
#[derive(Debug, Clone, Copy)]
enum RowKind {
    MessageChunks,
    Memories,
    DocumentChunks,
}

impl RowKind {
    fn target(self, id: i64) -> EmbeddingTarget {
        match self {
            Self::MessageChunks | Self::DocumentChunks => EmbeddingTarget::Chunk(ChunkId(id)),
            Self::Memories => EmbeddingTarget::Memory(MemoryId(id)),
        }
    }
}

/// The rows of one kind missing an embedding, keyed by row id.
#[derive(Debug)]
struct EmbeddingBacklog<'a> {
    kind: RowKind,
    embedder: &'a dyn Embedder,
    store: &'a dyn EmbeddingQueue,
    permit: Option<watch::Receiver<bool>>,
    /// Times each row still missing has failed alone while the server worked.
    failures: HashMap<i64, u32>,
}

impl EmbeddingBacklog<'_> {
    /// What to store for row `id` after `outcome`, or `None` to leave it missing.
    fn embedding_to_store(
        &mut self,
        id: i64,
        text: String,
        outcome: InputOutcome,
    ) -> Option<Embedding> {
        let target = self.kind.target(id);
        let vector = match outcome {
            InputOutcome::Embedded(vector) => Some(vector),
            InputOutcome::Refused(err) => {
                log_refused_row(target, &err);
                None
            }
            InputOutcome::FailedAlone(err) => {
                let failures = self.failures.entry(id).or_default();
                *failures += 1;
                if *failures < FAILURES_BEFORE_REFUSAL {
                    tracing::debug!(
                        target: "assistd::embed",
                        row = %target,
                        error = %err,
                        failures = *failures,
                        "row failed alone while the server answered; leaving it for a retry"
                    );
                    return None;
                }
                log_refused_row(target, &err);
                None
            }
            InputOutcome::Unsettled => return None,
        };
        self.failures.remove(&id);
        Some(Embedding {
            target,
            text,
            vector,
        })
    }
}

impl Backlog for EmbeddingBacklog<'_> {
    type RowId = i64;
    type Row = String;
    type Error = PassError;

    fn permit(&mut self) -> Option<&mut watch::Receiver<bool>> {
        self.permit.as_mut()
    }

    async fn page_after(&self, after: Option<i64>) -> Result<Vec<(i64, String)>, PassError> {
        let chunks = async |source| {
            let page = self
                .store
                .chunks_missing_embedding(source, after.map(ChunkId), BATCH_SIZE)
                .await?;
            Ok(page.into_iter().map(|(id, text)| (id.0, text)).collect())
        };
        match self.kind {
            RowKind::MessageChunks => chunks(ChunkSource::Message).await,
            RowKind::DocumentChunks => chunks(ChunkSource::Document).await,
            RowKind::Memories => {
                let page = self
                    .store
                    .memories_missing_embedding(after.map(MemoryId), BATCH_SIZE)
                    .await?;
                Ok(page.into_iter().map(|(id, text)| (id.0, text)).collect())
            }
        }
    }

    async fn do_page(&mut self, page: Vec<(i64, String)>) -> Result<usize, PassError> {
        let texts: Vec<&str> = page.iter().map(|(_, text)| text.as_str()).collect();
        let (outcomes, server_away) = embed_page(self.embedder, &texts).await;
        let rows = page.len();
        let embeddings: Vec<Embedding> = page
            .into_iter()
            .zip(outcomes)
            .filter_map(|((id, text), outcome)| self.embedding_to_store(id, text, outcome))
            .collect();
        let rows_left = rows - embeddings.len();
        self.store.store_embeddings(embeddings).await?;
        server_away.map_or(Ok(rows_left), |err| Err(err.into()))
    }
}

/// Message chunks, memories, then document chunks, each drained in turn.
struct EmbeddingPass<'a> {
    backlogs: [EmbeddingBacklog<'a>; 3],
}

impl Pass for EmbeddingPass<'_> {
    type Error = PassError;

    async fn run(&mut self, signals: &Signals) -> Result<DrainEnd, PassError> {
        let mut rows_left = 0usize;
        for backlog in &mut self.backlogs {
            match drain(backlog, signals).await? {
                DrainEnd::Emptied { rows_left: left } => rows_left += left,
                cut_short => return Ok(cut_short),
            }
        }
        Ok(DrainEnd::Emptied { rows_left })
    }
}

/// What became of one input of a page.
#[derive(Debug)]
enum InputOutcome {
    Embedded(Vec<f32>),
    /// The server refused it for its own content.
    Refused(EmbedError),
    /// It failed alone for another reason, and the server then showed that it works.
    FailedAlone(EmbedError),
    /// It failed with nothing to show whether the server works.
    Unsettled,
}

/// The inputs of a page sent one request each, telling an input that fails alone from
/// a server that is away.
struct SingleInputs<'a> {
    embedder: &'a dyn Embedder,
    outcomes: Vec<InputOutcome>,
    /// The last input to fail for a reason other than its content, by its place in
    /// `outcomes`, while no later request has succeeded.
    unconfirmed: Option<(usize, EmbedError)>,
}

impl SingleInputs<'_> {
    /// Take in the result of sending the next input alone; errors when the server is away.
    async fn record(&mut self, result: Result<Vec<f32>, EmbedError>) -> Result<(), EmbedError> {
        let outcome = match result {
            Ok(vector) => {
                if let Some((failed, err)) = self.unconfirmed.take() {
                    self.outcomes[failed] = InputOutcome::FailedAlone(err);
                }
                InputOutcome::Embedded(vector)
            }
            Err(err) if err.is_input_caused() => InputOutcome::Refused(err),
            Err(EmbedError::NotReady) => return Err(EmbedError::NotReady),
            Err(err) => {
                self.confirm_server_works().await?;
                self.unconfirmed = Some((self.outcomes.len(), err));
                InputOutcome::Unsettled
            }
        };
        self.outcomes.push(outcome);
        Ok(())
    }

    /// Settle the unconfirmed failure, when there is one, by having the server embed
    /// [`SERVER_CHECK_TEXT`]; errors with that failure when the server cannot.
    async fn confirm_server_works(&mut self) -> Result<(), EmbedError> {
        let Some((failed, err)) = self.unconfirmed.take() else {
            return Ok(());
        };
        if embed_document(self.embedder, SERVER_CHECK_TEXT)
            .await
            .is_err()
        {
            return Err(err);
        }
        self.outcomes[failed] = InputOutcome::FailedAlone(err);
        Ok(())
    }
}

/// Spawn the worker. It embeds the rows missing an embedding at once and again each time
/// `store` signals that a write left more, document chunks last and only while
/// `documents_gate` reads `true`, retrying with backoff a pass that stopped early or left
/// rows. It exits on `shutdown` between batches; `store`'s database must outlive it.
pub fn spawn_embedder_task(
    embedder: Arc<dyn Embedder>,
    store: Arc<dyn EmbeddingQueue>,
    documents_gate: watch::Receiver<bool>,
    shutdown: watch::Receiver<bool>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let signals = Signals {
            wake: store.rows_to_embed(),
            shutdown,
        };
        let backlogs = [
            (RowKind::MessageChunks, None),
            (RowKind::Memories, None),
            (RowKind::DocumentChunks, Some(documents_gate)),
        ]
        .map(|(kind, permit)| EmbeddingBacklog {
            kind,
            embedder: &*embedder,
            store: &*store,
            permit,
            failures: HashMap::new(),
        });
        keep_drained(
            "embedding",
            &signals,
            backoff_delay,
            EmbeddingPass { backlogs },
        )
        .await;
        tracing::debug!(target: "assistd::embed", "embed worker exiting");
    })
}

/// Embed `texts` in one request, or one input at a time when that fails. Returns an
/// outcome for each input reached, in order, and why the server is taken to be away
/// when it is.
async fn embed_page(
    embedder: &dyn Embedder,
    texts: &[&str],
) -> (Vec<InputOutcome>, Option<EmbedError>) {
    let batch_failure = match embedder.embed_documents(texts).await {
        Ok(vectors) => {
            let outcomes = vectors.into_iter().map(InputOutcome::Embedded).collect();
            return (outcomes, None);
        }
        Err(EmbedError::NotReady) => return (Vec::new(), Some(EmbedError::NotReady)),
        Err(err) => err,
    };
    let mut singles = SingleInputs {
        embedder,
        outcomes: Vec::with_capacity(texts.len()),
        unconfirmed: None,
    };
    let server_away = match texts {
        [_] => singles.record(Err(batch_failure)).await,
        _ => {
            tracing::debug!(
                target: "assistd::embed",
                batch = texts.len(),
                error = %batch_failure,
                "batch embed failed; retrying inputs individually"
            );
            record_each_alone(&mut singles, texts).await
        }
    };
    let server_away = match server_away {
        Ok(()) => singles.confirm_server_works().await.err(),
        Err(err) => Some(err),
    };
    (singles.outcomes, server_away)
}

async fn record_each_alone(
    singles: &mut SingleInputs<'_>,
    texts: &[&str],
) -> Result<(), EmbedError> {
    for &text in texts {
        let result = embed_document(singles.embedder, text).await;
        singles.record(result).await?;
    }
    Ok(())
}

async fn embed_document(embedder: &dyn Embedder, text: &str) -> Result<Vec<f32>, EmbedError> {
    embedder
        .embed_documents(&[text])
        .await?
        .pop()
        .ok_or(EmbedError::CountMismatch {
            got: 0,
            expected: 1,
        })
}

fn log_refused_row(target: EmbeddingTarget, err: &EmbedError) {
    tracing::warn!(
        target: "assistd::embed",
        row = %target,
        error = %err,
        "row cannot be embedded; marking it so it is not retried"
    );
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    use async_trait::async_trait;
    use tokio::sync::Notify;

    use assistd_memory::{
        ChunkId, Embedding, EmbeddingSpace, MemoryId, Result as MemoryResult, UnindexedRows,
    };

    /// The id of the first document chunk a [`Harness`] adds, above every message chunk's.
    const FIRST_DOCUMENT_CHUNK: i64 = 1_000_000;

    use super::*;

    /// Embeds every text as `[1.0]` after `batch_time`, except that `"bad"` fails for its
    /// own content, `"stuck"` fails with a server error, `"crowded"` does so unless sent
    /// alone, and every call fails with a server error while `down`. Records the inputs
    /// of every call.
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
            } else if texts.contains(&"stuck") || (texts.len() > 1 && texts.contains(&"crowded")) {
                Some(reqwest::StatusCode::INTERNAL_SERVER_ERROR)
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
        rows_to_embed: Arc<Notify>,
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
    impl EmbeddingQueue for PendingStore {
        async fn adopt_embedding_space(
            &self,
            _space: &EmbeddingSpace,
        ) -> MemoryResult<Option<EmbeddingSpace>> {
            Ok(None)
        }
        fn rows_to_embed(&self) -> Arc<Notify> {
            Arc::clone(&self.rows_to_embed)
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
    }

    struct Harness {
        embedder: Arc<ScriptedEmbedder>,
        store: Arc<PendingStore>,
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
            let (documents_gate, documents_gate_rx) = watch::channel(true);
            let (shutdown, shutdown_rx) = watch::channel(false);
            Self {
                task: spawn_embedder_task(
                    embedder.clone(),
                    store.clone(),
                    documents_gate_rx,
                    shutdown_rx,
                ),
                embedder,
                store,
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
        harness.store.rows_to_embed.notify_one();
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert_eq!(harness.missing_chunks(), 0);

        let calls = harness.shut_down().await;
        let batch_sizes: Vec<usize> = calls.iter().map(Vec::len).collect();
        assert_eq!(batch_sizes, [BATCH_SIZE, 1, 1, 1]);
    }

    #[tokio::test(start_paused = true)]
    async fn worker_marks_no_row_while_the_server_answers_nothing_and_embeds_them_once_it_does() {
        let harness = Harness::spawn(&["a", "b"], true);

        assert_eq!(harness.stored_after(Duration::from_secs(600)).await, []);
        assert_eq!(harness.missing_chunks(), 2);
        assert!(
            harness
                .embedder
                .calls
                .lock()
                .unwrap()
                .contains(&vec![SERVER_CHECK_TEXT.to_owned()]),
            "the server was never asked for the fixed text"
        );

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
    async fn worker_stores_a_whole_batch_that_fails_together_when_its_inputs_succeed_alone() {
        let harness = Harness::spawn(&["a", "crowded", "c"], false);

        assert_eq!(
            harness.stored_after(Duration::from_millis(10)).await,
            [
                (0, Some(vec![1.0])),
                (1, Some(vec![1.0])),
                (2, Some(vec![1.0]))
            ]
        );
        assert_eq!(
            harness.shut_down().await,
            [
                vec!["a", "crowded", "c"],
                vec!["a"],
                vec!["crowded"],
                vec!["c"]
            ]
        );
    }

    #[tokio::test(start_paused = true)]
    async fn worker_marks_a_row_that_fails_alone_three_times_beside_rows_that_embed() {
        let mut texts = vec!["a", "stuck"];
        texts.resize(BATCH_SIZE + 1, "later");
        let harness = Harness::spawn(&texts, false);

        let stored = harness.stored_after(Duration::from_millis(10)).await;
        assert_eq!(
            stored.len(),
            BATCH_SIZE,
            "a row behind the failing one waited"
        );
        assert!(
            stored
                .iter()
                .all(|(id, vector)| *id != 1 && vector.is_some())
        );

        let stored = harness.stored_after(Duration::from_secs(600)).await;
        assert_eq!(stored.last(), Some(&(1, None)));
        assert_eq!(stored.len(), BATCH_SIZE + 1);
        let calls = harness.shut_down().await;
        let attempts = calls
            .iter()
            .filter(|texts| texts.iter().any(|text| text == "stuck"))
            .count();
        assert_eq!(attempts, 1 + FAILURES_BEFORE_REFUSAL as usize);
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
        harness.store.rows_to_embed.notify_one();
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
        harness.store.rows_to_embed.notify_one();
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
