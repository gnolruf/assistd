use super::*;
use async_trait::async_trait;
use std::sync::Mutex;
use std::time::Duration;

/// Returns `vec` for every input except `"bad"`, which fails any call
/// that includes it. Records the inputs of every call.
struct MockEmbedder {
    calls: Mutex<Vec<Vec<String>>>,
    vec: Vec<f32>,
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
        if texts.contains(&"bad") {
            return Err(EmbedError::Disabled);
        }
        Ok(vec![self.vec.clone(); texts.len()])
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
            calls: Mutex::new(Vec::new()),
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

    async fn send_chunks(&self, texts: &[&str]) {
        for (chunk_id, text) in (0..).zip(texts) {
            self.jobs
                .send(EmbedJob::Chunk {
                    chunk_id,
                    text: (*text).to_owned(),
                })
                .await
                .unwrap();
        }
    }

    async fn next_write(&mut self) -> WriteOp {
        tokio::time::timeout(Duration::from_secs(2), self.writes.recv())
            .await
            .expect("write op arrived in time")
            .expect("writer channel open")
    }

    async fn expect_chunk_writes(&mut self, ids: &[i64]) {
        for &id in ids {
            match self.next_write().await {
                WriteOp::StoreChunkEmbedding { chunk_id, ack, .. } => {
                    assert_eq!(chunk_id, id);
                    ack.send(Ok(())).unwrap();
                }
                _ => panic!("expected StoreChunkEmbedding"),
            }
        }
    }

    /// Signal shutdown while keeping the job sender alive, so only
    /// the shutdown path can end the worker. Returns the inputs of
    /// every embed call.
    async fn shut_down(mut self) -> Vec<Vec<String>> {
        self.shutdown.send_replace(true);
        tokio::time::timeout(Duration::from_secs(2), self.task)
            .await
            .expect("worker exited after shutdown")
            .expect("worker did not panic");
        assert!(self.writes.try_recv().is_err(), "unexpected extra write");
        std::mem::take(&mut *self.embedder.calls.lock().unwrap())
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
    assert_eq!(h.shut_down().await, vec![vec!["hello"]]);
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
    assert_eq!(h.shut_down().await, vec![vec!["vim"]]);
}

#[tokio::test]
async fn worker_coalesces_queued_jobs_into_one_embed_call() {
    let mut h = Harness::spawn(vec![1.0]);
    h.send_chunks(&["t0", "t1", "t2"]).await;

    h.expect_chunk_writes(&[0, 1, 2]).await;
    assert_eq!(h.shut_down().await, vec![vec!["t0", "t1", "t2"]]);
}

#[tokio::test]
async fn worker_drops_only_the_bad_job_when_a_batch_fails() {
    let mut h = Harness::spawn(vec![1.0]);
    h.send_chunks(&["t0", "bad", "t2"]).await;

    h.expect_chunk_writes(&[0, 2]).await;
    assert_eq!(
        h.shut_down().await,
        vec![vec!["t0", "bad", "t2"], vec!["t0"], vec!["bad"], vec!["t2"]]
    );
}

#[tokio::test]
async fn worker_embeds_queued_jobs_before_honouring_shutdown() {
    let mut h = Harness::spawn(vec![1.0]);
    h.send_chunks(&["t0", "t1", "t2"]).await;
    h.shutdown.send(true).unwrap();

    h.expect_chunk_writes(&[0, 1, 2]).await;
    assert_eq!(h.shut_down().await.concat(), vec!["t0", "t1", "t2"]);
}
