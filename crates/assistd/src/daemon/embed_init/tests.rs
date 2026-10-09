use std::time::Duration;

use assistd_embed::{EmbedError, enqueue_embed_job};
use async_trait::async_trait;

use super::*;
use crate::daemon::test_support::app_state;

#[derive(Debug)]
struct StubEmbedder;

#[async_trait]
impl Embedder for StubEmbedder {
    async fn embed(&self, _text: String) -> Result<Vec<f32>, EmbedError> {
        Ok(vec![1.0, 0.0])
    }
    fn model(&self) -> &'static str {
        "stub"
    }
    fn dim(&self) -> usize {
        2
    }
}

fn enabled_config() -> Config {
    let mut config = Config::default();
    config.embedding.enabled = true;
    config
}

#[tokio::test]
async fn jobs_queued_before_the_embedder_is_up_are_embedded_once_it_is() {
    let config = enabled_config();
    let (handles, startup) = prepare(&config, None);
    let mut startup = startup.expect("embedding is enabled");
    let (writer_tx, mut writer_rx) = mpsc::channel(4);
    startup.writer_tx = Arc::new(writer_tx);
    enqueue_embed_job(
        &handles.embed_tx,
        EmbedJob::Chunk {
            chunk_id: 7,
            text: "queued during startup".into(),
        },
    );

    let mut state = app_state(config);
    state.memory.embedder = handles.embedder;
    assert_eq!(
        state.memory.embedder.get().unwrap_err().to_string(),
        "embedding is still starting"
    );

    let (_worker_shutdown, worker_shutdown_rx) = watch::channel(false);
    let worker = serve_embedder(&state, Arc::new(StubEmbedder), startup, worker_shutdown_rx);
    assert!(state.memory.embedder.get().is_ok());

    let stored = tokio::time::timeout(Duration::from_secs(5), writer_rx.recv())
        .await
        .expect("queued job never embedded");
    let Some(WriteOp::StoreChunkEmbedding {
        chunk_id,
        model,
        ack,
        ..
    }) = stored
    else {
        panic!("expected a chunk embedding write");
    };
    assert_eq!((chunk_id, model.as_str()), (7, "stub"));
    let _ = ack.send(Ok(()));
    worker.abort();
}

#[tokio::test]
async fn disabled_embedding_says_why_and_closes_the_queue() {
    let mut config = Config::default();
    config.embedding.enabled = false;
    let (handles, startup) = prepare(&config, None);
    assert!(startup.is_none());
    assert_eq!(
        handles.embedder.get().unwrap_err().to_string(),
        "embedding is unavailable: disabled in config (embedding.enabled = false)"
    );
    assert!(handles.embed_tx.is_closed());
}
