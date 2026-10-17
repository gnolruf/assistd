//! `MemoryStack`: persistence and embedding handles owned by `AppState`.

use std::sync::Arc;

use tokio::sync::mpsc;

use assistd_config::EmbeddingConfig;
use assistd_embed::{EmbedJob, EmbedderHandle};
use assistd_memory::{
    MemoryStore, NoMemoryStore, NoSemanticStore, NoTranscriptStore, SemanticStore, TranscriptStore,
};
use assistd_tools::MemoryOps;
use assistd_utils::readiness::Readiness;

/// Persistent stores, embedding pipeline, and the memory tool ops built
/// over them.
#[derive(Debug)]
pub struct MemoryStack {
    pub memory: Arc<dyn MemoryStore>,
    pub transcripts: Arc<dyn TranscriptStore>,
    pub memory_ops: Arc<MemoryOps>,
    pub embedder: Arc<EmbedderHandle>,
    pub semantic: Arc<dyn SemanticStore>,
    pub embed_tx: mpsc::Sender<EmbedJob>,
    pub embedding_cfg: EmbeddingConfig,
}

impl MemoryStack {
    /// Construct a stack with every store wired to its no-op placeholder
    /// and the embedder unavailable.
    pub fn disabled(embedding_cfg: EmbeddingConfig) -> Self {
        let memory: Arc<dyn MemoryStore> = Arc::new(NoMemoryStore);
        let transcripts: Arc<dyn TranscriptStore> = Arc::new(NoTranscriptStore);
        let memory_ops = Arc::new(MemoryOps::new(memory.clone(), transcripts.clone()));
        let (embed_tx, embed_rx) = mpsc::channel::<EmbedJob>(1);
        drop(embed_rx);
        Self {
            memory,
            transcripts,
            memory_ops,
            embedder: Arc::new(EmbedderHandle::new(Readiness::Unavailable(
                "embedding not configured".into(),
            ))),
            semantic: Arc::new(NoSemanticStore),
            embed_tx,
            embedding_cfg,
        }
    }

    /// Replace the fact store, rebuilding `memory_ops` over it.
    pub fn with_memory(mut self, memory: Arc<dyn MemoryStore>) -> Self {
        self.memory = memory.clone();
        self.memory_ops = Arc::new(MemoryOps::new(memory, self.transcripts.clone()));
        self
    }

    /// Replace the transcript store, rebuilding `memory_ops` over it.
    pub fn with_transcripts(mut self, transcripts: Arc<dyn TranscriptStore>) -> Self {
        self.transcripts = transcripts.clone();
        self.memory_ops = Arc::new(MemoryOps::new(self.memory.clone(), transcripts));
        self
    }

    /// Replace the embedder.
    pub fn with_embedder(mut self, embedder: Arc<EmbedderHandle>) -> Self {
        self.embedder = embedder;
        self
    }

    /// Replace the semantic store.
    pub fn with_semantic(mut self, semantic: Arc<dyn SemanticStore>) -> Self {
        self.semantic = semantic;
        self
    }

    /// Replace the embed-job queue sender.
    pub fn with_embed_tx(mut self, tx: mpsc::Sender<EmbedJob>) -> Self {
        self.embed_tx = tx;
        self
    }
}
