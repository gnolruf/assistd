//! `MemoryStack`: the persistent stores and embedding handles, which also answer the
//! `Memory*` requests.

use std::sync::Arc;

use tokio::sync::Notify;

use assistd_config::EmbeddingConfig;
use assistd_embed::EmbedderHandle;
use assistd_memory::{
    MemoryStore, NoMemoryStore, NoSemanticStore, NoTaskStore, NoTranscriptStore, SemanticStore,
    TaskStore, TranscriptStore,
};
use assistd_tools::MemoryOps;
use assistd_utils::readiness::Readiness;

/// Persistent stores, embedding pipeline, and the memory tool ops built
/// over them.
#[derive(Debug)]
pub struct MemoryStack {
    pub memory: Arc<dyn MemoryStore>,
    pub transcripts: Arc<dyn TranscriptStore>,
    pub tasks: Arc<dyn TaskStore>,
    pub memory_ops: Arc<MemoryOps>,
    pub embedder: Arc<EmbedderHandle>,
    pub semantic: Arc<dyn SemanticStore>,
    /// Wakes the embedding worker after a write leaves rows to embed.
    pub embed_wake: Arc<Notify>,
    pub embedding_cfg: EmbeddingConfig,
}

impl MemoryStack {
    /// Construct a stack with every store wired to its no-op placeholder
    /// and the embedder unavailable.
    pub fn disabled(embedding_cfg: EmbeddingConfig) -> Self {
        let memory: Arc<dyn MemoryStore> = Arc::new(NoMemoryStore);
        let transcripts: Arc<dyn TranscriptStore> = Arc::new(NoTranscriptStore);
        let memory_ops = Arc::new(MemoryOps::new(memory.clone(), transcripts.clone()));
        Self {
            memory,
            transcripts,
            tasks: Arc::new(NoTaskStore),
            memory_ops,
            embedder: Arc::new(EmbedderHandle::new(Readiness::Unavailable(
                "embedding not configured".into(),
            ))),
            semantic: Arc::new(NoSemanticStore),
            embed_wake: Arc::default(),
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

    /// Replace the task store.
    pub fn with_tasks(mut self, tasks: Arc<dyn TaskStore>) -> Self {
        self.tasks = tasks;
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

    /// Replace the embedding worker's wake signal.
    pub fn with_embed_wake(mut self, embed_wake: Arc<Notify>) -> Self {
        self.embed_wake = embed_wake;
        self
    }
}
