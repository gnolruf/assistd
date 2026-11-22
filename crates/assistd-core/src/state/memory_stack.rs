//! `MemoryStack`: the persistent stores and embedding handles, which also answer the
//! `Memory*` requests.

use std::sync::Arc;

use assistd_config::EmbeddingConfig;
use assistd_embed::EmbedderHandle;
use assistd_memory::{
    DocumentStore, EmbeddingQueue, MemoryStore, NoSemanticStore, RecallStore, SemanticStore,
    TaskStore, TranscriptStore,
};
use assistd_tools::MemoryOps;
use assistd_utils::readiness::Readiness;

/// Persistent stores, embedding pipeline, and the memory tool ops built
/// over them.
#[derive(Debug)]
pub struct MemoryStack {
    pub memory: Arc<MemoryStore>,
    pub transcripts: Arc<dyn TranscriptStore>,
    pub tasks: Arc<TaskStore>,
    pub memory_ops: Arc<MemoryOps>,
    pub embedder: Arc<EmbedderHandle>,
    pub recall: Arc<dyn RecallStore>,
    pub embedding_queue: Arc<dyn EmbeddingQueue>,
    pub document_store: Arc<dyn DocumentStore>,
    pub embedding_cfg: EmbeddingConfig,
}

impl MemoryStack {
    /// A stack over the three stores with embedding off: the embedder unavailable
    /// and a semantic store that finds nothing.
    pub fn new(
        memory: Arc<MemoryStore>,
        transcripts: Arc<dyn TranscriptStore>,
        tasks: Arc<TaskStore>,
        embedding_cfg: EmbeddingConfig,
    ) -> Self {
        let memory_ops = Arc::new(MemoryOps::new(memory.clone(), transcripts.clone()));
        Self {
            memory,
            transcripts,
            tasks,
            memory_ops,
            embedder: Arc::new(EmbedderHandle::new(Readiness::Unavailable(
                "embedding not configured".into(),
            ))),
            recall: Arc::new(NoSemanticStore),
            embedding_queue: Arc::new(NoSemanticStore),
            document_store: Arc::new(NoSemanticStore),
            embedding_cfg,
        }
    }

    /// Replace the embedder.
    pub fn with_embedder(mut self, embedder: Arc<EmbedderHandle>) -> Self {
        self.embedder = embedder;
        self
    }

    /// Replace all three parts of the semantic store with `semantic`.
    pub fn with_semantic(mut self, semantic: Arc<dyn SemanticStore>) -> Self {
        self.recall = semantic.clone();
        self.embedding_queue = semantic.clone();
        self.document_store = semantic;
        self
    }
}
