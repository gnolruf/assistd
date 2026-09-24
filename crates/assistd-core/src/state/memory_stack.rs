//! `MemoryStack`: persistence and embedding handles owned by `AppState`.

use assistd_config::EmbeddingConfig;
use assistd_embed::{EmbedJob, Embedder, NoEmbedder};
use assistd_memory::{
    ConversationStore, MemoryStore, NoConversationStore, NoMemoryStore, NoSemanticStore,
    SemanticStore, SqliteHandle,
};
use assistd_tools::MemoryOps;
use std::sync::Arc;
use tokio::sync::mpsc;

/// Persistent stores, embedding pipeline, and the memory tool ops built
/// over them.
pub struct MemoryStack {
    pub memory: Arc<dyn MemoryStore>,
    pub conversations: Arc<dyn ConversationStore>,
    pub memory_ops: Arc<MemoryOps>,
    pub embedder: Arc<dyn Embedder>,
    pub semantic: Arc<dyn SemanticStore>,
    pub embed_tx: mpsc::Sender<EmbedJob>,
    pub chunks: Option<Arc<SqliteHandle>>,
    pub embedding_cfg: EmbeddingConfig,
}

impl MemoryStack {
    /// Construct a stack with every store wired to its no-op placeholder.
    pub fn disabled(embedding_cfg: EmbeddingConfig) -> Self {
        let memory: Arc<dyn MemoryStore> = Arc::new(NoMemoryStore);
        let conversations: Arc<dyn ConversationStore> = Arc::new(NoConversationStore);
        let memory_ops = Arc::new(MemoryOps::new(memory.clone(), conversations.clone()));
        let (embed_tx, embed_rx) = mpsc::channel::<EmbedJob>(1);
        drop(embed_rx);
        Self {
            memory,
            conversations,
            memory_ops,
            embedder: Arc::new(NoEmbedder),
            semantic: Arc::new(NoSemanticStore),
            embed_tx,
            chunks: None,
            embedding_cfg,
        }
    }

    /// Replace the fact store, rebuilding `memory_ops` over it.
    pub fn with_memory(mut self, m: Arc<dyn MemoryStore>) -> Self {
        self.memory = m.clone();
        self.memory_ops = Arc::new(MemoryOps::new(m, self.conversations.clone()));
        self
    }

    /// Replace the conversation store, rebuilding `memory_ops` over it.
    pub fn with_conversations(mut self, c: Arc<dyn ConversationStore>) -> Self {
        self.conversations = c.clone();
        self.memory_ops = Arc::new(MemoryOps::new(self.memory.clone(), c));
        self
    }

    /// Replace the embedder.
    pub fn with_embedder(mut self, e: Arc<dyn Embedder>) -> Self {
        self.embedder = e;
        self
    }

    /// Replace the semantic store.
    pub fn with_semantic(mut self, s: Arc<dyn SemanticStore>) -> Self {
        self.semantic = s;
        self
    }

    /// Replace the embed-job queue sender.
    pub fn with_embed_tx(mut self, tx: mpsc::Sender<EmbedJob>) -> Self {
        self.embed_tx = tx;
        self
    }

    /// Attach the SQLite handle new messages are chunked into. Without
    /// one, persisted messages are never chunked or embedded.
    pub fn with_chunks(mut self, h: Arc<SqliteHandle>) -> Self {
        self.chunks = Some(h);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_wired(
        stack: &MemoryStack,
        m: &Arc<dyn MemoryStore>,
        c: &Arc<dyn ConversationStore>,
        order: &str,
    ) {
        assert!(Arc::ptr_eq(&stack.memory, m), "{order}: memory");
        assert!(
            Arc::ptr_eq(&stack.conversations, c),
            "{order}: conversations"
        );
        assert!(
            Arc::ptr_eq(&stack.memory_ops.store, m),
            "{order}: ops store"
        );
        assert!(
            Arc::ptr_eq(&stack.memory_ops.conversations, c),
            "{order}: ops conversations"
        );
    }

    #[test]
    fn memory_ops_tracks_both_stores_in_either_builder_order() {
        let m: Arc<dyn MemoryStore> = Arc::new(NoMemoryStore);
        let c: Arc<dyn ConversationStore> = Arc::new(NoConversationStore);

        let stack = MemoryStack::disabled(EmbeddingConfig::default())
            .with_memory(m.clone())
            .with_conversations(c.clone());
        assert_wired(&stack, &m, &c, "memory first");

        let stack = MemoryStack::disabled(EmbeddingConfig::default())
            .with_conversations(c.clone())
            .with_memory(m.clone());
        assert_wired(&stack, &m, &c, "conversations first");
    }
}
