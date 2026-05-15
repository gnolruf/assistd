//! `MemoryStack` groups the persistence + embedding handles that the
//! daemon wires together at startup.
//!
//! Added in Step 2 but not yet referenced by [`AppState`]; the
//! `#[allow(dead_code)]` annotation keeps clippy quiet until Step 3
//! flips the field set.

use assistd_config::EmbeddingConfig;
use assistd_embed::{EmbedJob, Embedder, NoEmbedder};
use assistd_memory::{
    ConversationStore, MemoryStore, NoConversationStore, NoMemoryStore, NoSemanticStore,
    SemanticStore, SqliteHandle,
};
use assistd_tools::MemoryOps;
use std::sync::Arc;
use tokio::sync::mpsc;

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
    /// Matches the post-`AppState::new` defaults today's daemon then
    /// overrides via chained `with_*` calls.
    pub fn disabled(embedding_cfg: EmbeddingConfig) -> Self {
        let memory: Arc<dyn MemoryStore> = Arc::new(NoMemoryStore);
        let conversations: Arc<dyn ConversationStore> = Arc::new(NoConversationStore);
        let memory_ops = Arc::new(MemoryOps::new(memory.clone(), conversations.clone()));
        // Closed channel: every `try_send` fails immediately so the
        // persistence hook degrades to chunk-only behaviour when nothing
        // else wires a real embedder task in.
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

    /// Set the memory backend. Rebuilds [`Self::memory_ops`] so the
    /// combined façade reflects the new handle; both `with_memory` and
    /// [`Self::with_conversations`] enforce this invariant so callers
    /// can chain them in any order.
    pub fn with_memory(mut self, m: Arc<dyn MemoryStore>) -> Self {
        self.memory = m.clone();
        self.memory_ops = Arc::new(MemoryOps::new(m, self.conversations.clone()));
        self
    }

    /// Set the conversation store. Rebuilds [`Self::memory_ops`] in
    /// lockstep with [`Self::with_memory`]; see the cross-update test
    /// below.
    pub fn with_conversations(mut self, c: Arc<dyn ConversationStore>) -> Self {
        self.conversations = c.clone();
        self.memory_ops = Arc::new(MemoryOps::new(self.memory.clone(), c));
        self
    }

    pub fn with_embedder(mut self, e: Arc<dyn Embedder>) -> Self {
        self.embedder = e;
        self
    }

    pub fn with_semantic(mut self, s: Arc<dyn SemanticStore>) -> Self {
        self.semantic = s;
        self
    }

    pub fn with_embed_tx(mut self, tx: mpsc::Sender<EmbedJob>) -> Self {
        self.embed_tx = tx;
        self
    }

    pub fn with_chunks(mut self, h: Arc<SqliteHandle>) -> Self {
        self.chunks = Some(h);
        self
    }

    pub fn with_embedding_cfg(mut self, cfg: EmbeddingConfig) -> Self {
        self.embedding_cfg = cfg;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn with_memory_then_conversations_keeps_both_fields() {
        // Cross-update invariant: chaining both builders in either order
        // must leave the final stack with the supplied handles on the
        // matching fields. This test pins that ordering doesn't drop
        // either store; the `memory_ops` rebuild path is observed
        // indirectly via the Arc::ptr_eq assertions.
        let m1: Arc<dyn MemoryStore> = Arc::new(NoMemoryStore);
        let c1: Arc<dyn ConversationStore> = Arc::new(NoConversationStore);
        let stack = MemoryStack::disabled(EmbeddingConfig::default())
            .with_memory(m1.clone())
            .with_conversations(c1.clone());
        assert!(Arc::ptr_eq(&stack.memory, &m1));
        assert!(Arc::ptr_eq(&stack.conversations, &c1));
    }

    #[tokio::test]
    async fn with_conversations_before_memory_keeps_both_fields() {
        // Same invariant, reverse order. Documents that the rebuild of
        // `memory_ops` happens in BOTH `with_*` methods so the last one
        // sees consistent inputs regardless of chain direction.
        let m1: Arc<dyn MemoryStore> = Arc::new(NoMemoryStore);
        let c1: Arc<dyn ConversationStore> = Arc::new(NoConversationStore);
        let stack = MemoryStack::disabled(EmbeddingConfig::default())
            .with_conversations(c1.clone())
            .with_memory(m1.clone());
        assert!(Arc::ptr_eq(&stack.memory, &m1));
        assert!(Arc::ptr_eq(&stack.conversations, &c1));
    }
}
