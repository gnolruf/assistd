//! Internal CRUD façade over the persistent memory subsystem.
//!
//! Wraps [`assistd_memory::MemoryStore`] and
//! [`assistd_memory::ConversationStore`] in a single struct so the
//! daemon's IPC handlers and the `assistd memory ...` CLI hit one
//! object instead of juggling two trait objects.
//!
//! No LLM-callable [`crate::Tool`] impls live here — the model gets
//! access to memory only via the daemon's `run` chain or via `Tool`
//! impls a future milestone might layer on top of this façade.

use std::sync::Arc;

use anyhow::Result;
use assistd_memory::{ConversationStore, MemoryStore, SearchHit, TurnSummary};

/// Default cap on `search` / `list` result sizes when the IPC client
/// asks for `limit = 0` (the wire default for the `MemorySearch`
/// variant). Big enough to be useful for casual `assistd memory search`
/// runs; the LLM-facing path will likely want to pass an explicit
/// smaller limit.
pub const DEFAULT_SEARCH_LIMIT: usize = 50;

/// Combined CRUD handle over both the flat KV store and the richer
/// conversation history. Cheap to clone (just two `Arc`s).
#[derive(Clone)]
pub struct MemoryOps {
    pub store: Arc<dyn MemoryStore>,
    pub conversations: Arc<dyn ConversationStore>,
}

impl MemoryOps {
    pub fn new(store: Arc<dyn MemoryStore>, conversations: Arc<dyn ConversationStore>) -> Self {
        Self {
            store,
            conversations,
        }
    }

    pub async fn save(&self, key: &str, value: String) -> Result<()> {
        self.store.save(key, value).await
    }

    pub async fn load(&self, key: &str) -> Result<Option<String>> {
        self.store.load(key).await
    }

    pub async fn list(&self, prefix: &str) -> Result<Vec<String>> {
        self.store.list(prefix).await
    }

    pub async fn delete(&self, key: &str) -> Result<()> {
        self.store.delete(key).await
    }

    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchHit>> {
        let limit = if limit == 0 {
            DEFAULT_SEARCH_LIMIT
        } else {
            limit
        };
        self.conversations.search(query, limit).await
    }

    pub async fn recent_turns(&self, limit: usize) -> Result<Vec<TurnSummary>> {
        let limit = if limit == 0 {
            DEFAULT_SEARCH_LIMIT
        } else {
            limit
        };
        self.conversations.recent_turns(limit).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assistd_memory::{NoConversationStore, NoMemoryStore};

    fn no_ops() -> MemoryOps {
        MemoryOps::new(Arc::new(NoMemoryStore), Arc::new(NoConversationStore))
    }

    #[tokio::test]
    async fn no_backend_save_then_load_returns_none() {
        // Mirrors the contract of `NoMemoryStore` exactly: every method
        // succeeds and `load` reports the key as absent. The façade
        // must not pretend the placeholder backend stores anything.
        let ops = no_ops();
        ops.save("k", "v".into()).await.unwrap();
        assert_eq!(ops.load("k").await.unwrap(), None);
    }

    #[tokio::test]
    async fn no_backend_search_and_list_return_empty() {
        let ops = no_ops();
        assert!(ops.list("pref:").await.unwrap().is_empty());
        assert!(ops.search("anything", 10).await.unwrap().is_empty());
        assert!(ops.recent_turns(0).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn search_zero_limit_falls_back_to_default() {
        // Use a SqliteConversationStore so the limit handling is real.
        // Path: write enough rows that DEFAULT_SEARCH_LIMIT actually
        // bounds the result. Verifies the façade rewrites `0` to a sane
        // default before delegating.
        use assistd_memory::{
            PersistedMessage, SqliteConversationStore, SqliteHandle, SqliteMemoryStore,
        };
        use tokio::sync::watch;

        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("memory.db");
        let (_tx, rx) = watch::channel(false);
        let (handle, _writer) = SqliteHandle::open(&path, rx).await.unwrap();
        let handle = Arc::new(handle);
        let conv = SqliteConversationStore::new(handle.clone());
        let mem = SqliteMemoryStore::new(handle);

        let session = conv.begin_session(0).await.unwrap();
        for i in 0..(DEFAULT_SEARCH_LIMIT + 5) {
            conv.append_message(
                &session,
                None,
                PersistedMessage::user(format!("alpha message {i}")),
            )
            .await
            .unwrap();
        }

        let ops = MemoryOps::new(Arc::new(mem), Arc::new(conv));
        let hits = ops.search("alpha", 0).await.unwrap();
        assert_eq!(hits.len(), DEFAULT_SEARCH_LIMIT);
    }
}
