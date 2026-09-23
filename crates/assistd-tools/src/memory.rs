//! CRUD façade over [`assistd_memory::MemoryStore`] and
//! [`assistd_memory::ConversationStore`]. The LLM-callable tools built
//! on it live in [`crate::memory_tools`].

use std::sync::Arc;

use assistd_memory::{ConversationStore, MemoryError, MemoryStore};

pub use assistd_memory::MemoryRecord;

type Result<T> = std::result::Result<T, MemoryError>;

/// Result cap applied when a caller passes `limit = 0`.
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

    /// Save a key/value memory and return its row id.
    pub async fn save(&self, key: &str, value: String) -> Result<i64> {
        self.store.save(key, value).await
    }

    /// Load the value for `key`, returning `None` if not present.
    pub async fn load(&self, key: &str) -> Result<Option<String>> {
        self.store.load(key).await
    }

    /// List keys with the given `prefix`.
    pub async fn list(&self, prefix: &str) -> Result<Vec<String>> {
        self.store.list(prefix).await
    }

    /// Delete the memory at `key`.
    pub async fn delete(&self, key: &str) -> Result<()> {
        self.store.delete(key).await
    }

    /// Delete a memory by row id, returning its key on a hit.
    pub async fn forget(&self, id: i64) -> Result<Option<String>> {
        self.store.delete_by_id(id).await
    }

    /// Like [`MemoryOps::list`] but returns full `(id, key, value)`
    /// rows, in whatever order the backend yields.
    pub async fn list_full(&self, prefix: &str) -> Result<Vec<MemoryRecord>> {
        self.store.list_full(prefix).await
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
    async fn no_backend_list_returns_empty() {
        let ops = no_ops();
        assert!(ops.list("pref:").await.unwrap().is_empty());
    }
}
