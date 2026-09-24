//! CRUD façade over [`assistd_memory::MemoryStore`] and
//! [`assistd_memory::ConversationStore`].

use std::sync::Arc;

use assistd_memory::{ConversationStore, MemoryError, MemoryStore};

pub use assistd_memory::MemoryRecord;

type Result<T> = std::result::Result<T, MemoryError>;

/// Result cap applied when a search passes `limit = 0`.
pub const DEFAULT_SEARCH_LIMIT: usize = 50;

/// Combined handle over the key/value store and conversation history.
/// Cheap to clone.
#[derive(Clone)]
pub struct MemoryOps {
    pub store: Arc<dyn MemoryStore>,
    pub conversations: Arc<dyn ConversationStore>,
}

impl MemoryOps {
    /// Bundle the two stores into one handle.
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
