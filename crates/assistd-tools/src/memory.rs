//! CRUD façade over [`assistd_memory::MemoryStore`] and
//! [`assistd_memory::TranscriptStore`].

use std::sync::Arc;

use assistd_memory::{MemoryError, MemoryId, MemoryStore, SavedMemory, TranscriptStore};

pub use assistd_memory::MemoryRecord;

type Result<T> = std::result::Result<T, MemoryError>;

/// Result cap applied when a search passes `limit = 0`.
pub const DEFAULT_SEARCH_LIMIT: usize = 50;

/// Combined handle over the key/value store and session transcripts.
/// Cheap to clone.
#[derive(Debug, Clone)]
pub struct MemoryOps {
    pub store: Arc<dyn MemoryStore>,
    pub transcripts: Arc<dyn TranscriptStore>,
}

impl MemoryOps {
    /// Bundle the two stores into one handle.
    pub fn new(store: Arc<dyn MemoryStore>, transcripts: Arc<dyn TranscriptStore>) -> Self {
        Self { store, transcripts }
    }

    /// Save a key/value memory.
    pub async fn save(&self, key: &str, value: String) -> Result<SavedMemory> {
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
    pub async fn forget(&self, id: MemoryId) -> Result<Option<String>> {
        self.store.delete_by_id(id).await
    }

    /// Full rows under `prefix`, in backend order.
    pub async fn list_full(&self, prefix: &str) -> Result<Vec<MemoryRecord>> {
        self.store.list_full(prefix).await
    }
}
