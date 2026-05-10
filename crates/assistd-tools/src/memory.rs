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
use assistd_memory::{ConversationStore, MemoryStore, TurnSummary};

pub use assistd_memory::MemoryRecord;

/// Default cap on `recent_turns` / semantic-search result sizes when
/// the IPC client asks for `limit = 0` (the wire default for variants
/// that omit a cap). Big enough to be useful for casual inspection;
/// the LLM-facing path passes an explicit smaller limit.
pub const DEFAULT_SEARCH_LIMIT: usize = 50;

/// Combined CRUD handle over both the flat KV store and the richer
/// conversation history. Cheap to clone (just two `Arc`s).
#[derive(Clone)]
pub struct MemoryOps {
    pub store: Arc<dyn MemoryStore>,
    pub conversations: Arc<dyn ConversationStore>,
}

impl MemoryOps {
    /// Construct a `MemoryOps` from the provided store and conversation backends.
    pub fn new(store: Arc<dyn MemoryStore>, conversations: Arc<dyn ConversationStore>) -> Self {
        Self {
            store,
            conversations,
        }
    }

    /// Save a key/value memory. Returns the row id of the saved memory
    /// so the caller (typically `RememberTool`) can enqueue an embed
    /// job that FKs the new row. The IPC `MemorySave` handler discards
    /// the id; the LLM-callable `remember` tool consumes it.
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

    /// Delete a memory by row id. Returns `Some(key)` of the deleted
    /// row on hit, `None` when no row matched. Used by the IPC
    /// `MemoryForget` handler so the CLI can echo the key back to the
    /// user (`forgot id=N key=...`) and distinguish hit/miss without
    /// a second probe.
    pub async fn forget(&self, id: i64) -> Result<Option<String>> {
        self.store.delete_by_id(id).await
    }

    /// Like [`MemoryOps::list`] but returns full `(id, key, value)`
    /// rows in one round trip — used by the `assistd memory list` CLI.
    /// Order is whatever the backend yields (lexicographic by key for
    /// the SQLite impl).
    pub async fn list_full(&self, prefix: &str) -> Result<Vec<MemoryRecord>> {
        self.store.list_full(prefix).await
    }

    /// Return recent conversation turns, up to `limit` (or [`DEFAULT_SEARCH_LIMIT`] when `limit` is 0).
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
    async fn no_backend_list_and_recent_turns_return_empty() {
        let ops = no_ops();
        assert!(ops.list("pref:").await.unwrap().is_empty());
        assert!(ops.recent_turns(0).await.unwrap().is_empty());
    }
}
