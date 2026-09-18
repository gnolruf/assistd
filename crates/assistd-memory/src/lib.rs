#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::print_stdout,
        clippy::print_stderr
    )
)]

//! Persistent memory: the flat key/value [`MemoryStore`] trait, the
//! conversation and semantic stores under [`sqlite`], and no-op
//! fallbacks for when memory is disabled.

pub mod chunking;
pub mod migrations;
pub mod sqlite;

pub use chunking::{ChunkingConfig, chunk_message};
pub use sqlite::{
    BranchId, BranchInfo, ConversationStore, EmbeddingHit, HistoryRow, MemoryHit,
    NoConversationStore, NoSemanticStore, PersistedMessage, PersistedRole, ResumeCandidate,
    SemanticStore, SessionId, SqliteConversationStore, SqliteHandle, SqliteMemoryStore,
    SqliteSemanticStore, TurnId, TurnSummary, UndoOutcome, WriteOp, vector_to_blob,
};

use anyhow::Result;
use async_trait::async_trait;

/// One row from the `memories` table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryRecord {
    pub id: i64,
    pub key: String,
    pub value: String,
}

/// Persistent string-keyed, string-valued memory.
#[async_trait]
pub trait MemoryStore: Send + Sync + 'static {
    /// Persist `value` under `key`, overwriting any existing value.
    /// Returns the row id of the saved memory; the next `load(key)`
    /// from this process observes the write.
    async fn save(&self, key: &str, value: String) -> Result<i64>;

    /// Value stored at `key`, or `None` when absent. `Err` is reserved
    /// for backend failures.
    async fn load(&self, key: &str) -> Result<Option<String>>;

    /// Remove `key`. No-op when already absent.
    async fn delete(&self, key: &str) -> Result<()>;

    /// Remove the row with `id`. Returns the deleted row's key, or
    /// `None` when no row matched.
    async fn delete_by_id(&self, id: i64) -> Result<Option<String>>;

    /// Keys starting with `prefix`, in unspecified order.
    async fn list(&self, prefix: &str) -> Result<Vec<String>>;

    /// Full rows whose key starts with `prefix`, in unspecified order.
    async fn list_full(&self, prefix: &str) -> Result<Vec<MemoryRecord>>;
}

/// No-op fallback used when no persistent backend is configured.
pub struct NoMemoryStore;

#[async_trait]
impl MemoryStore for NoMemoryStore {
    async fn save(&self, key: &str, _value: String) -> Result<i64> {
        tracing::debug!(target: "assistd::memory", key, "save: no backend configured (drop)");
        Ok(0)
    }

    async fn load(&self, _key: &str) -> Result<Option<String>> {
        Ok(None)
    }

    async fn delete(&self, _key: &str) -> Result<()> {
        Ok(())
    }

    async fn delete_by_id(&self, _id: i64) -> Result<Option<String>> {
        Ok(None)
    }

    async fn list(&self, _prefix: &str) -> Result<Vec<String>> {
        Ok(Vec::new())
    }

    async fn list_full(&self, _prefix: &str) -> Result<Vec<MemoryRecord>> {
        Ok(Vec::new())
    }
}

/// Returns the crate version string from `CARGO_PKG_VERSION`.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn no_memory_store_save_and_load_round_trip_returns_none() {
        let store = NoMemoryStore;
        let id = store.save("fact:user.name", "Ben".into()).await.unwrap();
        assert_eq!(id, 0, "no-backend save returns sentinel id 0");
        assert_eq!(store.load("fact:user.name").await.unwrap(), None);
    }

    #[tokio::test]
    async fn no_memory_store_delete_is_silent() {
        let store = NoMemoryStore;
        store.delete("fact:nonexistent").await.unwrap();
    }

    #[tokio::test]
    async fn no_memory_store_delete_by_id_returns_none() {
        let store = NoMemoryStore;
        assert!(store.delete_by_id(42).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn no_memory_store_list_returns_empty() {
        let store = NoMemoryStore;
        assert!(store.list("fact:").await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn no_memory_store_list_full_returns_empty() {
        let store = NoMemoryStore;
        assert!(store.list_full("fact:").await.unwrap().is_empty());
    }

    #[test]
    fn version_is_not_empty() {
        assert!(!version().is_empty());
    }

    /// Compile-only: the trait must stay object-safe.
    #[test]
    fn memory_store_is_object_safe() {
        let _: std::sync::Arc<dyn MemoryStore> = std::sync::Arc::new(NoMemoryStore);
    }
}
