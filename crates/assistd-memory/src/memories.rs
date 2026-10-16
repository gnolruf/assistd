//! The key/value [`MemoryStore`] and its no-op fallback.

use std::fmt;

use async_trait::async_trait;
use chrono::{DateTime, Utc};

use crate::{MemoryId, Result};

/// One row from the `memories` table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryRecord {
    pub id: MemoryId,
    pub key: String,
    pub value: String,
    pub created_at: DateTime<Utc>,
    /// When the value last changed; re-saving an identical value leaves it alone.
    pub updated_at: DateTime<Utc>,
}

/// Result of [`MemoryStore::save`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SavedMemory {
    pub id: MemoryId,
    /// Whether the stored value differs from the previous one; true for a new key.
    pub value_changed: bool,
}

/// Persistent string-keyed, string-valued memory.
#[async_trait]
pub trait MemoryStore: fmt::Debug + Send + Sync + 'static {
    /// Persist `value` under `key`, overwriting any existing value.
    /// A subsequent `load(key)` observes the write.
    async fn save(&self, key: &str, value: String) -> Result<SavedMemory>;

    /// Value stored at `key`, or `None` when absent.
    async fn load(&self, key: &str) -> Result<Option<String>>;

    /// Remove `key`. No-op when already absent.
    async fn delete(&self, key: &str) -> Result<()>;

    /// Remove the row with `id`; returns its key, or `None` when no row matched.
    async fn delete_by_id(&self, id: MemoryId) -> Result<Option<String>>;

    /// Keys starting with `prefix` (case-sensitive), in unspecified order.
    async fn list(&self, prefix: &str) -> Result<Vec<String>>;

    /// Full rows whose key starts with `prefix` (case-sensitive), in unspecified order.
    async fn list_full(&self, prefix: &str) -> Result<Vec<MemoryRecord>>;
}

/// No-op store: writes are discarded and reads find nothing. Returned ids are
/// placeholders, meaningful only when passed back to this store.
#[derive(Debug)]
pub struct NoMemoryStore;

#[async_trait]
impl MemoryStore for NoMemoryStore {
    async fn save(&self, key: &str, _value: String) -> Result<SavedMemory> {
        tracing::debug!(target: "assistd::memory", key, "save: no backend configured (drop)");
        Ok(SavedMemory {
            id: MemoryId(0),
            value_changed: false,
        })
    }

    async fn load(&self, _key: &str) -> Result<Option<String>> {
        Ok(None)
    }

    async fn delete(&self, _key: &str) -> Result<()> {
        Ok(())
    }

    async fn delete_by_id(&self, _id: MemoryId) -> Result<Option<String>> {
        Ok(None)
    }

    async fn list(&self, _prefix: &str) -> Result<Vec<String>> {
        Ok(Vec::new())
    }

    async fn list_full(&self, _prefix: &str) -> Result<Vec<MemoryRecord>> {
        Ok(Vec::new())
    }
}
