//! Rows and results of the key/value [`MemoryStore`](crate::MemoryStore).

use chrono::{DateTime, Utc};

use crate::MemoryId;

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

/// Result of [`MemoryStore::save`](crate::MemoryStore::save).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SavedMemory {
    pub id: MemoryId,
    /// Whether the stored value differs from the previous one; true for a new key.
    pub value_changed: bool,
}
