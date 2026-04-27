#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::print_stdout,
        clippy::print_stderr
    )
)]

//! Persistent-memory subsystem trait and a no-op placeholder.
//!
//! Milestone 4 will land a SQLite-backed concrete implementation; this
//! crate exists now so the trait shape and the [`AppState`] wiring are
//! settled before the first storage code is written. Locking the
//! interface down up front avoids retrofitting every call site once
//! the real backend arrives.
//!
//! # Trait shape
//!
//! [`MemoryStore`] is a small key/value surface scoped to **string
//! values keyed by string keys**. Richer types (e.g. embeddings,
//! conversation snapshots) are intentionally *not* on this trait —
//! they will live in higher-level adapters that serialize to/from this
//! flat surface so the storage backend can stay simple. The methods
//! mirror the four operations the agent loop will need at minimum:
//! `save` to commit a fact, `load` to recall one, `delete` to forget,
//! and `list` to enumerate keys under a prefix (used for namespaced
//! categories like `pref:`, `fact:`, `summary:`).
//!
//! # The `NoMemoryStore` placeholder
//!
//! Mirrors the shape of `assistd_voice::NoVoiceOutput`: every method
//! is a successful no-op so the daemon's startup path can wire a
//! `MemoryStore` unconditionally and degrade gracefully on a build or
//! environment where persistent storage isn't configured. `load`
//! returns `Ok(None)`, `list` returns `Ok(vec![])`, and `save` /
//! `delete` succeed silently — so an agent calling them in this
//! configuration just behaves as if it has no long-term memory.

use anyhow::Result;
use async_trait::async_trait;

/// Persistent key/value memory accessible to the daemon and the
/// agent loop. Implementors must be `Send + Sync + 'static` because
/// `AppState` holds them as `Arc<dyn MemoryStore>`.
#[async_trait]
pub trait MemoryStore: Send + Sync + 'static {
    /// Persist `value` under `key`. Overwrites any existing value at
    /// the same key. Concrete implementations decide their own
    /// durability semantics (write-through vs. periodic flush) — the
    /// trait makes no guarantees beyond "the next `load(key)` from
    /// this process should observe the write".
    async fn save(&self, key: &str, value: String) -> Result<()>;

    /// Read the value previously stored at `key`. Returns `Ok(None)`
    /// when the key is absent (distinct from an `Err` path, which is
    /// reserved for backend failures).
    async fn load(&self, key: &str) -> Result<Option<String>>;

    /// Remove `key`. No-op when the key is already absent — this
    /// matches the agent-loop usage pattern of "forget X if you
    /// remember it" and saves a probe-then-delete round trip.
    async fn delete(&self, key: &str) -> Result<()>;

    /// Enumerate keys whose name starts with `prefix`. Order is
    /// unspecified. Used for namespaced categories (`pref:`, `fact:`,
    /// `summary:`) so the agent can list "all preferences" without
    /// scanning the whole store.
    async fn list(&self, prefix: &str) -> Result<Vec<String>>;
}

/// Successful-no-op fallback used when no persistent backend is
/// configured. Wired unconditionally in `AppState` so the agent loop
/// can call `MemoryStore` methods without checking for an `Option`.
pub struct NoMemoryStore;

#[async_trait]
impl MemoryStore for NoMemoryStore {
    async fn save(&self, key: &str, _value: String) -> Result<()> {
        tracing::debug!(target: "assistd::memory", key, "save: no backend configured (drop)");
        Ok(())
    }

    async fn load(&self, _key: &str) -> Result<Option<String>> {
        Ok(None)
    }

    async fn delete(&self, _key: &str) -> Result<()> {
        Ok(())
    }

    async fn list(&self, _prefix: &str) -> Result<Vec<String>> {
        Ok(Vec::new())
    }
}

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn no_memory_store_save_and_load_round_trip_returns_none() {
        // The placeholder accepts saves silently and reports the key
        // as absent on load. This is the contract the agent loop
        // depends on for the no-backend configuration.
        let store = NoMemoryStore;
        store.save("fact:user.name", "Ben".into()).await.unwrap();
        assert_eq!(store.load("fact:user.name").await.unwrap(), None);
    }

    #[tokio::test]
    async fn no_memory_store_delete_is_silent() {
        let store = NoMemoryStore;
        store.delete("fact:nonexistent").await.unwrap();
    }

    #[tokio::test]
    async fn no_memory_store_list_returns_empty() {
        let store = NoMemoryStore;
        assert!(store.list("fact:").await.unwrap().is_empty());
    }

    #[test]
    fn version_is_not_empty() {
        assert!(!version().is_empty());
    }

    /// Compile-only: prove the trait is object-safe by holding it
    /// behind `Arc<dyn MemoryStore>`. AppState relies on this shape.
    #[test]
    fn memory_store_is_object_safe() {
        let _: std::sync::Arc<dyn MemoryStore> = std::sync::Arc::new(NoMemoryStore);
    }
}
