//! SQLite-backed [`crate::MemoryStore`] — flat string KV over the
//! `memories` table. Shares the [`super::SqliteHandle`] with
//! [`super::SqliteConversationStore`] so both go through the same
//! background writer.

use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;

use crate::MemoryStore;

use super::connection::SqliteHandle;
use super::writer::{WriteCall, WriteOp};

#[derive(Clone)]
pub struct SqliteMemoryStore {
    handle: Arc<SqliteHandle>,
}

impl SqliteMemoryStore {
    pub fn new(handle: Arc<SqliteHandle>) -> Self {
        Self { handle }
    }

    /// Save a memory with provenance — links the row back to the
    /// conversation row that produced it. Returns the row id of the
    /// saved memory so callers can FK an embedding row.
    pub async fn save_with_source(
        &self,
        key: &str,
        value: String,
        source_conversation_id: Option<i64>,
    ) -> Result<i64> {
        let key = key.to_string();
        WriteCall::run(self.handle.writer(), |ack| WriteOp::SaveMemory {
            key,
            value,
            source_conversation_id,
            ack,
        })
        .await
    }
}

#[async_trait]
impl MemoryStore for SqliteMemoryStore {
    async fn save(&self, key: &str, value: String) -> Result<i64> {
        self.save_with_source(key, value, None).await
    }

    async fn load(&self, key: &str) -> Result<Option<String>> {
        let key = key.to_string();
        self.handle
            .conn()
            .call(move |c| {
                let result = c
                    .query_row(
                        "SELECT value FROM memories WHERE key = ?1",
                        rusqlite::params![key],
                        |r| r.get::<_, String>(0),
                    )
                    .map(Some)
                    .or_else(|e| {
                        if matches!(e, rusqlite::Error::QueryReturnedNoRows) {
                            Ok(None)
                        } else {
                            Err(e)
                        }
                    })?;
                Ok(result)
            })
            .await
            .context("memory load")
    }

    async fn delete(&self, key: &str) -> Result<()> {
        let key = key.to_string();
        WriteCall::run(self.handle.writer(), |ack| WriteOp::DeleteMemory {
            key,
            ack,
        })
        .await
    }

    async fn list(&self, prefix: &str) -> Result<Vec<String>> {
        // SQLite `LIKE` with a literal terminator is the simple path —
        // we escape `%` and `_` in the prefix so a key like `pref:%`
        // doesn't match every key starting with `pref:`.
        let escaped = prefix
            .replace('\\', "\\\\")
            .replace('%', "\\%")
            .replace('_', "\\_");
        let pattern = format!("{escaped}%");
        self.handle
            .conn()
            .call(move |c| {
                let mut stmt = c.prepare(
                    "SELECT key FROM memories WHERE key LIKE ?1 ESCAPE '\\' ORDER BY key",
                )?;
                let rows = stmt
                    .query_map(rusqlite::params![pattern], |r| r.get::<_, String>(0))?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .await
            .context("memory list")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::watch;

    async fn fresh() -> (SqliteMemoryStore, tokio::task::JoinHandle<()>) {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("memory.db");
        std::mem::forget(temp);
        let (_tx, rx) = watch::channel(false);
        let (handle, writer) = SqliteHandle::open(&path, rx).await.unwrap();
        (SqliteMemoryStore::new(Arc::new(handle)), writer)
    }

    #[tokio::test]
    async fn save_load_round_trip() {
        let (store, _w) = fresh().await;
        store.save("fact:user.name", "Ben".into()).await.unwrap();
        assert_eq!(
            store.load("fact:user.name").await.unwrap().as_deref(),
            Some("Ben")
        );
    }

    #[tokio::test]
    async fn save_overwrites_existing_value() {
        let (store, _w) = fresh().await;
        store.save("k", "v1".into()).await.unwrap();
        store.save("k", "v2".into()).await.unwrap();
        assert_eq!(store.load("k").await.unwrap().as_deref(), Some("v2"));
    }

    #[tokio::test]
    async fn load_missing_returns_none() {
        let (store, _w) = fresh().await;
        assert_eq!(store.load("nope").await.unwrap(), None);
    }

    #[tokio::test]
    async fn delete_is_silent_for_missing_key() {
        let (store, _w) = fresh().await;
        store.delete("missing").await.unwrap();
    }

    #[tokio::test]
    async fn list_returns_keys_with_prefix_only() {
        let (store, _w) = fresh().await;
        store.save("pref:a", "1".into()).await.unwrap();
        store.save("pref:b", "2".into()).await.unwrap();
        store.save("other:c", "3".into()).await.unwrap();
        let keys = store.list("pref:").await.unwrap();
        assert_eq!(keys, vec!["pref:a", "pref:b"]);
    }

    #[tokio::test]
    async fn list_escapes_like_metacharacters_in_prefix() {
        let (store, _w) = fresh().await;
        store.save("pref:a", "1".into()).await.unwrap();
        store.save("prefXa", "X".into()).await.unwrap(); // would match "pref_" without escape
        let keys = store.list("pref_").await.unwrap();
        // Without escaping, SQLite `_` is a single-char wildcard — the
        // escape we add forces a literal underscore, so neither key
        // matches and we get an empty list.
        assert!(
            keys.is_empty(),
            "expected empty for literal `pref_`: {keys:?}"
        );
    }
}
