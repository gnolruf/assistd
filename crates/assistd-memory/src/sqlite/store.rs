//! SQLite-backed [`crate::MemoryStore`] over the `memories` table.

use std::sync::Arc;

use async_trait::async_trait;
use rusqlite::OptionalExtension;

use crate::{MemoryError, MemoryRecord, MemoryStore, Result};

use super::connection::SqliteHandle;
use super::writer::{WriteOp, dispatch_write};

/// SQLite-backed [`crate::MemoryStore`] implementation.
#[derive(Clone)]
pub struct SqliteMemoryStore {
    handle: Arc<SqliteHandle>,
}

impl SqliteMemoryStore {
    pub fn new(handle: Arc<SqliteHandle>) -> Self {
        Self { handle }
    }

    /// Save a memory linked to the conversation row that produced it.
    /// Returns the row id.
    pub async fn save_with_source(
        &self,
        key: &str,
        value: String,
        source_conversation_id: Option<i64>,
    ) -> Result<i64> {
        let key = key.to_string();
        dispatch_write(self.handle.writer(), |ack| WriteOp::SaveMemory {
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
            .call(move |c| -> rusqlite::Result<_> {
                c.query_row(
                    "SELECT value FROM memories WHERE key = ?1",
                    rusqlite::params![key],
                    |r| r.get::<_, String>(0),
                )
                .optional()
            })
            .await
            .map_err(MemoryError::sqlite("memory load"))
    }

    async fn delete(&self, key: &str) -> Result<()> {
        let key = key.to_string();
        dispatch_write(self.handle.writer(), |ack| WriteOp::DeleteMemory {
            key,
            ack,
        })
        .await
    }

    async fn delete_by_id(&self, id: i64) -> Result<Option<String>> {
        dispatch_write(self.handle.writer(), |ack| WriteOp::DeleteMemoryById {
            id,
            ack,
        })
        .await
    }

    async fn list(&self, prefix: &str) -> Result<Vec<String>> {
        // Escape LIKE metacharacters so `pref:%` matches literally.
        let escaped = prefix
            .replace('\\', "\\\\")
            .replace('%', "\\%")
            .replace('_', "\\_");
        let pattern = format!("{escaped}%");
        self.handle
            .conn()
            .call(move |c| -> rusqlite::Result<_> {
                let mut stmt = c.prepare(
                    "SELECT key FROM memories WHERE key LIKE ?1 ESCAPE '\\' ORDER BY key",
                )?;
                let rows = stmt
                    .query_map(rusqlite::params![pattern], |r| r.get::<_, String>(0))?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .await
            .map_err(MemoryError::sqlite("memory list"))
    }

    async fn list_full(&self, prefix: &str) -> Result<Vec<MemoryRecord>> {
        let escaped = prefix
            .replace('\\', "\\\\")
            .replace('%', "\\%")
            .replace('_', "\\_");
        let pattern = format!("{escaped}%");
        self.handle
            .conn()
            .call(move |c| -> rusqlite::Result<_> {
                let mut stmt = c.prepare(
                    "SELECT id, key, value FROM memories \
                     WHERE key LIKE ?1 ESCAPE '\\' ORDER BY key",
                )?;
                let rows = stmt
                    .query_map(rusqlite::params![pattern], |r| {
                        Ok(MemoryRecord {
                            id: r.get(0)?,
                            key: r.get(1)?,
                            value: r.get(2)?,
                        })
                    })?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .await
            .map_err(MemoryError::sqlite("memory list_full"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::watch;

    /// The guard keeps the database directory and the writer's shutdown
    /// sender alive for the test's duration.
    async fn fresh() -> (SqliteMemoryStore, (tempfile::TempDir, watch::Sender<bool>)) {
        let temp = tempfile::tempdir().unwrap();
        let (tx, rx) = watch::channel(false);
        let (handle, _writer) = SqliteHandle::open(&temp.path().join("memory.db"), rx)
            .await
            .unwrap();
        (SqliteMemoryStore::new(Arc::new(handle)), (temp, tx))
    }

    #[tokio::test]
    async fn save_overwrites_in_place_and_load_sees_latest_value() {
        let (store, _guard) = fresh().await;
        let id = store.save("fact:user.name", "Ben".into()).await.unwrap();
        assert_eq!(
            store.load("fact:user.name").await.unwrap().as_deref(),
            Some("Ben")
        );

        let resaved = store
            .save("fact:user.name", "Benjamin".into())
            .await
            .unwrap();
        assert_eq!(resaved, id, "upsert must keep the row id");
        assert_eq!(
            store.load("fact:user.name").await.unwrap().as_deref(),
            Some("Benjamin")
        );
    }

    #[tokio::test]
    async fn load_missing_returns_none() {
        let (store, _guard) = fresh().await;
        assert_eq!(store.load("nope").await.unwrap(), None);
    }

    #[tokio::test]
    async fn delete_removes_key_and_is_silent_when_absent() {
        let (store, _guard) = fresh().await;
        store.save("k", "v".into()).await.unwrap();
        store.delete("k").await.unwrap();
        assert_eq!(store.load("k").await.unwrap(), None);
        store.delete("k").await.unwrap();
    }

    #[tokio::test]
    async fn list_returns_keys_with_prefix_only() {
        let (store, _guard) = fresh().await;
        store.save("pref:a", "1".into()).await.unwrap();
        store.save("pref:b", "2".into()).await.unwrap();
        store.save("other:c", "3".into()).await.unwrap();
        let keys = store.list("pref:").await.unwrap();
        assert_eq!(keys, ["pref:a", "pref:b"]);
    }

    #[tokio::test]
    async fn list_full_returns_id_key_value_in_lex_order() {
        let (store, _guard) = fresh().await;
        let b = store.save("pref:b", "two".into()).await.unwrap();
        let a = store.save("pref:a", "one".into()).await.unwrap();
        store.save("other:c", "three".into()).await.unwrap();
        let rows = store.list_full("pref:").await.unwrap();
        assert_eq!(
            rows,
            [
                MemoryRecord {
                    id: a,
                    key: "pref:a".into(),
                    value: "one".into(),
                },
                MemoryRecord {
                    id: b,
                    key: "pref:b".into(),
                    value: "two".into(),
                },
            ]
        );
    }

    #[tokio::test]
    async fn list_full_empty_prefix_returns_all_rows() {
        let (store, _guard) = fresh().await;
        store.save("b", "2".into()).await.unwrap();
        store.save("a", "1".into()).await.unwrap();
        let keys: Vec<String> = store
            .list_full("")
            .await
            .unwrap()
            .into_iter()
            .map(|r| r.key)
            .collect();
        assert_eq!(keys, ["a", "b"]);
    }

    #[tokio::test]
    async fn delete_by_id_returns_key_on_hit_and_none_on_miss() {
        let (store, _guard) = fresh().await;
        let id = store.save("fact:user.name", "Ben".into()).await.unwrap();
        assert_eq!(
            store.delete_by_id(id).await.unwrap().as_deref(),
            Some("fact:user.name")
        );
        assert_eq!(store.load("fact:user.name").await.unwrap(), None);
        assert_eq!(store.delete_by_id(id).await.unwrap(), None);
    }

    #[tokio::test]
    async fn prefix_like_metacharacters_match_literally() {
        let (store, _guard) = fresh().await;
        store.save("pref:a", "1".into()).await.unwrap();
        store.save("prefXa", "X".into()).await.unwrap();
        store.save("100%:a", "p".into()).await.unwrap();
        store.save("100x:a", "q".into()).await.unwrap();
        assert_eq!(store.list("pref_").await.unwrap(), Vec::<String>::new());
        assert_eq!(store.list("100%").await.unwrap(), ["100%:a"]);
        let full: Vec<String> = store
            .list_full("100%")
            .await
            .unwrap()
            .into_iter()
            .map(|r| r.key)
            .collect();
        assert_eq!(full, ["100%:a"]);
    }
}
