//! SQLite-backed [`MemoryStore`] over the `memories` table.

use std::sync::Arc;

use async_trait::async_trait;
use rusqlite::OptionalExtension;

use crate::{MemoryError, MemoryRecord, MemoryStore, Result};

use super::connection::SqliteHandle;
use super::writer::{WriteOp, dispatch_write};

/// SQLite-backed [`MemoryStore`].
#[derive(Debug, Clone)]
pub struct SqliteMemoryStore {
    handle: Arc<SqliteHandle>,
}

impl SqliteMemoryStore {
    /// Store over the shared database handle.
    pub fn new(handle: Arc<SqliteHandle>) -> Self {
        Self { handle }
    }

    /// Save a memory linked to the conversation row that produced it; returns the row id.
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
        let pattern = like_prefix_pattern(prefix);
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
        let pattern = like_prefix_pattern(prefix);
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

/// A `LIKE` pattern (escape char `\`) matching keys that start with `prefix` literally.
fn like_prefix_pattern(prefix: &str) -> String {
    let escaped = prefix
        .replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_");
    format!("{escaped}%")
}

#[cfg(test)]
mod tests {
    use tokio::sync::watch;

    use super::*;

    /// The returned guard keeps the temp dir and the writer's shutdown sender alive.
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
