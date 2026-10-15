//! SQLite-backed [`MemoryStore`] over the `memories` table.

use async_trait::async_trait;
use rusqlite::{OptionalExtension, Row};

use crate::{MemoryError, MemoryRecord, MemoryStore, Result};

use super::connection::SqliteHandle;
use super::ids::MemoryId;
use super::timestamp::timestamp_at;
use super::writer::{WriteOp, dispatch_write};

/// SQLite-backed [`MemoryStore`].
#[derive(Debug, Clone)]
pub struct SqliteMemoryStore {
    handle: SqliteHandle,
}

impl SqliteMemoryStore {
    /// Store over the shared database handle.
    pub fn new(handle: SqliteHandle) -> Self {
        Self { handle }
    }
}

#[async_trait]
impl MemoryStore for SqliteMemoryStore {
    async fn save(&self, key: &str, value: String) -> Result<MemoryId> {
        let key = key.to_string();
        dispatch_write(self.handle.writer(), |ack| WriteOp::SaveMemory {
            key,
            value,
            ack,
        })
        .await
    }

    async fn load(&self, key: &str) -> Result<Option<String>> {
        let key = key.to_string();
        self.handle
            .reader()
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

    async fn delete_by_id(&self, id: MemoryId) -> Result<Option<String>> {
        dispatch_write(self.handle.writer(), |ack| WriteOp::DeleteMemoryById {
            id,
            ack,
        })
        .await
    }

    async fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let pattern = glob_prefix_pattern(prefix);
        self.handle
            .reader()
            .call(move |c| -> rusqlite::Result<_> {
                let mut stmt =
                    c.prepare("SELECT key FROM memories WHERE key GLOB ?1 ORDER BY key")?;
                let rows = stmt
                    .query_map(rusqlite::params![pattern], |r| r.get::<_, String>(0))?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .await
            .map_err(MemoryError::sqlite("memory list"))
    }

    async fn list_full(&self, prefix: &str) -> Result<Vec<MemoryRecord>> {
        let pattern = glob_prefix_pattern(prefix);
        self.handle
            .reader()
            .call(move |c| -> rusqlite::Result<_> {
                let mut stmt = c.prepare(
                    "SELECT id, key, value, created_at, updated_at FROM memories \
                     WHERE key GLOB ?1 ORDER BY key",
                )?;
                let rows = stmt
                    .query_map(rusqlite::params![pattern], memory_record)?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .await
            .map_err(MemoryError::sqlite("memory list_full"))
    }
}

fn memory_record(row: &Row<'_>) -> rusqlite::Result<MemoryRecord> {
    Ok(MemoryRecord {
        id: row.get(0)?,
        key: row.get(1)?,
        value: row.get(2)?,
        created_at: timestamp_at(row, 3)?,
        updated_at: timestamp_at(row, 4)?,
    })
}

/// A case-sensitive `GLOB` pattern matching values that start with `prefix` literally.
fn glob_prefix_pattern(prefix: &str) -> String {
    let mut pattern = String::with_capacity(prefix.len() + 1);
    for ch in prefix.chars() {
        match ch {
            '*' | '?' | '[' => {
                pattern.push('[');
                pattern.push(ch);
                pattern.push(']');
            }
            _ => pattern.push(ch),
        }
    }
    pattern.push('*');
    pattern
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
        (SqliteMemoryStore::new(handle), (temp, tx))
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
    async fn prefix_matches_literally_and_case_sensitively() {
        let (store, _guard) = fresh().await;
        for key in [
            "pref:a", "prefXa", "Pref:b", "a*b:1", "axb:1", "q?:1", "qx:1", "[x]:1",
        ] {
            store.save(key, "v".into()).await.unwrap();
        }
        assert_eq!(store.list("pref:").await.unwrap(), ["pref:a"]);
        assert_eq!(store.list("a*").await.unwrap(), ["a*b:1"]);
        assert_eq!(store.list("q?").await.unwrap(), ["q?:1"]);
        assert_eq!(store.list("[x]").await.unwrap(), ["[x]:1"]);
        let full: Vec<String> = store
            .list_full("Pref")
            .await
            .unwrap()
            .into_iter()
            .map(|r| r.key)
            .collect();
        assert_eq!(full, ["Pref:b"]);
    }

    #[tokio::test]
    async fn deleted_memory_leaves_no_trace_in_database_files() {
        let (store, guard) = fresh().await;
        let secret = "zq-forgotten-secret-7f3a";
        store.save("fact:secret", secret.into()).await.unwrap();
        store.delete("fact:secret").await.unwrap();

        for entry in std::fs::read_dir(guard.0.path()).unwrap() {
            let file = entry.unwrap().path();
            let bytes = std::fs::read(&file).unwrap();
            assert!(
                !bytes.windows(secret.len()).any(|w| w == secret.as_bytes()),
                "{} still holds the deleted value",
                file.display()
            );
        }
    }
}
