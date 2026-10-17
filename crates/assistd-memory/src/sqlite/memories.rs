//! SQLite-backed [`MemoryStore`] over the `memories` table.

use async_trait::async_trait;
use rusqlite::{OptionalExtension, Row};

use crate::{MemoryError, MemoryId, MemoryRecord, MemoryStore, Result, SavedMemory};

use super::connection::{SqliteHandle, truncate_wal};
use super::timestamp::{now_millis, timestamp_at};

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
    /// A changed value bumps `updated_at`, drops the row's embedding (by schema
    /// trigger), and truncates the WAL so the old value leaves no trace.
    async fn save(&self, key: &str, value: String) -> Result<SavedMemory> {
        let key = key.to_string();
        let now = now_millis();
        self.handle
            .writer()
            .call(move |c| -> rusqlite::Result<_> {
                let tx = c.transaction()?;
                let previous_value = value_at(&tx, &key)?;
                let id: MemoryId = tx.query_row(
                    "INSERT INTO memories (key, value, created_at, updated_at)
                     VALUES (?1, ?2, ?3, ?3)
                     ON CONFLICT(key) DO UPDATE SET
                         value = excluded.value,
                         updated_at = CASE WHEN value = excluded.value
                                           THEN updated_at
                                           ELSE MAX(excluded.updated_at, created_at) END
                     RETURNING id",
                    rusqlite::params![key, value, now],
                    |r| r.get(0),
                )?;
                tx.commit()?;
                let value_changed = previous_value.as_deref() != Some(value.as_str());
                if value_changed && previous_value.is_some() {
                    truncate_wal(c);
                }
                Ok(SavedMemory { id, value_changed })
            })
            .await
            .map_err(MemoryError::sqlite("memory save"))
    }

    async fn load(&self, key: &str) -> Result<Option<String>> {
        let key = key.to_string();
        self.handle
            .reader()
            .call(move |c| value_at(c, &key))
            .await
            .map_err(MemoryError::sqlite("memory load"))
    }

    async fn delete(&self, key: &str) -> Result<()> {
        let key = key.to_string();
        self.handle
            .writer()
            .call(move |c| -> rusqlite::Result<_> {
                let deleted = c.execute(
                    "DELETE FROM memories WHERE key = ?1",
                    rusqlite::params![key],
                )?;
                if deleted > 0 {
                    truncate_wal(c);
                }
                Ok(())
            })
            .await
            .map_err(MemoryError::sqlite("memory delete"))
    }

    async fn delete_by_id(&self, id: MemoryId) -> Result<Option<String>> {
        self.handle
            .writer()
            .call(move |c| -> rusqlite::Result<_> {
                let key = c
                    .query_row(
                        "DELETE FROM memories WHERE id = ?1 RETURNING key",
                        rusqlite::params![id],
                        |r| r.get::<_, String>(0),
                    )
                    .optional()?;
                if key.is_some() {
                    truncate_wal(c);
                }
                Ok(key)
            })
            .await
            .map_err(MemoryError::sqlite("memory delete_by_id"))
    }

    async fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let pattern = glob_prefix_pattern(prefix);
        self.handle
            .reader()
            .call(move |c| -> rusqlite::Result<_> {
                let mut stmt =
                    c.prepare_cached("SELECT key FROM memories WHERE key GLOB ?1 ORDER BY key")?;
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
                let mut stmt = c.prepare_cached(
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

fn value_at(c: &rusqlite::Connection, key: &str) -> rusqlite::Result<Option<String>> {
    c.query_row(
        "SELECT value FROM memories WHERE key = ?1",
        rusqlite::params![key],
        |r| r.get(0),
    )
    .optional()
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
    use super::*;

    /// The returned temp dir must outlive the store.
    async fn fresh() -> (SqliteMemoryStore, tempfile::TempDir) {
        let temp = tempfile::tempdir().unwrap();
        let handle = SqliteHandle::open(&temp.path().join("memory.db"))
            .await
            .unwrap();
        (SqliteMemoryStore::new(handle), temp)
    }

    #[tokio::test]
    async fn save_overwrites_in_place_and_load_sees_latest_value() {
        let (store, _temp) = fresh().await;
        let id = store.save("fact:user.name", "Ben".into()).await.unwrap().id;
        assert_eq!(
            store.load("fact:user.name").await.unwrap().as_deref(),
            Some("Ben")
        );

        let resaved = store
            .save("fact:user.name", "Benjamin".into())
            .await
            .unwrap()
            .id;
        assert_eq!(resaved, id, "upsert must keep the row id");
        assert_eq!(
            store.load("fact:user.name").await.unwrap().as_deref(),
            Some("Benjamin")
        );
    }

    #[tokio::test]
    async fn prefix_matches_literally_and_case_sensitively() {
        let (store, _temp) = fresh().await;
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
    async fn deleted_and_overwritten_memories_leave_no_trace_in_database_files() {
        let (store, temp) = fresh().await;
        let deleted = "zq-forgotten-secret-7f3a";
        let overwritten = "zq-replaced-secret-91c4";
        store.save("fact:deleted", deleted.into()).await.unwrap();
        store.delete("fact:deleted").await.unwrap();
        store.save("fact:edited", overwritten.into()).await.unwrap();
        store.save("fact:edited", "redacted".into()).await.unwrap();

        for entry in std::fs::read_dir(temp.path()).unwrap() {
            let file = entry.unwrap().path();
            let bytes = std::fs::read(&file).unwrap();
            for secret in [deleted, overwritten] {
                assert!(
                    !bytes.windows(secret.len()).any(|w| w == secret.as_bytes()),
                    "{} still holds {secret}",
                    file.display()
                );
            }
        }
    }
}
