//! SQLite-backed [`MemoryStore`] over the `memories` table.

use async_trait::async_trait;
use rusqlite::{OptionalExtension, Row};

use crate::{MemoryId, MemoryRecord, MemoryStore, Result, SavedMemory};

use super::connection::{Erasing, SqliteHandle};
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
            .write_erasing("memory save", move |tx| {
                let previous_value = value_at(tx, &key)?;
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
                let value_changed = previous_value.as_deref() != Some(value.as_str());
                Ok(Erasing {
                    value: SavedMemory { id, value_changed },
                    erased: value_changed && previous_value.is_some(),
                })
            })
            .await
    }

    async fn load(&self, key: &str) -> Result<Option<String>> {
        let key = key.to_string();
        self.handle
            .read("memory load", move |c| value_at(c, &key))
            .await
    }

    async fn delete(&self, key: &str) -> Result<()> {
        let key = key.to_string();
        self.handle
            .write_erasing("memory delete", move |tx| {
                let deleted = tx.execute(
                    "DELETE FROM memories WHERE key = ?1",
                    rusqlite::params![key],
                )?;
                Ok(Erasing {
                    value: (),
                    erased: deleted > 0,
                })
            })
            .await
    }

    async fn delete_by_id(&self, id: MemoryId) -> Result<Option<String>> {
        self.handle
            .write_erasing("memory delete_by_id", move |tx| {
                let key = tx
                    .query_row(
                        "DELETE FROM memories WHERE id = ?1 RETURNING key",
                        rusqlite::params![id],
                        |r| r.get::<_, String>(0),
                    )
                    .optional()?;
                Ok(Erasing {
                    erased: key.is_some(),
                    value: key,
                })
            })
            .await
    }

    async fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let pattern = glob_prefix_pattern(prefix);
        self.handle
            .read("memory list", move |c| {
                c.prepare_cached("SELECT key FROM memories WHERE key GLOB ?1 ORDER BY key")?
                    .query_map(rusqlite::params![pattern], |r| r.get(0))?
                    .collect()
            })
            .await
    }

    async fn list_full(&self, prefix: &str) -> Result<Vec<MemoryRecord>> {
        let pattern = glob_prefix_pattern(prefix);
        self.handle
            .read("memory list_full", move |c| {
                c.prepare_cached(
                    "SELECT id, key, value, created_at, updated_at FROM memories \
                     WHERE key GLOB ?1 ORDER BY key",
                )?
                .query_map(rusqlite::params![pattern], memory_record)?
                .collect()
            })
            .await
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
        let deleted = "zq-forgotten-sentinel-7f3a";
        let overwritten = "zq-replaced-sentinel-91c4";
        store.save("fact:deleted", deleted.into()).await.unwrap();
        store.delete("fact:deleted").await.unwrap();
        store.save("fact:edited", overwritten.into()).await.unwrap();
        store.save("fact:edited", "redacted".into()).await.unwrap();

        for entry in std::fs::read_dir(temp.path()).unwrap() {
            let file = entry.unwrap().path();
            let bytes = std::fs::read(&file).unwrap();
            for sentinel in [deleted, overwritten] {
                assert!(
                    !bytes
                        .windows(sentinel.len())
                        .any(|w| w == sentinel.as_bytes()),
                    "{} still holds {sentinel}",
                    file.display()
                );
            }
        }
    }
}
