//! The text the OCR model read from pages, kept in `vectors.db` under the hash of each
//! page's file, so that it outlasts a rebuilt `index.db` and a file that moves.

use rusqlite::{Connection, OptionalExtension, Transaction};

use crate::{DocumentId, FileHash, PageReader};

/// The text kept as read from page `page`, counted from zero, of the file with hash
/// `file`; `None` when that page was not read.
pub(super) fn kept_reading(
    conn: &Connection,
    file: FileHash,
    page: usize,
) -> rusqlite::Result<Option<String>> {
    conn.prepare_cached("SELECT text FROM page_readings WHERE file_hash = ?1 AND page = ?2")?
        .query_row(rusqlite::params![file, page], |r| r.get(0))
        .optional()
}

/// Keep `text` as read from page `page`, counted from zero, of the file `document`
/// was stored from.
pub(super) fn keep_reading(
    conn: &Connection,
    document: DocumentId,
    page: usize,
    text: &str,
) -> rusqlite::Result<()> {
    conn.prepare_cached(
        "INSERT INTO page_readings (file_hash, page, text)
         SELECT d.file_hash, ?2, ?3 FROM documents d
         WHERE d.id = ?1 AND d.file_hash IS NOT NULL
         ON CONFLICT(file_hash, page) DO UPDATE SET text = excluded.text",
    )?
    .execute(rusqlite::params![document, page, text])?;
    Ok(())
}

/// Delete the readings of each file the transaction removed a document of, unless a
/// document is still stored from those bytes, and forget the removals.
pub(super) fn delete_readings_of_removed_files(tx: &Transaction<'_>) -> rusqlite::Result<()> {
    let removed_any: bool = tx
        .prepare_cached("SELECT EXISTS (SELECT 1 FROM removed_file_hashes)")?
        .query_row([], |r| r.get(0))?;
    if !removed_any {
        return Ok(());
    }
    tx.prepare_cached(
        "DELETE FROM page_readings
         WHERE file_hash IN (SELECT hash FROM removed_file_hashes)
           AND NOT EXISTS (SELECT 1 FROM documents d
                           WHERE d.file_hash = page_readings.file_hash)",
    )?
    .execute([])?;
    tx.prepare_cached("DELETE FROM removed_file_hashes")?
        .execute([])?;
    Ok(())
}

/// Delete every reading of a file no document is stored from, returning how many.
/// Every root must have been walked: until then, a reading waits for its document.
pub(super) fn delete_unheld_readings(conn: &Connection) -> rusqlite::Result<usize> {
    conn.prepare_cached(
        "DELETE FROM page_readings
         WHERE NOT EXISTS (SELECT 1 FROM documents d
                           WHERE d.file_hash = page_readings.file_hash)",
    )?
    .execute([])
}

/// Record `reader` as the one every kept reading is by. When another was recorded,
/// drop every reading and return that one.
pub(super) fn adopt_reader(
    tx: &Transaction<'_>,
    reader: &PageReader,
) -> rusqlite::Result<Option<PageReader>> {
    let recorded = tx
        .query_row("SELECT model, prompt FROM page_reader", [], |row| {
            Ok(PageReader {
                model: row.get(0)?,
                prompt: row.get(1)?,
            })
        })
        .optional()?;
    if recorded.as_ref() == Some(reader) {
        return Ok(None);
    }
    tx.execute("DELETE FROM page_readings", [])?;
    tx.execute(
        "INSERT INTO page_reader (id, model, prompt) VALUES (0, ?1, ?2)
         ON CONFLICT(id) DO UPDATE SET model = excluded.model, prompt = excluded.prompt",
        rusqlite::params![reader.model, reader.prompt],
    )?;
    Ok(recorded)
}
