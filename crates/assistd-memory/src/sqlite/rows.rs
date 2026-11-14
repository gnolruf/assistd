//! Row collection that leaves out rows whose stored values no longer decode.

use rusqlite::{Row, Rows};
use tracing::warn;

/// Decode each of `rows` with `decode`, leaving out those that fail with a warning naming
/// `table` and the integer row id in column 0. A failure to step `rows` is still an error.
pub(super) fn decodable_rows<T>(
    mut rows: Rows<'_>,
    table: &'static str,
    decode: impl Fn(&Row<'_>) -> rusqlite::Result<T>,
) -> rusqlite::Result<Vec<T>> {
    let mut decoded = Vec::new();
    while let Some(row) = rows.next()? {
        let id: i64 = row.get(0)?;
        match decode(row) {
            Ok(value) => decoded.push(value),
            Err(error) => warn!(
                target: "assistd::memory",
                table,
                id,
                %error,
                "skipping a stored row that does not decode"
            ),
        }
    }
    Ok(decoded)
}
