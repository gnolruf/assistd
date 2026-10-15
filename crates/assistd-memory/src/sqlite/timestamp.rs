//! Conversion between stored Unix-millisecond timestamps and [`DateTime<Utc>`].

use chrono::{DateTime, Utc};
use rusqlite::Row;
use rusqlite::types::{FromSqlError, Type};

/// The current time in Unix milliseconds, as stored.
pub(super) fn now_millis() -> i64 {
    Utc::now().timestamp_millis()
}

/// The timestamp in column `idx` of `row`.
pub(super) fn timestamp_at(row: &Row<'_>, idx: usize) -> rusqlite::Result<DateTime<Utc>> {
    from_millis(idx, row.get(idx)?)
}

/// The nullable timestamp in column `idx` of `row`.
pub(super) fn optional_timestamp_at(
    row: &Row<'_>,
    idx: usize,
) -> rusqlite::Result<Option<DateTime<Utc>>> {
    row.get::<_, Option<i64>>(idx)?
        .map(|millis| from_millis(idx, millis))
        .transpose()
}

fn from_millis(idx: usize, millis: i64) -> rusqlite::Result<DateTime<Utc>> {
    DateTime::from_timestamp_millis(millis).ok_or_else(|| {
        rusqlite::Error::FromSqlConversionFailure(
            idx,
            Type::Integer,
            Box::new(FromSqlError::OutOfRange(millis)),
        )
    })
}
