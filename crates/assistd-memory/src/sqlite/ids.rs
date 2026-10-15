//! SQLite conversions for the typed ids: row ids as `INTEGER`, session ids as
//! hyphenated UUID `TEXT`.

use rusqlite::ToSql;
use rusqlite::types::{FromSql, FromSqlError, FromSqlResult, ToSqlOutput, ValueRef};

use crate::{BranchId, ChunkId, MemoryId, MessageId, SessionId, TurnId};

macro_rules! row_id_sql {
    ($($name:ident),+) => {$(
        impl ToSql for $name {
            fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
                Ok(ToSqlOutput::from(self.0))
            }
        }

        impl FromSql for $name {
            fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
                value.as_i64().map(Self)
            }
        }
    )+};
}

row_id_sql!(TurnId, BranchId, MessageId, ChunkId, MemoryId);

impl ToSql for SessionId {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        Ok(ToSqlOutput::from(self.to_string()))
    }
}

impl FromSql for SessionId {
    fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
        value
            .as_str()?
            .parse()
            .map_err(|e| FromSqlError::Other(Box::new(e)))
    }
}
