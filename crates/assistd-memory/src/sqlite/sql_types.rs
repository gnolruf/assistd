//! SQLite conversions for the typed columns: row ids, seqs, and role and chunker
//! codes as `INTEGER`, session ids as hyphenated UUID `TEXT`.

use rusqlite::ToSql;
use rusqlite::types::{FromSql, FromSqlError, FromSqlResult, ToSqlOutput, ValueRef};

use crate::chunking::Chunker;
use crate::{
    BranchId, ChunkId, DocumentId, MemoryId, MessageId, PersistedRole, Seq, SessionId, SkippedRead,
    TurnId,
};

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

row_id_sql!(
    TurnId, BranchId, MessageId, ChunkId, DocumentId, MemoryId, Seq
);

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

impl ToSql for PersistedRole {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        let code: i64 = match self {
            PersistedRole::User => 0,
            PersistedRole::Assistant => 1,
            PersistedRole::Tool => 2,
        };
        Ok(ToSqlOutput::from(code))
    }
}

impl FromSql for PersistedRole {
    fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
        match value.as_i64()? {
            0 => Ok(PersistedRole::User),
            1 => Ok(PersistedRole::Assistant),
            2 => Ok(PersistedRole::Tool),
            code => Err(FromSqlError::OutOfRange(code)),
        }
    }
}

impl ToSql for SkippedRead {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        let code: i64 = match self {
            SkippedRead::Unsupported => 0,
            SkippedRead::Binary => 1,
            SkippedRead::TooLarge => 2,
            SkippedRead::NotDecodable => 3,
            SkippedRead::PasswordProtected => 4,
            SkippedRead::NoTextLayer => 5,
            SkippedRead::Unreadable => 6,
            SkippedRead::ExtractorFailed => 7,
            SkippedRead::TimedOut => 8,
        };
        Ok(ToSqlOutput::from(code))
    }
}

impl FromSql for SkippedRead {
    fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
        match value.as_i64()? {
            0 => Ok(SkippedRead::Unsupported),
            1 => Ok(SkippedRead::Binary),
            2 => Ok(SkippedRead::TooLarge),
            3 => Ok(SkippedRead::NotDecodable),
            4 => Ok(SkippedRead::PasswordProtected),
            5 => Ok(SkippedRead::NoTextLayer),
            6 => Ok(SkippedRead::Unreadable),
            7 => Ok(SkippedRead::ExtractorFailed),
            8 => Ok(SkippedRead::TimedOut),
            code => Err(FromSqlError::OutOfRange(code)),
        }
    }
}

impl ToSql for Chunker {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        let code: i64 = match self {
            Chunker::Windows => 0,
            Chunker::Paragraphs => 1,
        };
        Ok(ToSqlOutput::from(code))
    }
}
