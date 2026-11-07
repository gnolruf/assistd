//! Typed row identifiers, each a table's integer primary key, the branch-local
//! [`Seq`], and the UUID-backed [`SessionId`].

use std::fmt;
use std::str::FromStr;

use uuid::Uuid;

const SHORT_SESSION_ID_CHARS: usize = 8;

macro_rules! row_id {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(pub i64);

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }
    };
}

row_id!(
    /// Row id in `turns`.
    TurnId
);
row_id!(
    /// Row id in `branches`.
    BranchId
);
row_id!(
    /// Row id in `messages`.
    MessageId
);
row_id!(
    /// Row id in `chunks`.
    ChunkId
);
row_id!(
    /// Row id in `documents`.
    DocumentId
);
row_id!(
    /// Row id in `memories`.
    MemoryId
);
row_id!(
    /// A message's position on its branch, counting up from 0.
    Seq
);

/// Session identifier: a v4 UUID, stable across daemon restarts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SessionId(Uuid);

impl SessionId {
    /// A fresh random session id.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// The UUID's first eight hex digits, the prefix shown to the model and
    /// accepted back when it names a session.
    pub fn short(&self) -> String {
        self.to_string()
            .chars()
            .take(SHORT_SESSION_ID_CHARS)
            .collect()
    }
}

impl Default for SessionId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.hyphenated().fmt(f)
    }
}

impl FromStr for SessionId {
    type Err = uuid::Error;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        Uuid::parse_str(text).map(Self)
    }
}
