//! Search on the wire: the kinds of stored text a search ranks, and where a hit's
//! passage is.

use std::fmt;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::event::Role;

/// One kind of stored text a search ranks.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SearchKind {
    Sessions,
    Memories,
    Documents,
}

impl SearchKind {
    pub const ALL: [Self; 3] = [Self::Sessions, Self::Memories, Self::Documents];

    /// The wire spelling, which is also what `--in` takes.
    pub fn name(self) -> &'static str {
        match self {
            Self::Sessions => "sessions",
            Self::Memories => "memories",
            Self::Documents => "documents",
        }
    }

    /// The kind `name` spells; `None` when it spells none.
    pub fn named(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.name() == name)
    }
}

impl fmt::Display for SearchKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Where the passage of a search hit is, by the kind of text that holds it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SearchPlace {
    /// A message of a saved session.
    Session {
        session_id: String,
        title: String,
        timestamp: DateTime<Utc>,
        role: Role,
    },
    /// A saved memory, whose value is the whole passage.
    Memory { key: String },
    /// An indexed document. `page` counts from one and the byte range is within that
    /// page's text.
    Document {
        path: String,
        page: u64,
        page_count: u64,
        byte_start: u64,
        byte_end: u64,
        indexed_at: DateTime<Utc>,
    },
}
