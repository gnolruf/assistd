//! What a tool call touched, reported by the call and stored beside its result.

use std::fmt;
use std::path::PathBuf;

use assistd_utils::text::flatten_line_breaks;
use serde::{Deserialize, Serialize};

/// Most characters of a fetched page's title that are kept.
const MAX_TITLE_CHARS: usize = 200;

/// One thing a tool call touched. Paths are absolute; a URL and a title are the
/// page's own choice and so untrusted text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Access {
    /// A file whose content the call read.
    Read(PathBuf),
    /// A file the call wrote.
    Wrote(PathBuf),
    /// A page the call fetched: the URL the response came from, after redirects, and the
    /// page's title on one line of at most 200 characters.
    Fetched { url: String, title: Option<String> },
    /// A past session the call looked into, by its id.
    Recalled(String),
    /// A memory the call saved, by its key.
    Saved(String),
}

impl Access {
    /// A fetch answered from `url` by a page titled `title`, which is flattened to one
    /// line and cut; a blank title is none.
    pub fn fetched(url: impl Into<String>, title: Option<&str>) -> Self {
        let title = title
            .map(|title| {
                let flat = flatten_line_breaks(title);
                flat.trim()
                    .chars()
                    .take(MAX_TITLE_CHARS)
                    .collect::<String>()
            })
            .filter(|title| !title.is_empty());
        Self::Fetched {
            url: url.into(),
            title,
        }
    }
}

/// What a tool call did to a file or a page, for the accesses a search ranks.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AccessAction {
    Read,
    Wrote,
    Fetched,
}

impl fmt::Display for AccessAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Read => "read",
            Self::Wrote => "wrote",
            Self::Fetched => "fetched",
        })
    }
}
