//! Document-search configuration: the directories indexed and which of their files are left out.

use std::num::NonZeroU32;

use serde::{Deserialize, Serialize};

use crate::defaults::nz32;
use crate::settings::{DescribeSettings, describe_fields};
use crate::values::AbsoluteDir;

const BYTES_PER_MB: u64 = 1024 * 1024;

/// Which directories are indexed for document search.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct DocumentsConfig {
    /// When `false`, no document is indexed or searched. Needs
    /// `embedding.enabled`.
    pub enabled: bool,
    /// Directories whose files are indexed, each absolute or starting with
    /// `~/`. With symlinks followed, none may be `/`, be or sit under another, or hold
    /// or sit under assistd's config, data, state or cache directory. A missing one is skipped.
    pub roots: Vec<AbsoluteDir>,
    /// Glob patterns matched against a file's path relative to its root; a
    /// file one matches is not indexed.
    pub exclude: Vec<String>,
    /// Largest file indexed, in MB of 1024 × 1024 bytes; a larger one is
    /// skipped.
    pub max_file_mb: NonZeroU32,
}

impl DescribeSettings for DocumentsConfig {
    describe_fields! {
        enabled => ("enabled", "Indexes the folders below so the model can search them"),
        roots => (
            "indexed folders",
            "Folders whose files are indexed; none may hold another"
        ),
        exclude => (
            "excluded files",
            "Glob patterns, matched below each folder, for files left out"
        ),
        max_file_mb => (
            "max file size (MB)",
            "Files larger than this are skipped",
            advanced
        ),
    }
}

impl Default for DocumentsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            roots: Vec::new(),
            exclude: Vec::new(),
            max_file_mb: nz32(20),
        }
    }
}

impl DocumentsConfig {
    /// `max_file_mb` expressed in bytes.
    pub fn max_file_bytes(&self) -> u64 {
        u64::from(self.max_file_mb.get()) * BYTES_PER_MB
    }
}
