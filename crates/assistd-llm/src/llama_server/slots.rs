//! The private directory llama-server saves and restores slot state in.

use std::fs::{self, DirBuilder};
use std::io::{self, ErrorKind};
use std::os::unix::fs::DirBuilderExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const OWNER_ONLY: u32 = 0o700;

/// A mode-0700 directory passed as `--slot-save-path`, emptied before every
/// server spawn since a slot file is only valid for the model and settings
/// that wrote it.
#[derive(Debug, Clone)]
pub struct SlotDir {
    path: Arc<Path>,
}

impl SlotDir {
    /// Use `path` as the slot directory; nothing is touched until [`Self::reset`].
    pub fn new(path: PathBuf) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Replace whatever is at the path with an empty owner-only directory.
    pub fn reset(&self) -> io::Result<()> {
        if let Err(e) = fs::remove_dir_all(&self.path)
            && e.kind() != ErrorKind::NotFound
        {
            return Err(e);
        }
        DirBuilder::new()
            .recursive(true)
            .mode(OWNER_ONLY)
            .create(&self.path)
    }
}
