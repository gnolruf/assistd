//! The scratch directory `bash` and the in-process commands share: created
//! owner-only at startup, with entries past their retention removed.

use std::fs::{DirBuilder, Permissions};
use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use tracing::warn;

use crate::{BuildToolsError, OVERFLOW_DIR_MODE, ToolsScratchConfig};

const SECS_PER_DAY: u64 = 24 * 60 * 60;

/// Create `scratch.dir` owner-only, remove what has not changed in
/// `scratch.retention_days`, and return the directory's canonical path.
///
/// # Errors
/// [`BuildToolsError::CreateScratchDir`] when the directory cannot be
/// created, restricted or resolved; failures to remove entries are logged.
pub(crate) fn prepare(scratch: &ToolsScratchConfig) -> Result<PathBuf, BuildToolsError> {
    let scratch_dir = scratch.dir.as_path();
    let create_error = |source| BuildToolsError::CreateScratchDir {
        path: scratch_dir.to_path_buf(),
        source,
    };
    DirBuilder::new()
        .recursive(true)
        .mode(OVERFLOW_DIR_MODE)
        .create(scratch_dir)
        .map_err(create_error)?;
    std::fs::set_permissions(scratch_dir, Permissions::from_mode(OVERFLOW_DIR_MODE))
        .map_err(create_error)?;
    let scratch_dir = std::fs::canonicalize(scratch_dir).map_err(create_error)?;
    prune_entries(&scratch_dir, retention_cutoff(scratch.retention_days));
    Ok(scratch_dir)
}

/// The modification time before which an entry has outlived
/// `retention_days`.
fn retention_cutoff(retention_days: u32) -> SystemTime {
    let retention = Duration::from_secs(u64::from(retention_days) * SECS_PER_DAY);
    let now = SystemTime::now();
    now.checked_sub(retention).unwrap_or(SystemTime::UNIX_EPOCH)
}

/// Remove the entries of `dir` that [`prune`] lets go, returning whether
/// none are left.
fn prune_entries(dir: &Path, cutoff: SystemTime) -> bool {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) => {
            log_unpruned(dir, &error);
            return false;
        }
    };
    let kept = entries
        .filter(|entry| {
            !entry
                .as_ref()
                .is_ok_and(|entry| prune(&entry.path(), cutoff))
        })
        .count();
    kept == 0
}

/// Remove `path` when it last changed before `cutoff`, never following a
/// symlink; a directory goes only once its own entries have. Returns
/// whether it was removed.
fn prune(path: &Path, cutoff: SystemTime) -> bool {
    let metadata = match path.symlink_metadata() {
        Ok(metadata) => metadata,
        Err(error) => {
            log_unpruned(path, &error);
            return false;
        }
    };
    let expired = metadata.modified().is_ok_and(|changed| changed < cutoff);
    let removal = if metadata.is_dir() {
        let emptied = prune_entries(path, cutoff);
        (emptied && expired).then(|| std::fs::remove_dir(path))
    } else {
        expired.then(|| std::fs::remove_file(path))
    };
    match removal {
        Some(Ok(())) => true,
        Some(Err(error)) => {
            log_unpruned(path, &error);
            false
        }
        None => false,
    }
}

fn log_unpruned(path: &Path, error: &std::io::Error) {
    warn!(
        target: "assistd::policy",
        path = %path.display(),
        %error,
        "could not prune an expired tools.scratch.dir entry"
    );
}

#[cfg(test)]
mod tests;
