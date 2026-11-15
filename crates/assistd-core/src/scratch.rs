//! The scratch directory `bash` and the in-process commands share: created
//! owner-only at startup, with entries past their retention removed.

use std::fs::{DirBuilder, File, Permissions};
use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use tracing::warn;

use assistd_config::ToolsScratchConfig;
use assistd_tools::{SCRATCH_MARKER, ScratchCap};

use crate::build_tools::{BuildToolsError, OVERFLOW_DIR_MODE};

const SECS_PER_DAY: u64 = 24 * 60 * 60;

/// The most a scratch directory may hold, however large its filesystem.
const MAX_SCRATCH_BYTES: u64 = 512 * 1024 * 1024;

/// The part of its filesystem's size a scratch directory may hold: one in this many.
const FILESYSTEM_SHARE: u64 = 4;

/// The scratch directory's canonical path, and its cap when assistd created it.
#[derive(Debug)]
pub(crate) struct PreparedScratch {
    pub(crate) dir: PathBuf,
    pub(crate) cap: Option<ScratchCap>,
}

/// Create `scratch.dir` owner-only and, when assistd created it, remove what outlived
/// `scratch.retention_days` and trim the rest to its cap. Errors with [`BuildToolsError::CreateScratchDir`]
/// when the directory cannot be created, restricted or resolved; failed removals are only logged.
pub(crate) fn prepare(scratch: &ToolsScratchConfig) -> Result<PreparedScratch, BuildToolsError> {
    let scratch_dir = scratch.resolved_dir()?;
    let scratch_dir = scratch_dir.as_path();
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
    let dir = std::fs::canonicalize(scratch_dir).map_err(create_error)?;
    let cap = claim(&dir).then(|| {
        prune_owned(&dir, retention_cutoff(scratch.retention_days));
        let cap = ScratchCap::new(dir.clone(), cap_bytes(&dir));
        cap.trim();
        cap
    });
    if cap.is_none() {
        warn!(
            target: "assistd::policy",
            path = %dir.display(),
            "tools.scratch.dir held files before assistd used it, so it is never pruned or capped"
        );
    }
    Ok(PreparedScratch { dir, cap })
}

/// What the scratch directory `dir` may hold: [`FILESYSTEM_SHARE`] of its filesystem's size, at
/// most [`MAX_SCRATCH_BYTES`], which also stands in for a size that cannot be read.
fn cap_bytes(dir: &Path) -> u64 {
    rustix::fs::statvfs(dir).map_or(MAX_SCRATCH_BYTES, |filesystem| {
        let share = filesystem.f_blocks.saturating_mul(filesystem.f_frsize) / FILESYSTEM_SHARE;
        share.min(MAX_SCRATCH_BYTES)
    })
}

/// Whether assistd owns `dir`: it holds [`SCRATCH_MARKER`], or was empty and now does.
fn claim(dir: &Path) -> bool {
    let marker = dir.join(SCRATCH_MARKER);
    if marker.symlink_metadata().is_ok_and(|meta| meta.is_file()) {
        return true;
    }
    let empty = std::fs::read_dir(dir).is_ok_and(|mut entries| entries.next().is_none());
    empty && File::create_new(&marker).is_ok()
}

/// Remove the entries of the owned `dir` that [`prune`] lets go, keeping its marker.
fn prune_owned(dir: &Path, cutoff: SystemTime) {
    match std::fs::read_dir(dir) {
        Ok(entries) => entries
            .filter_map(Result::ok)
            .filter(|entry| entry.file_name() != SCRATCH_MARKER)
            .for_each(|entry| {
                prune(&entry.path(), cutoff);
            }),
        Err(error) => log_unpruned(dir, &error),
    }
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
