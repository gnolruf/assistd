//! The cap on what the shared scratch directory holds, kept by ending the
//! command that outgrows it and removing its oldest files.

use std::ffi::{OsStr, OsString};
use std::os::fd::OwnedFd;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use rustix::fs::{AtFlags, Dir, FileType, Mode, OFlags, ResolveFlags, Stat};
use rustix::io;
use tracing::warn;

use crate::policy::on_blocking_pool;

/// The file that marks a scratch directory assistd created, which
/// [`ScratchCap::trim`] never removes.
pub const SCRATCH_MARKER: &str = ".assistd-scratch";

/// How long a running command goes between measurements of its scratch
/// directory.
const POLL_INTERVAL: Duration = Duration::from_millis(25);

/// The size of the blocks `stat` counts a file's storage in.
const STAT_BLOCK_BYTES: u64 = 512;

/// The most the files of a scratch directory assistd created may hold
/// together, in bytes of storage. Measuring and trimming cross no symlink.
#[derive(Debug, Clone)]
pub struct ScratchCap {
    dir: PathBuf,
    max_bytes: u64,
}

/// What became of a future run under [`ScratchCap::guard`].
#[derive(Debug)]
pub(crate) struct Guarded<T> {
    /// `None` when the directory outgrew its cap first, which dropped the
    /// future.
    pub(crate) output: Option<T>,
    /// How many files were removed to bring the directory within its cap.
    pub(crate) removed: usize,
}

/// A file in a capped directory, named within `dir`, its directory's path
/// below the capped one.
struct ScratchFile {
    dir: PathBuf,
    name: OsString,
    /// Seconds and nanoseconds since the epoch.
    changed: (i64, u64),
    bytes: u64,
}

impl ScratchCap {
    /// A cap of `max_bytes` on the canonical `dir`.
    pub fn new(dir: PathBuf, max_bytes: u64) -> Self {
        Self { dir, max_bytes }
    }

    pub(crate) fn dir(&self) -> &Path {
        &self.dir
    }

    pub(crate) fn max_bytes(&self) -> u64 {
        self.max_bytes
    }

    /// Remove the files that changed longest ago until the rest fit the
    /// cap, returning how many went. Blocks on the filesystem.
    pub fn trim(&self) -> usize {
        let Ok((root, mut files)) = self.files() else {
            return 0;
        };
        let held: u64 = files.iter().map(|file| file.bytes).sum();
        files.sort_unstable_by_key(|file| file.changed);
        let removed = files
            .iter()
            .filter(|file| file.bytes > 0)
            .scan(held.saturating_sub(self.max_bytes), |excess, file| {
                let over = *excess > 0;
                *excess = excess.saturating_sub(file.bytes);
                over.then_some(file)
            })
            .filter(|file| self.remove(&root, file))
            .count();
        if removed > 0 {
            warn!(
                target: "assistd::policy",
                dir = %self.dir.display(),
                removed,
                "tools.scratch.dir outgrew its cap, so its oldest files were removed"
            );
        }
        removed
    }

    /// Run `command`, dropping it once the directory outgrows the cap, and
    /// trim the directory before and after.
    pub(crate) async fn guard<F: Future>(&self, command: F) -> Guarded<F::Output> {
        let removed_before = self.trim_on_blocking_pool().await;
        let output = tokio::select! {
            output = command => Some(output),
            () = self.outgrown() => None,
        };
        let removed_after = self.trim_on_blocking_pool().await;
        Guarded {
            output,
            removed: removed_before + removed_after,
        }
    }

    async fn trim_on_blocking_pool(&self) -> usize {
        let cap = self.clone();
        on_blocking_pool(move || cap.trim()).await
    }

    /// Resolves once the directory holds more than the cap.
    async fn outgrown(&self) {
        loop {
            tokio::time::sleep(POLL_INTERVAL).await;
            let cap = self.clone();
            if on_blocking_pool(move || cap.is_outgrown()).await {
                return;
            }
        }
    }

    fn is_outgrown(&self) -> bool {
        self.files().is_ok_and(|(_, files)| {
            files.iter().map(|file| file.bytes).sum::<u64>() > self.max_bytes
        })
    }

    /// The directory, opened, and every file under it but the marker, a
    /// symlink counting as itself.
    fn files(&self) -> io::Result<(OwnedFd, Vec<ScratchFile>)> {
        let root = rustix::fs::open(
            &self.dir,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC,
            Mode::empty(),
        )?;
        let mut files = Vec::new();
        let mut unread_dirs = vec![PathBuf::new()];
        while let Some(dir) = unread_dirs.pop() {
            for (name, stat) in entries_below(&root, &dir) {
                if FileType::from_raw_mode(stat.st_mode) == FileType::Directory {
                    unread_dirs.push(dir.join(name));
                } else if !(dir.as_os_str().is_empty() && name == SCRATCH_MARKER) {
                    files.push(ScratchFile {
                        dir: dir.clone(),
                        name,
                        changed: (stat.st_mtime, stat.st_mtime_nsec),
                        bytes: u64::try_from(stat.st_blocks).unwrap_or(0) * STAT_BLOCK_BYTES,
                    });
                }
            }
        }
        Ok((root, files))
    }

    /// Unlink `file` under the opened `root`, returning whether it went.
    fn remove(&self, root: &OwnedFd, file: &ScratchFile) -> bool {
        let unlinked = open_below(root, &file.dir, OFlags::PATH | OFlags::DIRECTORY)
            .and_then(|dir| rustix::fs::unlinkat(dir, &file.name, AtFlags::empty()));
        if let Err(error) = &unlinked {
            warn!(
                target: "assistd::policy",
                path = %self.dir.join(&file.dir).join(&file.name).display(),
                %error,
                "could not remove a file from an over-full tools.scratch.dir"
            );
        }
        unlinked.is_ok()
    }
}

/// The name and `stat` of each entry of `dir` below the opened `root`;
/// none when `dir` cannot be read, and an entry that cannot be examined is
/// left out.
fn entries_below(root: &OwnedFd, dir: &Path) -> Vec<(OsString, Stat)> {
    let Ok(dir) = open_below(root, dir, OFlags::RDONLY | OFlags::DIRECTORY) else {
        return Vec::new();
    };
    let Ok(entries) = Dir::read_from(&dir) else {
        return Vec::new();
    };
    entries
        .filter_map(Result::ok)
        .filter(|entry| !matches!(entry.file_name().to_bytes(), b"." | b".."))
        .filter_map(|entry| {
            let stat = rustix::fs::statat(&dir, entry.file_name(), AtFlags::SYMLINK_NOFOLLOW);
            let name = OsStr::from_bytes(entry.file_name().to_bytes()).to_owned();
            Some((name, stat.ok()?))
        })
        .collect()
}

/// `below` opened under the directory `root`, the empty path being `root`
/// itself; fails rather than cross a symlink or leave `root`.
fn open_below(root: &OwnedFd, below: &Path, flags: OFlags) -> io::Result<OwnedFd> {
    let below = if below.as_os_str().is_empty() {
        Path::new(".")
    } else {
        below
    };
    rustix::fs::openat2(
        root,
        below,
        flags | OFlags::CLOEXEC,
        Mode::empty(),
        ResolveFlags::BENEATH | ResolveFlags::NO_SYMLINKS,
    )
}

#[cfg(test)]
mod tests;
