//! Opening named files for reading without opening devices or blocking on FIFOs, and
//! replacing owner-only files atomically.

use std::fs::{self, File, Metadata, OpenOptions};
use std::io::{self, ErrorKind, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

use rustix::fs::{Mode, OFlags};

/// Mode of a file only its owner may read or write.
const PRIVATE_FILE_MODE: u32 = 0o600;

/// Open the regular file at `path` read-only with its size. Anything else,
/// directories included, fails with `InvalidInput`; see [`open_regular_or_dir`].
pub fn open_regular(path: &Path) -> io::Result<(File, u64)> {
    open_checked(path, Metadata::is_file)
}

/// Open `path` read-only with its size. A device, FIFO or socket is refused
/// with `InvalidInput` by a stat of the path, without opening it, and again by
/// an `fstat` after a non-blocking open; a directory opens and fails on read.
pub fn open_regular_or_dir(path: &Path) -> io::Result<(File, u64)> {
    open_checked(path, |meta| meta.is_file() || meta.is_dir())
}

/// Write `contents` to a mode-`0600` sibling of `path` and rename it over `path`, so a reader
/// sees the old file or the new one, never a partial one. The parent directory must exist.
pub fn replace_private_file(path: &Path, contents: &[u8]) -> io::Result<()> {
    let mut staging_name = path.file_name().unwrap_or_default().to_os_string();
    staging_name.push(".tmp");
    let staging = path.with_file_name(staging_name);
    match fs::remove_file(&staging) {
        Err(error) if error.kind() != ErrorKind::NotFound => return Err(error),
        _ => {}
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(PRIVATE_FILE_MODE)
        .open(&staging)?;
    file.write_all(contents)?;
    file.sync_all()?;
    fs::rename(&staging, path)
}

fn open_checked(path: &Path, accept: fn(&Metadata) -> bool) -> io::Result<(File, u64)> {
    refuse_unless(accept, &fs::metadata(path)?)?;
    let file = File::from(rustix::fs::open(
        path,
        OFlags::RDONLY | OFlags::NONBLOCK | OFlags::NOCTTY | OFlags::CLOEXEC,
        Mode::empty(),
    )?);
    let meta = file.metadata()?;
    refuse_unless(accept, &meta)?;
    rustix::fs::fcntl_setfl(
        &file,
        rustix::fs::fcntl_getfl(&file)?.difference(OFlags::NONBLOCK),
    )?;
    Ok((file, meta.len()))
}

fn refuse_unless(accept: fn(&Metadata) -> bool, meta: &Metadata) -> io::Result<()> {
    if accept(meta) {
        Ok(())
    } else {
        Err(io::Error::new(
            ErrorKind::InvalidInput,
            "not a regular file (device, pipe, or socket)",
        ))
    }
}

#[cfg(test)]
mod tests;
