//! Opening named files for reading without opening devices or blocking on FIFOs.

use std::fs::{File, Metadata};
use std::io::{self, ErrorKind};
use std::path::Path;

use rustix::fs::{Mode, OFlags};

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

fn open_checked(path: &Path, accept: fn(&Metadata) -> bool) -> io::Result<(File, u64)> {
    refuse_unless(accept, &std::fs::metadata(path)?)?;
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
