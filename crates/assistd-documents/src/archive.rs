//! Zip archives read under limits, so one that expands far past its own size is
//! skipped rather than filling the worker's memory.

use std::io::{Cursor, Read};

use zip::ZipArchive;
use zip::result::ZipError;

use crate::protocol::SkipReason;

/// Most entries an archive may hold; one with more is not read.
pub(crate) const MAX_ENTRIES: usize = 10_000;

/// Most bytes one entry may expand to.
pub(crate) const MAX_ENTRY_BYTES: u64 = 64 * 1024 * 1024;

/// Most bytes the entries read from one archive may expand to together.
pub(crate) const MAX_ARCHIVE_BYTES: u64 = 256 * 1024 * 1024;

/// A zip archive whose entries are read whole, each within [`MAX_ENTRY_BYTES`]
/// and all of them within [`MAX_ARCHIVE_BYTES`].
pub(crate) struct Archive {
    zip: ZipArchive<Cursor<Vec<u8>>>,
    /// Bytes the entries still to be read may expand to.
    room: u64,
}

impl Archive {
    /// The archive in `bytes`; [`SkipReason::TooLarge`] for one of more than
    /// [`MAX_ENTRIES`] entries.
    pub(crate) fn open(bytes: Vec<u8>) -> Result<Self, SkipReason> {
        let zip = ZipArchive::new(Cursor::new(bytes)).map_err(|error| failed(&error))?;
        if zip.len() > MAX_ENTRIES {
            return Err(SkipReason::TooLarge);
        }
        Ok(Self {
            zip,
            room: MAX_ARCHIVE_BYTES,
        })
    }

    /// The entry called `name`, expanded; [`SkipReason::TooLarge`] when it, or
    /// it with the entries read before it, expands past a limit, whatever size
    /// the archive declares for it.
    pub(crate) fn read(&mut self, name: &str) -> Result<Vec<u8>, SkipReason> {
        let entry = self.zip.by_name(name).map_err(|error| match error {
            ZipError::FileNotFound => failure(format!("the archive has no entry {name}")),
            error => failed(&error),
        })?;
        let limit = MAX_ENTRY_BYTES.min(self.room);
        if entry.size() > limit {
            return Err(SkipReason::TooLarge);
        }
        let mut bytes = Vec::with_capacity(usize::try_from(entry.size()).unwrap_or_default());
        let read = entry
            .take(limit.saturating_add(1))
            .read_to_end(&mut bytes)
            .map_err(|error| failure(format!("{name}: {error}")))? as u64;
        if read > limit {
            return Err(SkipReason::TooLarge);
        }
        self.room -= read;
        Ok(bytes)
    }
}

fn failed(error: &ZipError) -> SkipReason {
    failure(error.to_string())
}

fn failure(message: String) -> SkipReason {
    SkipReason::ExtractorFailed { message }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample_archive::archive_of;

    #[test]
    fn an_entry_or_an_archive_that_expands_past_its_limit_is_not_read() {
        let full = vec![b' '; usize::try_from(MAX_ENTRY_BYTES).expect("fits")];
        let over = [full.as_slice(), b" "].concat();
        let bytes = archive_of(&[("over", &over), ("full", &full), ("small", b"a")]);
        assert!(bytes.len() < full.len() / 100, "the fixture is a zip bomb");
        let mut archive = Archive::open(bytes).expect("opens");

        assert_eq!(archive.read("over"), Err(SkipReason::TooLarge));
        for _ in 0..MAX_ARCHIVE_BYTES / MAX_ENTRY_BYTES {
            assert_eq!(archive.read("full").map(|read| read.len()), Ok(full.len()));
        }
        assert_eq!(archive.read("small"), Err(SkipReason::TooLarge));
    }
}
