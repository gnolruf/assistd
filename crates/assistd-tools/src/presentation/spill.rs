//! Spill files: the whole text of an output that was cut to its budget, kept in an overflow
//! directory that holds only the newest of them.

use std::cmp::Reverse;
use std::fs::{DirEntry, Metadata, OpenOptions};
use std::io::{ErrorKind, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// Spill files hold raw tool output, so only the daemon's user may read them.
pub(super) const OVERFLOW_FILE_MODE: u32 = 0o600;

/// Most spill files kept in an overflow directory; older ones are deleted.
const MAX_SPILL_FILES: usize = 64;

/// Most spill bytes kept in an overflow directory; older files are deleted.
const MAX_SPILL_BYTES: u64 = 128 * 1024 * 1024;

/// An existing `<stem>-<n>.txt` spill file in an overflow directory.
#[derive(Debug)]
struct SpillFile {
    path: PathBuf,
    modified: SystemTime,
    sequence: u64,
    len: u64,
}

impl SpillFile {
    /// `None` unless `entry` is a regular file named like a spill.
    fn from_dir_entry(entry: &DirEntry) -> Option<Self> {
        let sequence = spill_sequence(entry.file_name().to_str()?)?;
        let metadata = entry.metadata().ok().filter(Metadata::is_file)?;
        Some(Self {
            path: entry.path(),
            modified: metadata.modified().ok()?,
            sequence,
            len: metadata.len(),
        })
    }
}

pub(super) fn write_overflow_file(raw: &[u8], path: &Path) -> std::io::Result<PathBuf> {
    OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(OVERFLOW_FILE_MODE)
        .open(path)?
        .write_all(raw)?;
    Ok(path.to_path_buf())
}

/// Delete the oldest spill files in `dir` until it holds at most [`MAX_SPILL_FILES`] and
/// [`MAX_SPILL_BYTES`], never touching `newest` (`newest_len` bytes); failures are only logged.
pub(super) fn prune_spills(dir: &Path, newest: &Path, newest_len: u64) {
    let older = match older_spills(dir, newest) {
        Ok(older) => older,
        Err(e) => {
            tracing::warn!("failed to list spill files in {}: {e}", dir.display());
            return;
        }
    };
    for spill in evicted_spills(older, newest_len) {
        if let Err(e) = std::fs::remove_file(&spill.path)
            && e.kind() != ErrorKind::NotFound
        {
            tracing::warn!("failed to delete spill file {}: {e}", spill.path.display());
        }
    }
}

/// The spill files in `dir` other than `newest`, newest first.
fn older_spills(dir: &Path, newest: &Path) -> std::io::Result<Vec<SpillFile>> {
    let mut spills: Vec<SpillFile> = std::fs::read_dir(dir)?
        .filter_map(Result::ok)
        .filter(|entry| entry.path() != newest)
        .filter_map(|entry| SpillFile::from_dir_entry(&entry))
        .collect();
    spills.sort_unstable_by_key(|spill| Reverse((spill.modified, spill.sequence)));
    Ok(spills)
}

/// The tail of newest-first `older` that does not fit in [`MAX_SPILL_FILES`] and
/// [`MAX_SPILL_BYTES`] alongside the newest spill of `newest_len` bytes.
fn evicted_spills(mut older: Vec<SpillFile>, newest_len: u64) -> Vec<SpillFile> {
    let mut kept_bytes = newest_len;
    let kept = older
        .iter()
        .take(MAX_SPILL_FILES - 1)
        .take_while(|spill| {
            kept_bytes = kept_bytes.saturating_add(spill.len);
            kept_bytes <= MAX_SPILL_BYTES
        })
        .count();
    older.split_off(kept)
}

/// Whether `file_name` names a spill file: `<stem>-<n>.txt` with a
/// non-empty stem and `n` ASCII digits that fit in a `u64`.
pub fn is_spill_file_name(file_name: &str) -> bool {
    spill_sequence(file_name).is_some()
}

/// The counter `n` of a spill file named `<stem>-<n>.txt`.
fn spill_sequence(file_name: &str) -> Option<u64> {
    let (stem, sequence) = file_name.strip_suffix(".txt")?.rsplit_once('-')?;
    let digits_only = sequence.bytes().all(|b| b.is_ascii_digit());
    (!stem.is_empty() && digits_only)
        .then_some(sequence)?
        .parse()
        .ok()
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use tempfile::tempdir;

    use super::super::{OutputCap, PresentSpec};
    use super::*;

    fn sorted_file_names(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        names
    }

    /// A sparse file of `len` bytes, last modified `age_secs` ago.
    fn write_file_aged(dir: &Path, name: &str, len: u64, age_secs: u64) -> PathBuf {
        let path = dir.join(name);
        let file = std::fs::File::create(&path).unwrap();
        file.set_len(len).unwrap();
        file.set_modified(SystemTime::now() - Duration::from_secs(age_secs))
            .unwrap();
        path
    }

    #[test]
    fn spilling_past_the_file_cap_keeps_only_the_newest_spills() {
        let dir = tempdir().unwrap();
        let spec = PresentSpec {
            max_lines: 1,
            max_bytes: 50 * 1024,
            overflow_dir: dir.path().to_path_buf(),
        };
        let cap = OutputCap::new(spec);
        let total = MAX_SPILL_FILES + 5;

        for _ in 0..total {
            let cut = cap.cap("cmd", "a\nb\n".into());
            assert!(cut.overflow_file.as_deref().is_some_and(Path::exists));
        }

        let mut expected: Vec<String> = (6..=total).map(|n| format!("cmd-{n}.txt")).collect();
        expected.sort();
        assert_eq!(sorted_file_names(dir.path()), expected);
    }

    #[test]
    fn is_spill_file_name_accepts_only_stem_dash_counter_dot_txt() {
        let cases = [
            ("cmd-1.txt", true),
            ("mcp-files-12.txt", true),
            ("cmd-007.txt", true),
            ("cmd-18446744073709551615.txt", true),
            ("cmd-18446744073709551616.txt", false),
            ("notes.txt", false),
            ("cmd1.txt", false),
            ("cmd-.txt", false),
            ("-3.txt", false),
            ("cmd-+1.txt", false),
            ("cmd-x.txt", false),
            ("cmd-1.md", false),
            ("cmd-1.txt.bak", false),
        ];
        for (name, expected) in cases {
            assert_eq!(is_spill_file_name(name), expected, "{name}");
        }
    }

    /// Age decides across stems whose counters are unrelated, and files not
    /// named like spills are never touched.
    #[test]
    fn prune_spills_evicts_the_oldest_until_under_the_byte_cap() {
        let dir = tempdir().unwrap();
        let third_of_cap = MAX_SPILL_BYTES / 3;
        write_file_aged(dir.path(), "notes.txt", 5 * third_of_cap, 50);
        write_file_aged(dir.path(), "mcp-web-9.txt", third_of_cap, 40);
        write_file_aged(dir.path(), "cmd-1.txt", third_of_cap, 30);
        write_file_aged(dir.path(), "mcp-web-10.txt", third_of_cap, 20);
        let newest = write_file_aged(dir.path(), "cmd-2.txt", third_of_cap, 0);

        prune_spills(dir.path(), &newest, third_of_cap);

        assert_eq!(
            sorted_file_names(dir.path()),
            ["cmd-1.txt", "cmd-2.txt", "mcp-web-10.txt", "notes.txt"]
        );
    }

    #[test]
    fn prune_spills_keeps_the_new_spill_even_when_it_alone_exceeds_the_caps() {
        let dir = tempdir().unwrap();
        let over_cap = MAX_SPILL_BYTES + 1;
        write_file_aged(dir.path(), "cmd-1.txt", 10, 0);
        let newest = write_file_aged(dir.path(), "cmd-2.txt", over_cap, 60);

        prune_spills(dir.path(), &newest, over_cap);

        assert_eq!(sorted_file_names(dir.path()), ["cmd-2.txt"]);
    }
}
