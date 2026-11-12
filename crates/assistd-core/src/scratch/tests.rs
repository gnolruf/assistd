use std::fs::File;

use super::*;

const LONG_AGO: Duration = Duration::from_secs(90 * SECS_PER_DAY);

fn scratch(dir: &Path, retention_days: u32) -> ToolsScratchConfig {
    ToolsScratchConfig {
        dir: Some(dir.to_str().unwrap().parse().unwrap()),
        retention_days,
    }
}

/// A scratch directory assistd created earlier, with its marker.
fn owned_dir() -> tempfile::TempDir {
    let temp = tempfile::tempdir().unwrap();
    File::create(temp.path().join(OWNED_MARKER)).unwrap();
    temp
}

fn age(path: &Path) {
    let changed = SystemTime::now() - LONG_AGO;
    File::open(path)
        .and_then(|file| file.set_modified(changed))
        .expect("set mtime");
}

#[test]
fn prepare_creates_a_missing_dir_owner_only() {
    let temp = tempfile::tempdir().unwrap();
    let dir = temp.path().join("assistd/scratch");

    let prepared = prepare(&scratch(&dir, 30)).unwrap();

    assert_eq!(prepared, std::fs::canonicalize(&dir).unwrap());
    let mode = std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, OVERFLOW_DIR_MODE);
}

#[test]
fn prepare_removes_expired_entries_and_the_dirs_they_empty() {
    let temp = owned_dir();
    let dir = temp.path();
    for path in ["old-dir", "mixed-dir"] {
        std::fs::create_dir(dir.join(path)).unwrap();
    }
    for path in ["old.txt", "new.txt", "old-dir/a.txt", "mixed-dir/b.txt"] {
        std::fs::write(dir.join(path), b"x").unwrap();
    }
    std::fs::write(dir.join("mixed-dir/fresh.txt"), b"x").unwrap();
    for path in [
        "old.txt",
        "old-dir/a.txt",
        "old-dir",
        "mixed-dir/b.txt",
        "mixed-dir",
        OWNED_MARKER,
    ] {
        age(&dir.join(path));
    }

    prepare(&scratch(dir, 30)).unwrap();

    let exists = |path: &str| dir.join(path).symlink_metadata().is_ok();
    for gone in ["old.txt", "old-dir", "mixed-dir/b.txt"] {
        assert!(!exists(gone), "{gone} outlived its retention");
    }
    for kept in ["new.txt", "mixed-dir", "mixed-dir/fresh.txt", OWNED_MARKER] {
        assert!(exists(kept), "{kept} was removed early");
    }
}

#[test]
fn zero_retention_removes_symlinks_without_following_them() {
    let outside = tempfile::NamedTempFile::new().unwrap();
    let temp = owned_dir();
    let dir = temp.path();
    std::fs::create_dir(dir.join("sub")).unwrap();
    std::fs::write(dir.join("sub/a.txt"), b"x").unwrap();
    std::os::unix::fs::symlink(outside.path(), dir.join("link")).unwrap();

    prepare(&scratch(dir, 0)).unwrap();

    assert_eq!(std::fs::read_dir(dir).unwrap().count(), 1);
    assert!(outside.path().exists(), "the symlink's target was removed");
}

#[test]
fn a_dir_that_held_files_before_assistd_is_never_pruned() {
    let temp = tempfile::tempdir().unwrap();
    let dir = temp.path();
    std::fs::write(dir.join("thesis.tex"), b"x").unwrap();
    age(&dir.join("thesis.tex"));

    prepare(&scratch(dir, 0)).unwrap();

    assert!(dir.join("thesis.tex").exists());
    assert!(!dir.join(OWNED_MARKER).exists());
}
