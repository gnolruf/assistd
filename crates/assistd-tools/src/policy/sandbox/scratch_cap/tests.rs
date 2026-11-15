use std::fs::File;
use std::time::SystemTime;

use super::*;

const FILE_BYTES: usize = 64 * 1024;

fn capped(dir: &Path, max_bytes: u64) -> ScratchCap {
    ScratchCap::new(std::fs::canonicalize(dir).unwrap(), max_bytes)
}

fn write_changed(path: &Path, secs_ago: u64) {
    std::fs::write(path, vec![1u8; FILE_BYTES]).unwrap();
    let changed = SystemTime::now() - Duration::from_secs(secs_ago);
    File::options()
        .write(true)
        .open(path)
        .and_then(|file| file.set_modified(changed))
        .expect("set mtime");
}

#[test]
fn trim_removes_the_oldest_files_until_the_rest_fit() {
    let temp = tempfile::tempdir().unwrap();
    let dir = temp.path();
    std::fs::create_dir(dir.join("sub")).unwrap();
    write_changed(&dir.join("oldest.bin"), 300);
    write_changed(&dir.join("sub/older.bin"), 200);
    write_changed(&dir.join("newest.bin"), 100);
    write_changed(&dir.join(SCRATCH_MARKER), 400);

    let removed = capped(dir, (FILE_BYTES + FILE_BYTES / 2) as u64).trim();

    assert_eq!(removed, 2);
    let left = |path: &str| dir.join(path).exists();
    assert!(!left("oldest.bin") && !left("sub/older.bin"));
    assert!(left("newest.bin") && left(SCRATCH_MARKER));
}

#[test]
fn trim_leaves_what_a_symlink_points_at() {
    let outside = tempfile::tempdir().unwrap();
    write_changed(&outside.path().join("thesis.tex"), 300);
    let temp = tempfile::tempdir().unwrap();
    let dir = temp.path();
    std::os::unix::fs::symlink(outside.path(), dir.join("link")).unwrap();
    write_changed(&dir.join("inside.bin"), 100);

    capped(dir, 0).trim();

    assert!(!dir.join("inside.bin").exists());
    assert!(outside.path().join("thesis.tex").exists());
}

#[tokio::test]
async fn guard_drops_a_command_that_outgrows_the_cap() {
    let temp = tempfile::tempdir().unwrap();
    let grown = temp.path().join("grown.bin");

    let guarded = capped(temp.path(), (FILE_BYTES / 2) as u64)
        .guard(async {
            write_changed(&grown, 0);
            std::future::pending::<()>().await;
        })
        .await;

    assert!(guarded.output.is_none());
    assert_eq!(guarded.removed, 1);
    assert!(!grown.exists());
}
