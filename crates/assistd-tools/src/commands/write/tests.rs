use super::*;
use tempfile::tempdir;

fn cfg_from<P: AsRef<Path>>(paths: &[P]) -> Arc<WritePolicyCfg> {
    let abs: Vec<PathBuf> = paths
        .iter()
        .map(|p| std::fs::canonicalize(p.as_ref()).expect("canonicalize tempdir"))
        .collect();
    Arc::new(WritePolicyCfg::new(abs).expect("non-empty allowlist"))
}

async fn write_under(allowed: &Path, args: &[&str], stdin: Option<&[u8]>) -> CommandOutput {
    WriteCommand::new(cfg_from(&[allowed]))
        .run(CommandInput {
            args: args.iter().map(|s| s.to_string()).collect(),
            stdin: stdin.map(<[u8]>::to_vec),
        })
        .await
}

#[test]
fn empty_allowlist_yields_no_policy() {
    assert!(WritePolicyCfg::new(Vec::new()).is_none());
}

#[tokio::test]
async fn persists_stdin_to_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("out.txt");
    let out = write_under(dir.path(), &[&path.to_string_lossy()], Some(b"hi there\n")).await;
    assert_eq!(out.exit_code, 0);
    assert_eq!(std::fs::read(&path).unwrap(), b"hi there\n");
}

#[tokio::test]
async fn persists_args_content_to_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("out.txt");
    let out = write_under(
        dir.path(),
        &[&path.to_string_lossy(), "hello", "world"],
        None,
    )
    .await;
    assert_eq!(out.exit_code, 0);
    assert_eq!(std::fs::read(&path).unwrap(), b"hello world");
}

#[tokio::test]
async fn args_content_wins_over_stdin() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("out.txt");
    let out = write_under(
        dir.path(),
        &[&path.to_string_lossy(), "args"],
        Some(b"stdin"),
    )
    .await;
    assert_eq!(out.exit_code, 0);
    assert_eq!(std::fs::read(&path).unwrap(), b"args");
}

#[tokio::test]
async fn no_args_emits_usage() {
    let out = WriteCommand::permissive_for_tests()
        .run(CommandInput {
            args: Vec::new(),
            stdin: None,
        })
        .await;
    assert_eq!(out.exit_code, 2);
    assert!(out.stdout.starts_with(b"usage: write"), "{out:?}");
}

#[tokio::test]
async fn path_only_with_empty_stdin_creates_empty_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("out.txt");
    let out = write_under(dir.path(), &[&path.to_string_lossy()], None).await;
    assert_eq!(out.exit_code, 0);
    assert_eq!(std::fs::read(&path).unwrap(), b"");
}

#[tokio::test]
async fn write_rejected_outside_allowlist() {
    let dir = tempdir().unwrap();
    let outside = tempdir().unwrap();
    let target = outside.path().join("target.txt");
    let target_str = target.to_string_lossy().into_owned();
    let out = write_under(dir.path(), &[&target_str, "oops"], None).await;
    assert_eq!(out.exit_code, 126);
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        format!(
            "[error] write: {target_str}: path not in writable allowlist. \
             Check: [tools.write] writable_paths in config\n"
        )
    );
    assert!(!target.exists());
}

#[tokio::test]
async fn write_allowlist_resolves_dotdot() {
    let root = tempdir().unwrap();
    let allowed = root.path().join("allowed");
    std::fs::create_dir(&allowed).unwrap();
    let escaped = root.path().join("escaped.txt");
    let tricky = format!("{}/../escaped.txt", allowed.display());
    let out = write_under(&allowed, &[&tricky, "oops"], None).await;
    assert_eq!(out.exit_code, 126);
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        format!(
            "[error] write: {tricky}: path not in writable allowlist. \
             Check: [tools.write] writable_paths in config\n"
        )
    );
    assert!(!escaped.exists());
}

#[tokio::test]
async fn write_allowlist_rejects_relative_path() {
    let dir = tempdir().unwrap();
    let out = write_under(dir.path(), &["relative.txt", "hi"], None).await;
    assert_eq!(out.exit_code, 126);
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "[error] write: relative.txt: relative paths not permitted. \
         Try: an absolute path under an allowlisted directory\n"
    );
}

#[test]
fn write_allowlist_expands_tilde() {
    let dir = tempdir().unwrap();
    let home = dir.path().to_str().expect("tempdir path is valid utf-8");
    let resolved = resolve_for_allowlist("~/tilde-target.txt", Some(home)).expect("tilde resolves");
    let expected = std::fs::canonicalize(dir.path())
        .expect("tempdir canonicalizes")
        .join("tilde-target.txt");
    assert_eq!(resolved, expected);
}

#[test]
fn expand_tilde_without_home_errors() {
    let err = expand_tilde("~/foo", None).expect_err("missing home should error");
    assert!(matches!(err, PathResolveError::HomeNotSet));
}

/// A missing parent passes the allowlist (the anchor is its nearest
/// existing ancestor) and fails only at the write, with exit 1.
#[tokio::test]
async fn missing_parent_passes_policy_but_fails_the_write() {
    let dir = tempdir().unwrap();
    let parent = format!("{}/definitely/not/a/writable", dir.path().display());
    let target = format!("{parent}/path");
    let out = write_under(dir.path(), &[&target, "hi"], None).await;
    assert_eq!(out.exit_code, 1);
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        format!("[error] write: file not found: {target}. Use: ls {parent} to see what is there\n")
    );
}

#[test]
fn lexical_clean_collapses_dotdot() {
    assert_eq!(
        lexical_clean(Path::new("/tmp/foo/../bar")),
        PathBuf::from("/tmp/bar")
    );
    assert_eq!(
        lexical_clean(Path::new("/tmp/./foo")),
        PathBuf::from("/tmp/foo")
    );
    assert_eq!(lexical_clean(Path::new("/..")), PathBuf::from("/"));
}

#[tokio::test]
async fn dangling_symlink_escaping_allowlist_is_rejected() {
    let allowed = tempdir().unwrap();
    let outside = tempdir().unwrap();
    let escape_target = outside.path().join("escaped.txt");
    let link = allowed.path().join("link");
    std::os::unix::fs::symlink(&escape_target, &link).unwrap();

    let out = write_under(allowed.path(), &[&link.to_string_lossy(), "oops"], None).await;
    assert_eq!(out.exit_code, 126, "{:?}", out.stderr);
    assert!(!escape_target.exists());
}

#[tokio::test]
async fn dangling_symlink_directory_escaping_allowlist_is_rejected() {
    let allowed = tempdir().unwrap();
    let outside = tempdir().unwrap();
    let missing = outside.path().join("missing");
    let link = allowed.path().join("dir");
    std::os::unix::fs::symlink(&missing, &link).unwrap();

    let target = link.join("file.txt");
    let out = write_under(allowed.path(), &[&target.to_string_lossy(), "oops"], None).await;
    assert_eq!(out.exit_code, 126, "{:?}", out.stderr);
    assert!(!missing.exists());
}

#[tokio::test]
async fn symlink_to_allowlisted_file_writes_through() {
    let dir = tempdir().unwrap();
    let real = dir.path().join("real.txt");
    std::fs::write(&real, b"old").unwrap();
    let link = dir.path().join("link");
    std::os::unix::fs::symlink(&real, &link).unwrap();

    let out = write_under(dir.path(), &[&link.to_string_lossy(), "new"], None).await;
    assert_eq!(out.exit_code, 0, "{:?}", out.stderr);
    assert_eq!(std::fs::read(&real).unwrap(), b"new");
}

#[tokio::test]
async fn write_no_follow_refuses_final_symlink() {
    let dir = tempdir().unwrap();
    let target = dir.path().join("target.txt");
    let link = dir.path().join("link");
    std::os::unix::fs::symlink(&target, &link).unwrap();

    write_no_follow(&link, b"oops")
        .await
        .expect_err("symlink at final component must not be followed");
    assert!(!target.exists());
}

/// The old content is longer than the new, so a missing truncate would
/// leave its tail behind.
#[tokio::test]
async fn overwrites_and_truncates_existing_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("out.txt");
    std::fs::write(&path, b"much longer old content").unwrap();

    let out = write_under(dir.path(), &[&path.to_string_lossy(), "new"], None).await;
    assert_eq!(out.exit_code, 0, "{:?}", out.stderr);
    assert_eq!(std::fs::read(&path).unwrap(), b"new");
}
