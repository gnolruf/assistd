use super::*;
use tempfile::tempdir;

fn cfg_from<P: AsRef<Path>>(paths: &[P]) -> Arc<WritePolicyCfg> {
    let abs: Vec<PathBuf> = paths
        .iter()
        .map(|p| std::fs::canonicalize(p.as_ref()).expect("canonicalize tempdir"))
        .collect();
    Arc::new(WritePolicyCfg::new(abs).expect("non-empty allowlist"))
}

#[test]
fn empty_allowlist_yields_no_policy() {
    assert!(WritePolicyCfg::new(Vec::new()).is_none());
}

#[tokio::test]
async fn persists_stdin_to_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("out.txt");
    let out = WriteCommand::new(cfg_from(&[dir.path()]))
        .run(CommandInput {
            args: vec![path.to_string_lossy().into_owned()],
            stdin: Some(b"hi there\n".to_vec()),
        })
        .await
        .unwrap();
    assert_eq!(out.exit_code, 0);
    let content = tokio::fs::read(&path).await.unwrap();
    assert_eq!(content, b"hi there\n");
}

#[tokio::test]
async fn persists_args_content_to_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("out.txt");
    let out = WriteCommand::new(cfg_from(&[dir.path()]))
        .run(CommandInput {
            args: vec![
                path.to_string_lossy().into_owned(),
                "hello".into(),
                "world".into(),
            ],
            stdin: None,
        })
        .await
        .unwrap();
    assert_eq!(out.exit_code, 0);
    let content = tokio::fs::read(&path).await.unwrap();
    assert_eq!(content, b"hello world");
}

#[tokio::test]
async fn args_content_wins_over_stdin() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("out.txt");
    let out = WriteCommand::new(cfg_from(&[dir.path()]))
        .run(CommandInput {
            args: vec![path.to_string_lossy().into_owned(), "args".into()],
            stdin: Some(b"stdin".to_vec()),
        })
        .await
        .unwrap();
    assert_eq!(out.exit_code, 0);
    let content = tokio::fs::read(&path).await.unwrap();
    assert_eq!(content, b"args");
}

#[tokio::test]
async fn no_args_errors() {
    let out = WriteCommand::permissive_for_tests()
        .run(CommandInput {
            args: Vec::new(),
            stdin: None,
        })
        .await
        .unwrap();
    assert_eq!(out.exit_code, 2);
}

#[tokio::test]
async fn path_only_with_empty_stdin_creates_empty_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("out.txt");
    let out = WriteCommand::new(cfg_from(&[dir.path()]))
        .run(CommandInput {
            args: vec![path.to_string_lossy().into_owned()],
            stdin: None,
        })
        .await
        .unwrap();
    assert_eq!(out.exit_code, 0);
    let content = tokio::fs::read(&path).await.unwrap();
    assert!(content.is_empty());
}

#[tokio::test]
async fn write_rejected_outside_allowlist() {
    let dir = tempdir().unwrap();
    let cmd = WriteCommand::new(cfg_from(&[dir.path()]));
    let out = cmd
        .run(CommandInput {
            args: vec!["/etc/passwd".into(), "oops".into()],
            stdin: None,
        })
        .await
        .unwrap();
    assert_eq!(out.exit_code, 126);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("[error] write: /etc/passwd: path not in writable allowlist"),
        "{stderr}"
    );
    assert!(
        stderr.contains("Check: [tools.write] writable_paths in config"),
        "{stderr}"
    );
}

#[tokio::test]
async fn write_allowlist_permits_tmp() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("permitted.txt");
    let out = WriteCommand::new(cfg_from(&[dir.path()]))
        .run(CommandInput {
            args: vec![path.to_string_lossy().into_owned(), "ok".into()],
            stdin: None,
        })
        .await
        .unwrap();
    assert_eq!(out.exit_code, 0);
}

#[tokio::test]
async fn write_allowlist_resolves_dotdot() {
    let dir = tempdir().unwrap();
    let tricky = format!("{}/../../../etc/passwd", dir.path().display());
    let out = WriteCommand::new(cfg_from(&[dir.path()]))
        .run(CommandInput {
            args: vec![tricky, "oops".into()],
            stdin: None,
        })
        .await
        .unwrap();
    assert_eq!(out.exit_code, 126);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("not in writable allowlist"), "{stderr}");
}

#[tokio::test]
async fn write_allowlist_rejects_relative_path() {
    let dir = tempdir().unwrap();
    let out = WriteCommand::new(cfg_from(&[dir.path()]))
        .run(CommandInput {
            args: vec!["relative.txt".into(), "hi".into()],
            stdin: None,
        })
        .await
        .unwrap();
    assert_eq!(out.exit_code, 126);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("relative paths not permitted"), "{stderr}");
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

#[tokio::test]
async fn write_allowlist_handles_nonexistent_parent() {
    let dir = tempdir().unwrap();
    // tokio::fs::write fails on a missing parent too, so we only assert
    // the allowlist check passes; the underlying write may still fail,
    // but with exit 1 (I/O error), not 126 (policy).
    let target = dir.path().join("newsub").join("file.txt");
    let cmd = WriteCommand::new(cfg_from(&[dir.path()]));
    let out = cmd
        .run(CommandInput {
            args: vec![target.to_string_lossy().into_owned(), "hi".into()],
            stdin: None,
        })
        .await
        .unwrap();
    // Policy must not reject (126); the I/O error path is exit 1.
    assert_ne!(out.exit_code, 126, "{:?}", out.stderr);
}

#[tokio::test]
async fn unwritable_path_exits_1() {
    let dir = tempdir().unwrap();
    // Keep the allowlist wide: the path is under the tempdir but its
    // direct parent doesn't exist.
    let out = WriteCommand::new(cfg_from(&[dir.path()]))
        .run(CommandInput {
            args: vec![
                format!("{}/definitely/not/a/writable/path", dir.path().display()),
                "hi".into(),
            ],
            stdin: None,
        })
        .await
        .unwrap();
    // Parent-missing is a policy success but an I/O failure; exit 1.
    assert_eq!(out.exit_code, 1);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("[error] write: "), "{stderr}");
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
    // `/..` stays at root.
    assert_eq!(lexical_clean(Path::new("/..")), PathBuf::from("/"));
}

#[tokio::test]
async fn dangling_symlink_escaping_allowlist_is_rejected() {
    let allowed = tempdir().unwrap();
    let outside = tempdir().unwrap();
    let escape_target = outside.path().join("escaped.txt");
    let link = allowed.path().join("link");
    std::os::unix::fs::symlink(&escape_target, &link).unwrap();

    let out = WriteCommand::new(cfg_from(&[allowed.path()]))
        .run(CommandInput {
            args: vec![link.to_string_lossy().into_owned(), "oops".into()],
            stdin: None,
        })
        .await
        .unwrap();
    assert_eq!(out.exit_code, 126, "{:?}", out.stderr);
    assert!(!escape_target.exists());
}

#[tokio::test]
async fn dangling_symlink_directory_escaping_allowlist_is_rejected() {
    let allowed = tempdir().unwrap();
    let outside = tempdir().unwrap();
    let link = allowed.path().join("dir");
    std::os::unix::fs::symlink(outside.path().join("missing"), &link).unwrap();

    let out = WriteCommand::new(cfg_from(&[allowed.path()]))
        .run(CommandInput {
            args: vec![
                link.join("file.txt").to_string_lossy().into_owned(),
                "oops".into(),
            ],
            stdin: None,
        })
        .await
        .unwrap();
    assert_eq!(out.exit_code, 126, "{:?}", out.stderr);
}

#[tokio::test]
async fn symlink_to_allowlisted_file_writes_through() {
    let dir = tempdir().unwrap();
    let real = dir.path().join("real.txt");
    std::fs::write(&real, b"old").unwrap();
    let link = dir.path().join("link");
    std::os::unix::fs::symlink(&real, &link).unwrap();

    let out = WriteCommand::new(cfg_from(&[dir.path()]))
        .run(CommandInput {
            args: vec![link.to_string_lossy().into_owned(), "new".into()],
            stdin: None,
        })
        .await
        .unwrap();
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

#[tokio::test]
async fn overwrites_existing_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("out.txt");
    std::fs::write(&path, b"old").unwrap();

    let out = WriteCommand::new(cfg_from(&[dir.path()]))
        .run(CommandInput {
            args: vec![path.to_string_lossy().into_owned(), "new".into()],
            stdin: None,
        })
        .await
        .unwrap();
    assert_eq!(out.exit_code, 0, "{:?}", out.stderr);
    assert_eq!(std::fs::read(&path).unwrap(), b"new");
}
