use tempfile::tempdir;

use super::*;

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
async fn symlink_final_component_is_refused() {
    let dir = tempdir().unwrap();
    let real = dir.path().join("real.txt");
    std::fs::write(&real, b"old").unwrap();
    let link = dir.path().join("link");
    std::os::unix::fs::symlink(&real, &link).unwrap();

    let link_str = link.to_string_lossy().into_owned();
    let out = write_under(dir.path(), &[&link_str, "new"], None).await;
    assert_eq!(out.exit_code, POLICY_DENIED_EXIT);
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        format!("[error] write: {link_str}: is a symlink. Try: writing to the file it points at\n")
    );
    assert_eq!(std::fs::read(&real).unwrap(), b"old");
}

#[tokio::test]
async fn symlinked_ancestor_inside_the_allowlist_writes_to_its_target() {
    let dir = tempdir().unwrap();
    let real_dir = dir.path().join("real");
    std::fs::create_dir(&real_dir).unwrap();
    let dir_link = dir.path().join("dir-link");
    std::os::unix::fs::symlink(&real_dir, &dir_link).unwrap();

    let target = dir_link.join("out.txt");
    let out = write_under(dir.path(), &[&target.to_string_lossy(), "hi"], None).await;
    assert_eq!(
        out.exit_code,
        0,
        "{:?}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(std::fs::read(real_dir.join("out.txt")).unwrap(), b"hi");
}

#[test]
fn open_refuses_any_symlink_in_the_path() {
    let dir = tempdir().unwrap();
    let real_dir = dir.path().join("real");
    std::fs::create_dir(&real_dir).unwrap();
    let dir_link = dir.path().join("dir-link");
    std::os::unix::fs::symlink(&real_dir, &dir_link).unwrap();
    let file_link = dir.path().join("file-link");
    std::os::unix::fs::symlink(real_dir.join("target.txt"), &file_link).unwrap();

    for path in [dir_link.join("target.txt"), file_link] {
        open_without_symlinks(&path).expect_err("symlinks must not be followed");
    }
    assert!(!real_dir.join("target.txt").exists());
}

#[tokio::test]
async fn hidden_entries_directly_inside_a_prefix_are_refused() {
    let dir = tempdir().unwrap();
    std::fs::create_dir(dir.path().join(".ssh")).unwrap();
    for target in [
        dir.path().join(".bashrc"),
        dir.path().join(".ssh").join("authorized_keys"),
        dir.path()
            .join(".config")
            .join("autostart")
            .join("x.desktop"),
    ] {
        let target_str = target.to_string_lossy().into_owned();
        let out = write_under(dir.path(), &[&target_str, "oops"], None).await;
        assert_eq!(out.exit_code, POLICY_DENIED_EXIT, "{target_str}");
        assert_eq!(
            String::from_utf8_lossy(&out.stderr),
            format!(
                "[error] write: {target_str}: hidden files and directories are not writable. \
                 Try: asking the user to make this change themselves\n"
            )
        );
        assert!(!target.exists());
    }
}

#[tokio::test]
async fn hidden_entries_deeper_than_a_prefix_are_writable() {
    let dir = tempdir().unwrap();
    let repo = dir.path().join("repo");
    std::fs::create_dir(&repo).unwrap();
    let target = repo.join(".gitignore");
    let out = write_under(dir.path(), &[&target.to_string_lossy(), "target"], None).await;
    assert_eq!(
        out.exit_code,
        0,
        "{:?}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(std::fs::read(&target).unwrap(), b"target");
}

#[tokio::test]
async fn hidden_directory_listed_as_a_prefix_is_writable() {
    let dir = tempdir().unwrap();
    let hidden = dir.path().join(".notes");
    std::fs::create_dir(&hidden).unwrap();
    let target = hidden.join("todo.txt");
    let out = WriteCommand::new(cfg_from(&[dir.path(), hidden.as_path()]))
        .run(CommandInput {
            args: vec![target.to_string_lossy().into_owned(), "hi".into()],
            stdin: None,
        })
        .await;
    assert_eq!(
        out.exit_code,
        0,
        "{:?}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(std::fs::read(&target).unwrap(), b"hi");
}

#[tokio::test]
async fn overwrites_and_truncates_existing_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("out.txt");
    std::fs::write(&path, b"much longer old content").unwrap();

    let out = write_under(dir.path(), &[&path.to_string_lossy(), "new"], None).await;
    assert_eq!(out.exit_code, 0, "{:?}", out.stderr);
    assert_eq!(std::fs::read(&path).unwrap(), b"new");
}

#[tokio::test]
async fn refuses_a_protected_directory_inside_a_writable_one() {
    let dir = tempdir().unwrap();
    let config = dir.path().join("assistd");
    std::fs::create_dir(&config).unwrap();
    let canonical = std::fs::canonicalize(&config).unwrap();
    let cfg = WritePolicyCfg::new(vec![std::fs::canonicalize(dir.path()).unwrap()])
        .expect("non-empty allowlist")
        .protecting(vec![canonical]);
    let target = config.join("config.toml");
    let out = WriteCommand::new(Arc::new(cfg))
        .run(CommandInput {
            args: vec![target.to_string_lossy().into_owned(), "x".into()],
            stdin: None,
        })
        .await;
    assert_eq!(out.exit_code, POLICY_DENIED_EXIT);
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("assistd's own configuration"),
        "{:?}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!target.exists());
}
