use tempfile::tempdir;

use super::*;

async fn run_ls(args: &[&str]) -> CommandOutput {
    LsCommand
        .run(CommandInput {
            args: args.iter().map(ToString::to_string).collect(),
            stdin: None,
        })
        .await
}

#[tokio::test]
async fn ls_emits_type_size_name_rows() {
    let dir = tempdir().unwrap();
    std::fs::write(dir.path().join("alpha"), b"hi").unwrap();
    std::fs::write(dir.path().join("zebra"), b"longer content here").unwrap();
    std::fs::create_dir(dir.path().join("nested")).unwrap();
    let out = run_ls(&[&dir.path().to_string_lossy()]).await;
    assert_eq!(out.exit_code, 0);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = stdout.lines().collect();
    let &[alpha, nested, zebra] = lines.as_slice() else {
        panic!("expected three rows: {stdout}");
    };
    assert_eq!(alpha, "file\t2\talpha");
    assert!(
        nested.starts_with("dir\t") && nested.ends_with("\tnested"),
        "{nested}"
    );
    assert_eq!(zebra, "file\t19\tzebra");
}

#[tokio::test]
async fn ls_hides_dot_entries_until_dash_a() {
    let dir = tempdir().unwrap();
    std::fs::write(dir.path().join("visible"), b"x").unwrap();
    std::fs::write(dir.path().join(".hidden"), b"x").unwrap();
    let path = dir.path().to_string_lossy().into_owned();

    let plain = run_ls(&[&path]).await;
    assert_eq!(plain.stdout, b"file\t1\tvisible\n");

    for flags in ["-a", "-la"] {
        let all = run_ls(&[flags, &path]).await;
        assert_eq!(all.exit_code, 0, "{flags}");
        assert_eq!(
            String::from_utf8_lossy(&all.stdout),
            "file\t1\t.hidden\nfile\t1\tvisible\n",
            "{flags}"
        );
    }
}

#[tokio::test]
async fn ls_unknown_flag_errors() {
    let out = run_ls(&["-Z", "/tmp"]).await;
    assert_eq!(out.exit_code, 2);
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "[error] ls: unknown flag '-Z'. Use: ls -al PATH\n"
    );
}

#[tokio::test]
async fn ls_treats_dash_prefixed_path_after_path_as_path() {
    let out = run_ls(&["/tmp", "-weird"]).await;
    assert_eq!(out.exit_code, 1);
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "[error] ls: file not found: -weird. Use: ls . to see what is there\n"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn ls_reports_symlink_without_following() {
    let dir = tempdir().unwrap();
    let target = dir.path().join("target.txt");
    std::fs::write(&target, b"hello").unwrap();
    std::os::unix::fs::symlink(&target, dir.path().join("link")).unwrap();
    let out = run_ls(&[&dir.path().to_string_lossy()]).await;
    assert_eq!(out.exit_code, 0);
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        format!(
            "symlink\t{}\tlink\nfile\t5\ttarget.txt\n",
            target.as_os_str().len()
        )
    );
}

#[tokio::test]
async fn ls_missing_dir_exits_1() {
    let out = run_ls(&["/definitely/not/here"]).await;
    assert_eq!(out.exit_code, 1);
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        "[error] ls: file not found: /definitely/not/here. \
         Use: ls /definitely/not to see what is there\n"
    );
}

#[tokio::test]
async fn ls_file_emits_a_single_row_named_by_the_given_path() {
    let dir = tempdir().unwrap();
    let file = dir.path().join(".notes.txt");
    std::fs::write(&file, b"hello").unwrap();
    let path = file.to_string_lossy().into_owned();

    let out = run_ls(&[&path]).await;
    assert_eq!(out.exit_code, 0, "{out:?}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        format!("file\t5\t{path}\n")
    );
}

#[cfg(unix)]
#[tokio::test]
async fn ls_symlink_to_file_reports_the_link_itself() {
    let dir = tempdir().unwrap();
    let target = dir.path().join("target.txt");
    std::fs::write(&target, b"hello").unwrap();
    let link = dir.path().join("link");
    std::os::unix::fs::symlink(&target, &link).unwrap();
    let path = link.to_string_lossy().into_owned();

    let out = run_ls(&[&path]).await;
    assert_eq!(out.exit_code, 0, "{out:?}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        format!("symlink\t{}\t{path}\n", target.as_os_str().len())
    );
}

#[tokio::test]
async fn ls_path_through_a_file_names_the_offending_parent() {
    let dir = tempdir().unwrap();
    let file = dir.path().join("notes.txt");
    std::fs::write(&file, b"hello").unwrap();
    let file = file.to_string_lossy().into_owned();

    let out = run_ls(&[&format!("{file}/sub")]).await;
    assert_eq!(out.exit_code, 1);
    assert_eq!(
        String::from_utf8_lossy(&out.stderr),
        format!(
            "[error] ls: {file}/sub: a parent component is not a directory. Check: ls {file}\n"
        )
    );
}

#[tokio::test]
async fn ls_lists_every_directory_under_a_header_sorted_by_path() {
    let root = tempdir().unwrap();
    for (dir, file, contents) in [("first", "a", "x"), ("second", "b", "yy")] {
        std::fs::create_dir(root.path().join(dir)).unwrap();
        std::fs::write(root.path().join(dir).join(file), contents).unwrap();
    }
    let first = root.path().join("first").to_string_lossy().into_owned();
    let second = root.path().join("second").to_string_lossy().into_owned();

    let out = run_ls(&[&second, &first]).await;
    assert_eq!(out.exit_code, 0, "{out:?}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        format!("{first}:\nfile\t1\ta\n\n{second}:\nfile\t2\tb\n")
    );
}

#[tokio::test]
async fn ls_sorts_file_operands_by_path() {
    let root = tempdir().unwrap();
    std::fs::write(root.path().join("alpha"), b"x").unwrap();
    std::fs::write(root.path().join("beta"), b"yy").unwrap();
    let alpha = root.path().join("alpha").to_string_lossy().into_owned();
    let beta = root.path().join("beta").to_string_lossy().into_owned();

    let out = run_ls(&[&beta, &alpha]).await;
    assert_eq!(out.exit_code, 0, "{out:?}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        format!("file\t1\t{alpha}\nfile\t2\t{beta}\n")
    );
}

#[tokio::test]
async fn ls_puts_file_rows_before_directory_listings() {
    let dir = tempdir().unwrap();
    std::fs::write(dir.path().join("inner"), b"x").unwrap();
    let other = tempdir().unwrap();
    let file = other.path().join("notes.txt");
    std::fs::write(&file, b"hello").unwrap();
    let dir = dir.path().to_string_lossy().into_owned();
    let file = file.to_string_lossy().into_owned();

    let out = run_ls(&[&dir, &file]).await;
    assert_eq!(out.exit_code, 0, "{out:?}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        format!("file\t5\t{file}\n\n{dir}:\nfile\t1\tinner\n")
    );
}

#[tokio::test]
async fn ls_keeps_listing_after_a_missing_path() {
    let dir = tempdir().unwrap();
    std::fs::write(dir.path().join("present"), b"x").unwrap();
    let dir = dir.path().to_string_lossy().into_owned();

    let out = run_ls(&["/definitely/not/here", &dir]).await;
    assert_eq!(out.exit_code, 1);
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        format!("{dir}:\nfile\t1\tpresent\n")
    );
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("/definitely/not/here"),
        "{out:?}"
    );
}
