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

    for flags in ["-la"] {
        let all = run_ls(&[flags, &path]).await;
        assert_eq!(all.exit_code, 0, "{flags}");
        assert_eq!(
            String::from_utf8_lossy(&all.stdout),
            "file\t1\t.hidden\nfile\t1\tvisible\n",
            "{flags}"
        );
    }
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

#[cfg(unix)]
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
