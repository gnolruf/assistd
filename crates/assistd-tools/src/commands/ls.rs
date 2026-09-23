use std::io::ErrorKind;

use async_trait::async_trait;

use crate::command::{Command, CommandInput, CommandOutput, io_error_nav};

/// `ls [-al] [PATH]`: list directory entries alphabetically, one per
/// line, formatted as `<type>\t<size>\t<name>`. Type is `dir`, `file`,
/// or `symlink`; size is raw bytes from the entry's (symlink-preserving)
/// metadata. A PATH that is not a directory yields a single row named
/// by PATH as given. Defaults to the daemon's CWD if no path given.
///
/// Flags:
/// - `-a` include entries whose name starts with `.`
/// - `-l` long format; accepted for shell familiarity, and already the
///   only format this command emits
pub struct LsCommand;

fn parse_flags(argv: &[String]) -> Result<(bool, &str), String> {
    let mut show_hidden = false;
    let mut path = None;
    for arg in argv {
        match arg.strip_prefix('-') {
            Some(flags) if !flags.is_empty() && path.is_none() => {
                for ch in flags.chars() {
                    match ch {
                        'a' => show_hidden = true,
                        'l' => {}
                        other => return Err(format!("unknown flag '-{other}'")),
                    }
                }
            }
            _ => path = Some(arg.as_str()),
        }
    }
    Ok((show_hidden, path.unwrap_or(".")))
}

fn kind_and_size(md: &std::fs::Metadata) -> (&'static str, u64) {
    let ft = md.file_type();
    let kind = if ft.is_symlink() {
        "symlink"
    } else if ft.is_dir() {
        "dir"
    } else {
        "file"
    };
    (kind, md.len())
}

async fn list_file(path: &str) -> CommandOutput {
    match tokio::fs::symlink_metadata(path).await {
        Ok(md) => {
            let (kind, size) = kind_and_size(&md);
            CommandOutput::ok(format!("{kind}\t{size}\t{path}\n").into_bytes())
        }
        Err(e) => CommandOutput::failed(1, io_error_nav("ls", path, &e).into_bytes()),
    }
}

#[async_trait]
impl Command for LsCommand {
    fn name(&self) -> &str {
        "ls"
    }

    fn summary(&self) -> &'static str {
        "list a directory, or one file (type, size, name); -a for dot-entries"
    }

    fn help(&self) -> String {
        "usage: ls [-al] [PATH]\n\
         \n\
         List directory entries alphabetically, one per line, formatted as \
         `<type>\\t<size>\\t<name>`. `<type>` is `dir`, `file`, or `symlink`; \
         `<size>` is bytes from the entry's (symlink-preserving) metadata. \
         A PATH that is not a directory prints one row for that path. \
         Defaults to the daemon's CWD if PATH is omitted.\n\
         \n\
         Flags:\n  \
           -a  include dot-entries (hidden by default)\n  \
           -l  long format (already the only format; accepted for familiarity)\n"
            .to_string()
    }

    async fn run(&self, input: CommandInput) -> CommandOutput {
        let (show_hidden, path) = match parse_flags(&input.args) {
            Ok(v) => v,
            Err(msg) => return CommandOutput::usage_error("ls", msg, "ls -al PATH"),
        };
        let mut reader = match tokio::fs::read_dir(path).await {
            Ok(r) => r,
            Err(e) if e.kind() == ErrorKind::NotADirectory => return list_file(path).await,
            Err(e) => {
                return CommandOutput::failed(1, io_error_nav("ls", path, &e).into_bytes());
            }
        };
        let mut rows: Vec<(String, &'static str, u64)> = Vec::new();
        loop {
            match reader.next_entry().await {
                Ok(Some(entry)) => {
                    let name = entry.file_name().to_string_lossy().into_owned();
                    if !show_hidden && name.starts_with('.') {
                        continue;
                    }
                    let (kind, size) = tokio::fs::symlink_metadata(entry.path())
                        .await
                        .map_or(("file", 0), |md| kind_and_size(&md));
                    rows.push((name, kind, size));
                }
                Ok(None) => break,
                Err(e) => {
                    return CommandOutput::failed(1, io_error_nav("ls", path, &e).into_bytes());
                }
            }
        }
        rows.sort_by(|a, b| a.0.cmp(&b.0));
        let mut out = Vec::new();
        for (name, kind, size) in rows {
            out.extend_from_slice(format!("{kind}\t{size}\t{name}\n").as_bytes());
        }
        CommandOutput::ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    async fn run_ls(args: &[&str]) -> CommandOutput {
        LsCommand
            .run(CommandInput {
                args: args.iter().map(|s| s.to_string()).collect(),
                stdin: None,
            })
            .await
    }

    #[tokio::test]
    async fn ls_emits_type_size_name_rows() {
        let dir = tempdir().unwrap();
        tokio::fs::write(dir.path().join("alpha"), b"hi")
            .await
            .unwrap();
        tokio::fs::write(dir.path().join("zebra"), b"longer content here")
            .await
            .unwrap();
        tokio::fs::create_dir(dir.path().join("nested"))
            .await
            .unwrap();
        let out = LsCommand
            .run(CommandInput {
                args: vec![dir.path().to_string_lossy().into_owned()],
                stdin: None,
            })
            .await;
        assert_eq!(out.exit_code, 0);
        let stdout = String::from_utf8_lossy(&out.stdout);
        let lines: Vec<&str> = stdout.lines().collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], "file\t2\talpha");
        assert!(lines[1].starts_with("dir\t"));
        assert!(lines[1].ends_with("\tnested"));
        assert_eq!(lines[2], "file\t19\tzebra");
    }

    #[tokio::test]
    async fn ls_hides_dot_entries_until_dash_a() {
        let dir = tempdir().unwrap();
        tokio::fs::write(dir.path().join("visible"), b"x")
            .await
            .unwrap();
        tokio::fs::write(dir.path().join(".hidden"), b"x")
            .await
            .unwrap();
        let path = dir.path().to_string_lossy().into_owned();

        let plain = run_ls(&[&path]).await;
        let stdout = String::from_utf8_lossy(&plain.stdout);
        assert!(!stdout.contains(".hidden"), "{stdout}");
        assert!(stdout.contains("visible"), "{stdout}");

        let all = run_ls(&["-a", &path]).await;
        let stdout = String::from_utf8_lossy(&all.stdout);
        assert!(stdout.contains(".hidden"), "{stdout}");
    }

    #[tokio::test]
    async fn ls_accepts_combined_la_before_path() {
        let dir = tempdir().unwrap();
        tokio::fs::write(dir.path().join(".dotfile"), b"x")
            .await
            .unwrap();
        let out = run_ls(&["-la", &dir.path().to_string_lossy()]).await;
        assert_eq!(out.exit_code, 0);
        assert!(
            String::from_utf8_lossy(&out.stdout).contains(".dotfile"),
            "{out:?}"
        );
    }

    #[tokio::test]
    async fn ls_unknown_flag_errors() {
        let out = run_ls(&["-Z", "/tmp"]).await;
        assert_eq!(out.exit_code, 2);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("[error] ls: unknown flag '-Z'"), "{stderr}");
        assert!(stderr.contains("Use: "), "{stderr}");
    }

    #[tokio::test]
    async fn ls_treats_dash_prefixed_path_after_path_as_path() {
        // A second positional wins over flag parsing so a file literally
        // named `-weird` stays reachable once a path is already set.
        let out = run_ls(&["/tmp", "-weird"]).await;
        assert_eq!(out.exit_code, 1);
        assert!(
            String::from_utf8_lossy(&out.stderr).contains("-weird"),
            "{out:?}"
        );
    }

    #[tokio::test]
    async fn ls_reports_symlink_without_following() {
        let dir = tempdir().unwrap();
        let target = dir.path().join("target.txt");
        tokio::fs::write(&target, b"hello").await.unwrap();
        let link = dir.path().join("link");
        #[cfg(unix)]
        std::os::unix::fs::symlink(&target, &link).unwrap();
        let out = LsCommand
            .run(CommandInput {
                args: vec![dir.path().to_string_lossy().into_owned()],
                stdin: None,
            })
            .await;
        assert_eq!(out.exit_code, 0);
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(stdout.contains("symlink\t"), "{stdout}");
        assert!(stdout.contains("\tlink\n"), "{stdout}");
    }

    #[tokio::test]
    async fn ls_missing_dir_exits_1() {
        let out = LsCommand
            .run(CommandInput {
                args: vec!["/definitely/not/here".into()],
                stdin: None,
            })
            .await;
        assert_eq!(out.exit_code, 1);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("[error] ls: file not found: /definitely/not/here"),
            "{stderr}"
        );
        assert!(
            stderr.contains("Use: ls /definitely/not to see what is there"),
            "{stderr}"
        );
    }

    #[tokio::test]
    async fn ls_file_emits_a_single_row_named_by_the_given_path() {
        let dir = tempdir().unwrap();
        let file = dir.path().join(".notes.txt");
        tokio::fs::write(&file, b"hello").await.unwrap();
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
        tokio::fs::write(&target, b"hello").await.unwrap();
        let link = dir.path().join("link");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        let path = link.to_string_lossy().into_owned();

        let out = run_ls(&[&path]).await;
        assert_eq!(out.exit_code, 0, "{out:?}");
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(stdout.starts_with("symlink\t"), "{stdout}");
        assert!(stdout.ends_with(&format!("\t{path}\n")), "{stdout}");
    }

    #[tokio::test]
    async fn ls_path_through_a_file_names_the_offending_parent() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("notes.txt");
        tokio::fs::write(&file, b"hello").await.unwrap();
        let file = file.to_string_lossy().into_owned();

        let out = run_ls(&[&format!("{file}/sub")]).await;
        assert_eq!(out.exit_code, 1);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("a parent component is not a directory"),
            "{stderr}"
        );
        assert!(stderr.contains(&format!("Check: ls {file}\n")), "{stderr}");
    }
}
