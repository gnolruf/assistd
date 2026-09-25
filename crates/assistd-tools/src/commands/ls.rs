use std::fs::Metadata;
use std::io::{self, ErrorKind};

use async_trait::async_trait;
use tokio::fs::ReadDir;

use crate::command::{Command, CommandInput, CommandOutput, io_error_nav};

/// `ls [-al] [PATH]`: list PATH (default CWD) as sorted
/// `<type>\t<size>\t<name>` rows, without following symlinks.
pub struct LsCommand;

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
        let reader = match tokio::fs::read_dir(path).await {
            Ok(r) => r,
            Err(e) if e.kind() == ErrorKind::NotADirectory => return list_file(path).await,
            Err(e) => {
                return CommandOutput::failed(1, io_error_nav("ls", path, &e).into_bytes());
            }
        };
        let mut rows = match read_rows(reader, show_hidden).await {
            Ok(rows) => rows,
            Err(e) => {
                return CommandOutput::failed(1, io_error_nav("ls", path, &e).into_bytes());
            }
        };
        rows.sort_by(|a, b| a.0.cmp(&b.0));
        let mut out = Vec::new();
        for (name, kind, size) in rows {
            out.extend_from_slice(format!("{kind}\t{size}\t{name}\n").as_bytes());
        }
        CommandOutput::ok(out)
    }
}

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

fn kind_and_size(md: &Metadata) -> (&'static str, u64) {
    let file_type = md.file_type();
    let kind = if file_type.is_symlink() {
        "symlink"
    } else if file_type.is_dir() {
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

async fn read_rows(
    mut reader: ReadDir,
    show_hidden: bool,
) -> io::Result<Vec<(String, &'static str, u64)>> {
    let mut rows = Vec::new();
    while let Some(entry) = reader.next_entry().await? {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !show_hidden && name.starts_with('.') {
            continue;
        }
        let (kind, size) = tokio::fs::symlink_metadata(entry.path())
            .await
            .map_or(("file", 0), |md| kind_and_size(&md));
        rows.push((name, kind, size));
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

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
}
