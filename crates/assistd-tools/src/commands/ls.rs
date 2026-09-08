use anyhow::Result;
use async_trait::async_trait;

use crate::command::{Command, CommandInput, CommandOutput, error_line, io_error_nav};

/// `ls [-al] [PATH]`: list directory entries alphabetically, one per
/// line, formatted as `<type>\t<size>\t<name>`. Type is `dir`, `file`,
/// or `symlink`; size is raw bytes from the entry's (symlink-preserving)
/// metadata. Defaults to the daemon's CWD if no path given.
///
/// Flags:
/// - `-a` include entries whose name starts with `.`
/// - `-l` long format; accepted for shell familiarity, and already the
///   only format this command emits
pub struct LsCommand;

fn parse_args(argv: &[String]) -> Result<(bool, &str), String> {
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

#[async_trait]
impl Command for LsCommand {
    fn name(&self) -> &str {
        "ls"
    }

    fn summary(&self) -> &'static str {
        "list directory entries (type, size, name); -a for dot-entries"
    }

    fn help(&self) -> String {
        "usage: ls [-al] [PATH]\n\
         \n\
         List directory entries alphabetically, one per line, formatted as \
         `<type>\\t<size>\\t<name>`. `<type>` is `dir`, `file`, or `symlink`; \
         `<size>` is bytes from the entry's (symlink-preserving) metadata. \
         Defaults to the daemon's CWD if PATH is omitted.\n\
         \n\
         Flags:\n  \
           -a  include dot-entries (hidden by default)\n  \
           -l  long format (already the only format; accepted for familiarity)\n"
            .to_string()
    }

    async fn run(&self, input: CommandInput) -> Result<CommandOutput> {
        let (show_hidden, path) = match parse_args(&input.args) {
            Ok(v) => v,
            Err(msg) => {
                return Ok(CommandOutput::failed(
                    2,
                    error_line("ls", msg, "Use", "ls -al PATH").into_bytes(),
                ));
            }
        };
        let mut reader = match tokio::fs::read_dir(path).await {
            Ok(r) => r,
            Err(e) => {
                return Ok(CommandOutput::failed(
                    1,
                    io_error_nav("ls", path, &e).into_bytes(),
                ));
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
                    let (kind, size) = match tokio::fs::symlink_metadata(entry.path()).await {
                        Ok(md) => {
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
                        Err(_) => ("file", 0),
                    };
                    rows.push((name, kind, size));
                }
                Ok(None) => break,
                Err(e) => {
                    return Ok(CommandOutput::failed(
                        1,
                        io_error_nav("ls", path, &e).into_bytes(),
                    ));
                }
            }
        }
        rows.sort_by(|a, b| a.0.cmp(&b.0));
        let mut out = Vec::new();
        for (name, kind, size) in rows {
            out.extend_from_slice(format!("{kind}\t{size}\t{name}\n").as_bytes());
        }
        Ok(CommandOutput::ok(out))
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
            .expect("ls runs")
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
            .await
            .unwrap();
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
            .await
            .unwrap();
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
            .await
            .unwrap();
        assert_eq!(out.exit_code, 1);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("[error] ls: file not found: /definitely/not/here"),
            "{stderr}"
        );
        assert!(stderr.contains("Use: ls to check the path"), "{stderr}");
    }
}
