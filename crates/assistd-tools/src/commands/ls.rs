use anyhow::Result;
use async_trait::async_trait;

use crate::command::{Command, CommandInput, CommandOutput};

/// `ls [PATH]` — list directory entries alphabetically, one per line,
/// formatted as `<type>\t<size>\t<name>`. Type is `dir`, `file`, or
/// `symlink`; size is raw bytes from the entry's (symlink-preserving)
/// metadata. Defaults to the daemon's CWD if no path given.
pub struct LsCommand;

#[async_trait]
impl Command for LsCommand {
    fn name(&self) -> &str {
        "ls"
    }

    fn summary(&self) -> &'static str {
        "list directory entries (type, size, name); defaults to cwd"
    }

    fn help(&self) -> String {
        "usage: ls [PATH]\n\
         \n\
         List directory entries alphabetically, one per line, formatted as \
         `<type>\\t<size>\\t<name>`. `<type>` is `dir`, `file`, or `symlink`; \
         `<size>` is bytes from the entry's (symlink-preserving) metadata. \
         Defaults to the daemon's CWD if PATH is omitted.\n"
            .to_string()
    }

    async fn run(&self, input: CommandInput) -> Result<CommandOutput> {
        let path = input.args.first().map(|s| s.as_str()).unwrap_or(".");
        let mut reader = match tokio::fs::read_dir(path).await {
            Ok(r) => r,
            Err(e) => {
                return Ok(CommandOutput::failed(
                    1,
                    format!("{path}: {e}\n").into_bytes(),
                ));
            }
        };
        let mut rows: Vec<(String, &'static str, u64)> = Vec::new();
        loop {
            match reader.next_entry().await {
                Ok(Some(entry)) => {
                    let name = entry.file_name().to_string_lossy().into_owned();
                    // `symlink_metadata` so dangling symlinks don't error the whole listing.
                    let (kind, size) = match entry.path().symlink_metadata() {
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
                        format!("{path}: {e}\n").into_bytes(),
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
                stdin: Vec::new(),
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
                stdin: Vec::new(),
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
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 1);
    }
}
