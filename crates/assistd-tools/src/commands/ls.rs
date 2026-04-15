use anyhow::Result;
use async_trait::async_trait;

use crate::command::{Command, CommandInput, CommandOutput};

/// `ls [PATH]` — list directory entries alphabetically, one per line.
/// Defaults to the daemon's CWD if no path given.
pub struct LsCommand;

#[async_trait]
impl Command for LsCommand {
    fn name(&self) -> &str {
        "ls"
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
        let mut names: Vec<String> = Vec::new();
        loop {
            match reader.next_entry().await {
                Ok(Some(entry)) => {
                    names.push(entry.file_name().to_string_lossy().into_owned());
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
        names.sort();
        let mut out = Vec::new();
        for n in names {
            out.extend_from_slice(n.as_bytes());
            out.push(b'\n');
        }
        Ok(CommandOutput::ok(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn ls_lists_sorted_entries() {
        let dir = tempdir().unwrap();
        tokio::fs::write(dir.path().join("zebra"), b"")
            .await
            .unwrap();
        tokio::fs::write(dir.path().join("alpha"), b"")
            .await
            .unwrap();
        tokio::fs::write(dir.path().join("mango"), b"")
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
        assert_eq!(out.stdout, b"alpha\nmango\nzebra\n");
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
