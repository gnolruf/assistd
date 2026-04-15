use anyhow::Result;
use async_trait::async_trait;

use crate::command::{Command, CommandInput, CommandOutput};

/// `cat [FILE]...` — concatenate files, or echo stdin if no files given.
pub struct CatCommand;

#[async_trait]
impl Command for CatCommand {
    fn name(&self) -> &str {
        "cat"
    }

    async fn run(&self, input: CommandInput) -> Result<CommandOutput> {
        if input.args.is_empty() {
            return Ok(CommandOutput::ok(input.stdin));
        }
        let mut out = Vec::new();
        for path in &input.args {
            match tokio::fs::read(path).await {
                Ok(bytes) => out.extend_from_slice(&bytes),
                Err(e) => {
                    return Ok(CommandOutput::failed(
                        1,
                        format!("{path}: {e}\n").into_bytes(),
                    ));
                }
            }
        }
        Ok(CommandOutput::ok(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn cat_reads_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("hello.txt");
        tokio::fs::write(&path, b"hello world\n").await.unwrap();
        let out = CatCommand
            .run(CommandInput {
                args: vec![path.to_string_lossy().into_owned()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.stdout, b"hello world\n");
        assert_eq!(out.exit_code, 0);
    }

    #[tokio::test]
    async fn cat_concatenates_multiple_files() {
        let dir = tempdir().unwrap();
        let a = dir.path().join("a.txt");
        let b = dir.path().join("b.txt");
        tokio::fs::write(&a, b"A").await.unwrap();
        tokio::fs::write(&b, b"B").await.unwrap();
        let out = CatCommand
            .run(CommandInput {
                args: vec![
                    a.to_string_lossy().into_owned(),
                    b.to_string_lossy().into_owned(),
                ],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.stdout, b"AB");
    }

    #[tokio::test]
    async fn cat_missing_file_exits_1() {
        let out = CatCommand
            .run(CommandInput {
                args: vec!["/nonexistent/path/xyz".into()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 1);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("/nonexistent/path/xyz"), "{stderr}");
    }

    #[tokio::test]
    async fn cat_no_args_echoes_stdin() {
        let out = CatCommand
            .run(CommandInput {
                args: Vec::new(),
                stdin: b"from stdin".to_vec(),
            })
            .await
            .unwrap();
        assert_eq!(out.stdout, b"from stdin");
    }
}
