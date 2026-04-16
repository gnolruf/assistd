use anyhow::Result;
use async_trait::async_trait;

use crate::command::{Command, CommandInput, CommandOutput};

/// `write PATH [CONTENT...]` — write to PATH.
///
/// Two shapes:
/// - `echo "hi" | write /tmp/x` — stdin is the content (pipeline form).
/// - `write /tmp/x hello world` — args beyond the path are joined with
///   single spaces and written (convenience form for when the model
///   already has the content inline).
///
/// If both args and stdin are provided, args win and stdin is silently
/// discarded — callers who want stdin should avoid passing extra argv.
pub struct WriteCommand;

#[async_trait]
impl Command for WriteCommand {
    fn name(&self) -> &str {
        "write"
    }

    fn summary(&self) -> &'static str {
        "write a file from stdin, or from inline args joined by spaces"
    }

    fn help(&self) -> String {
        "usage: write PATH [CONTENT...]\n\
         \n\
         Write bytes to PATH. Two shapes:\n  \
           `echo \"hi\" | write /tmp/x`   — stdin is the file content (pipeline form)\n  \
           `write /tmp/x hello world`   — args beyond PATH are joined by spaces and written\n\
         \n\
         If both args and stdin are provided, args win and stdin is \
         silently discarded. Exit 1 on write failure (permissions, no-such-dir, etc.).\n"
            .to_string()
    }

    async fn run(&self, input: CommandInput) -> Result<CommandOutput> {
        if input.args.is_empty() {
            return Ok(CommandOutput {
                stdout: self.help().into_bytes(),
                stderr: Vec::new(),
                exit_code: 2,
                attachments: Vec::new(),
            });
        }
        let path = &input.args[0];
        let content: Vec<u8> = if input.args.len() > 1 {
            input.args[1..].join(" ").into_bytes()
        } else {
            input.stdin
        };
        match tokio::fs::write(path, &content).await {
            Ok(()) => Ok(CommandOutput::ok(Vec::new())),
            Err(e) => Ok(CommandOutput::failed(
                1,
                format!("{path}: {e}\n").into_bytes(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn persists_stdin_to_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("out.txt");
        let out = WriteCommand
            .run(CommandInput {
                args: vec![path.to_string_lossy().into_owned()],
                stdin: b"hi there\n".to_vec(),
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
        let out = WriteCommand
            .run(CommandInput {
                args: vec![
                    path.to_string_lossy().into_owned(),
                    "hello".into(),
                    "world".into(),
                ],
                stdin: Vec::new(),
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
        let out = WriteCommand
            .run(CommandInput {
                args: vec![path.to_string_lossy().into_owned(), "args".into()],
                stdin: b"stdin".to_vec(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 0);
        let content = tokio::fs::read(&path).await.unwrap();
        assert_eq!(content, b"args");
    }

    #[tokio::test]
    async fn no_args_errors() {
        let out = WriteCommand
            .run(CommandInput {
                args: Vec::new(),
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 2);
    }

    #[tokio::test]
    async fn path_only_with_empty_stdin_creates_empty_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("out.txt");
        let out = WriteCommand
            .run(CommandInput {
                args: vec![path.to_string_lossy().into_owned()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 0);
        let content = tokio::fs::read(&path).await.unwrap();
        assert!(content.is_empty());
    }

    #[tokio::test]
    async fn unwritable_path_exits_1() {
        let out = WriteCommand
            .run(CommandInput {
                args: vec!["/definitely/not/a/writable/path".into(), "hi".into()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 1);
    }
}
