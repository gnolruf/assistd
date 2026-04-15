use anyhow::Result;
use async_trait::async_trait;

use crate::command::{Command, CommandInput, CommandOutput};

/// `write PATH` — write stdin to PATH. Fails with a teaching message
/// if extra argv is passed; enforces the "content comes from stdin"
/// convention so pipelines like `grep ERROR log | write err.txt` work.
pub struct WriteCommand;

#[async_trait]
impl Command for WriteCommand {
    fn name(&self) -> &str {
        "write"
    }

    async fn run(&self, input: CommandInput) -> Result<CommandOutput> {
        if input.args.is_empty() {
            return Ok(CommandOutput::failed(
                2,
                b"missing path argument\n".to_vec(),
            ));
        }
        if input.args.len() > 1 {
            return Ok(CommandOutput::failed(
                2,
                b"content comes from stdin; use 'echo \"hello\" | write file.txt'\n".to_vec(),
            ));
        }
        let path = &input.args[0];
        match tokio::fs::write(path, &input.stdin).await {
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
    async fn write_persists_stdin_to_file() {
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
    async fn write_rejects_content_as_arg() {
        let out = WriteCommand
            .run(CommandInput {
                args: vec!["/tmp/x".into(), "hi".into()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 2);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("stdin"), "{stderr}");
    }

    #[tokio::test]
    async fn write_no_args_errors() {
        let out = WriteCommand
            .run(CommandInput {
                args: Vec::new(),
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 2);
    }
}
