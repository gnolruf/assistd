use anyhow::Result;
use async_trait::async_trait;

use crate::command::{Command, CommandInput, CommandOutput};

/// `echo [ARGS...]` — write args joined by spaces, followed by a newline.
pub struct EchoCommand;

#[async_trait]
impl Command for EchoCommand {
    fn name(&self) -> &str {
        "echo"
    }

    fn summary(&self) -> &'static str {
        "write arguments joined by spaces, followed by a newline"
    }

    fn help(&self) -> String {
        "usage: echo [ARGS...]\n\
         \n\
         Write the argument list joined by single spaces, followed by a \
         newline. With no arguments, emits a bare newline.\n"
            .to_string()
    }

    async fn run(&self, input: CommandInput) -> Result<CommandOutput> {
        let mut out = input.args.join(" ").into_bytes();
        out.push(b'\n');
        Ok(CommandOutput::ok(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn echo_joins_args_with_space_and_newline() {
        let out = EchoCommand
            .run(CommandInput {
                args: vec!["hello".into(), "world".into()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.stdout, b"hello world\n");
        assert_eq!(out.exit_code, 0);
    }

    #[tokio::test]
    async fn echo_no_args_is_bare_newline() {
        let out = EchoCommand
            .run(CommandInput {
                args: Vec::new(),
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.stdout, b"\n");
    }
}
