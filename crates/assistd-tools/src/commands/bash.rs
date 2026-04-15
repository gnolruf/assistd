//! The escape hatch: spawns a real `bash -c <script>` subprocess.
//!
//! This command reintroduces every filesystem/network concern that the
//! in-process commands carefully avoid. Sandboxing is deferred to a
//! later milestone; the daemon's current trust model assumes single-user
//! desktop + LLM-as-user. Callers who care should disable this command
//! by not registering it.
//!
//! The spawn pattern (`kill_on_drop(true)` + `process_group(0)` on Unix)
//! mirrors `assistd-llm/src/llama_server/process.rs` — any mid-command
//! daemon shutdown kills the whole process group, preventing grandchild
//! leaks if the script itself forked.

use std::process::Stdio;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use tokio::io::AsyncWriteExt;
use tokio::process::Command as ProcCommand;
use tokio::time::timeout;

use crate::command::{Command, CommandInput, CommandOutput};

pub struct BashCommand {
    pub timeout: Duration,
}

impl BashCommand {
    pub fn new(timeout: Duration) -> Self {
        Self { timeout }
    }
}

impl Default for BashCommand {
    fn default() -> Self {
        Self::new(Duration::from_secs(30))
    }
}

#[async_trait]
impl Command for BashCommand {
    fn name(&self) -> &str {
        "bash"
    }

    async fn run(&self, input: CommandInput) -> Result<CommandOutput> {
        if input.args.is_empty() {
            return Ok(CommandOutput::failed(
                2,
                b"missing script argument; use: bash \"<script>\"\n".to_vec(),
            ));
        }
        let script = input.args.join(" ");

        let mut cmd = ProcCommand::new("bash");
        cmd.arg("-c")
            .arg(&script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        #[cfg(unix)]
        cmd.process_group(0);

        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                return Ok(CommandOutput::failed(
                    127,
                    format!("spawn failed: {e}\n").into_bytes(),
                ));
            }
        };

        if let Some(mut stdin) = child.stdin.take() {
            if !input.stdin.is_empty() {
                let _ = stdin.write_all(&input.stdin).await;
            }
            // Drop closes the pipe, signaling EOF to the child.
            drop(stdin);
        }

        let output = match timeout(self.timeout, child.wait_with_output()).await {
            Ok(Ok(o)) => o,
            Ok(Err(e)) => {
                return Ok(CommandOutput::failed(
                    1,
                    format!("wait failed: {e}\n").into_bytes(),
                ));
            }
            Err(_) => {
                return Ok(CommandOutput::failed(
                    124, // GNU timeout convention
                    format!("script exceeded {}s timeout\n", self.timeout.as_secs()).into_bytes(),
                ));
            }
        };

        let exit_code = output.status.code().unwrap_or(137);
        Ok(CommandOutput {
            stdout: output.stdout,
            stderr: output.stderr,
            exit_code,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn bash_runs_echo() {
        let out = BashCommand::default()
            .run(CommandInput {
                args: vec!["echo hi".into()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, b"hi\n");
    }

    #[tokio::test]
    async fn bash_propagates_nonzero_exit() {
        let out = BashCommand::default()
            .run(CommandInput {
                args: vec!["exit 3".into()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 3);
    }

    #[tokio::test]
    async fn bash_receives_stdin() {
        let out = BashCommand::default()
            .run(CommandInput {
                args: vec!["tr a-z A-Z".into()],
                stdin: b"hello".to_vec(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, b"HELLO");
    }

    #[tokio::test]
    async fn bash_timeout_returns_124() {
        let cmd = BashCommand::new(Duration::from_millis(100));
        let out = cmd
            .run(CommandInput {
                args: vec!["sleep 5".into()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 124);
    }

    #[tokio::test]
    async fn bash_missing_script_errors() {
        let out = BashCommand::default()
            .run(CommandInput {
                args: Vec::new(),
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 2);
    }
}
