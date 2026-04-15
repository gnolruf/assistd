use anyhow::Result;
use async_trait::async_trait;
use regex::Regex;

use crate::command::{Command, CommandInput, CommandOutput};

/// `grep PATTERN [FILE]` — print lines from FILE (or stdin) that match
/// `PATTERN`. Exit 0 on any match, 1 on no matches, 2 on usage/input
/// errors. Flags (`-i`, `-v`, ...) are not supported in v1.
pub struct GrepCommand;

#[async_trait]
impl Command for GrepCommand {
    fn name(&self) -> &str {
        "grep"
    }

    async fn run(&self, input: CommandInput) -> Result<CommandOutput> {
        if input.args.is_empty() {
            return Ok(CommandOutput::failed(2, b"missing pattern\n".to_vec()));
        }
        for a in &input.args {
            if a.starts_with('-') && a.len() > 1 {
                return Ok(CommandOutput::failed(
                    2,
                    format!("flag '{a}' not supported in v1\n").into_bytes(),
                ));
            }
        }
        let pattern = &input.args[0];
        let re = match Regex::new(pattern) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CommandOutput::failed(
                    2,
                    format!("bad pattern: {e}\n").into_bytes(),
                ));
            }
        };

        let content: Vec<u8> = if input.args.len() > 1 {
            let path = &input.args[1];
            match tokio::fs::read(path).await {
                Ok(b) => b,
                Err(e) => {
                    return Ok(CommandOutput::failed(
                        2,
                        format!("{path}: {e}\n").into_bytes(),
                    ));
                }
            }
        } else {
            input.stdin
        };

        let text = match std::str::from_utf8(&content) {
            Ok(s) => s,
            Err(_) => {
                return Ok(CommandOutput::failed(
                    2,
                    b"input is not valid UTF-8\n".to_vec(),
                ));
            }
        };

        let mut out = Vec::new();
        let mut matched = false;
        for line in text.split_inclusive('\n') {
            if re.is_match(line) {
                matched = true;
                out.extend_from_slice(line.as_bytes());
            }
        }
        Ok(CommandOutput {
            stdout: out,
            stderr: Vec::new(),
            exit_code: if matched { 0 } else { 1 },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn grep_matches_from_stdin() {
        let out = GrepCommand
            .run(CommandInput {
                args: vec!["ERROR".into()],
                stdin: b"INFO ok\nERROR boom\nINFO also\n".to_vec(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, b"ERROR boom\n");
    }

    #[tokio::test]
    async fn grep_no_match_exits_1() {
        let out = GrepCommand
            .run(CommandInput {
                args: vec!["ZZZ".into()],
                stdin: b"nothing here\n".to_vec(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 1);
        assert!(out.stdout.is_empty());
    }

    #[tokio::test]
    async fn grep_rejects_flag() {
        let out = GrepCommand
            .run(CommandInput {
                args: vec!["-i".into(), "foo".into()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 2);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("-i"), "{stderr}");
    }

    #[tokio::test]
    async fn grep_missing_pattern_errors() {
        let out = GrepCommand
            .run(CommandInput {
                args: Vec::new(),
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 2);
    }
}
