use anyhow::Result;
use async_trait::async_trait;

use crate::command::{Command, CommandInput, CommandOutput, error_line};

/// `wc [-l]` — count lines (`-l`) or "lines words bytes" (default) from
/// stdin. Only `-l` is accepted in v1; other flags return exit 2.
pub struct WcCommand;

#[async_trait]
impl Command for WcCommand {
    fn name(&self) -> &str {
        "wc"
    }

    fn summary(&self) -> &'static str {
        "count lines/words/bytes on stdin; -l for lines only"
    }

    fn help(&self) -> String {
        "usage: wc [-l]\n\
         \n\
         Count newlines, whitespace-separated words, and bytes read from \
         stdin. Output is `<lines> <words> <bytes>` on one line.\n\
         \n\
         Flags:\n  \
           -l  print only the line count\n"
            .to_string()
    }

    async fn run(&self, input: CommandInput) -> Result<CommandOutput> {
        let mut lines_only = false;
        for a in &input.args {
            match a.as_str() {
                "-l" => lines_only = true,
                other => {
                    return Ok(CommandOutput::failed(
                        2,
                        error_line(
                            "wc",
                            format_args!("flag '{other}' not supported in v1"),
                            "Use",
                            "wc or wc -l",
                        )
                        .into_bytes(),
                    ));
                }
            }
        }
        let lines = input.stdin.iter().filter(|b| **b == b'\n').count();
        let out = if lines_only {
            format!("{lines}\n")
        } else {
            let bytes = input.stdin.len();
            let text = std::str::from_utf8(&input.stdin).unwrap_or("");
            let words = text.split_whitespace().count();
            format!("{lines} {words} {bytes}\n")
        };
        Ok(CommandOutput::ok(out.into_bytes()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn wc_l_counts_newlines() {
        let out = WcCommand
            .run(CommandInput {
                args: vec!["-l".into()],
                stdin: b"a\nb\nc\n".to_vec(),
            })
            .await
            .unwrap();
        assert_eq!(out.stdout, b"3\n");
        assert_eq!(out.exit_code, 0);
    }

    #[tokio::test]
    async fn wc_default_reports_lines_words_bytes() {
        let out = WcCommand
            .run(CommandInput {
                args: Vec::new(),
                stdin: b"hello world\nagain\n".to_vec(),
            })
            .await
            .unwrap();
        // 2 lines, 3 words, 18 bytes
        assert_eq!(out.stdout, b"2 3 18\n");
    }

    #[tokio::test]
    async fn wc_unknown_flag_errors() {
        let out = WcCommand
            .run(CommandInput {
                args: vec!["-w".into()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 2);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("[error] wc: flag '-w' not supported in v1"),
            "{stderr}"
        );
        assert!(stderr.contains("Use: wc or wc -l"), "{stderr}");
    }
}
