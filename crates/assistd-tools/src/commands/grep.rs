use anyhow::Result;
use async_trait::async_trait;
use regex::RegexBuilder;

use crate::command::{Command, CommandInput, CommandOutput, error_line, io_error_nav};

/// `grep [-i] [-v] [-c] PATTERN [FILE]` — print lines from FILE (or
/// stdin) that match `PATTERN`.
///
/// Flags:
/// - `-i` case-insensitive
/// - `-v` invert match
/// - `-c` print the count instead of the matching lines
///
/// Flags can be combined (`-ivc`). Exit 0 if any line matched (or the
/// count is non-zero under `-c`), 1 otherwise, 2 on usage/input errors.
pub struct GrepCommand;

#[derive(Default)]
struct Flags {
    case_insensitive: bool,
    invert: bool,
    count_only: bool,
}

fn parse_flags(argv: &[String]) -> Result<(Flags, &[String]), String> {
    let mut flags = Flags::default();
    let mut i = 0;
    while i < argv.len() {
        let a = &argv[i];
        if a == "--" {
            i += 1;
            break;
        }
        if let Some(rest) = a.strip_prefix('-') {
            if rest.is_empty() {
                break; // bare `-` = stdin sentinel; treat as positional
            }
            for ch in rest.chars() {
                match ch {
                    'i' => flags.case_insensitive = true,
                    'v' => flags.invert = true,
                    'c' => flags.count_only = true,
                    other => return Err(format!("unknown flag '-{other}'")),
                }
            }
            i += 1;
        } else {
            break;
        }
    }
    Ok((flags, &argv[i..]))
}

#[async_trait]
impl Command for GrepCommand {
    fn name(&self) -> &str {
        "grep"
    }

    fn summary(&self) -> &'static str {
        "filter lines matching a pattern (supports -i, -v, -c)"
    }

    fn help(&self) -> String {
        "usage: grep [-ivc] PATTERN [FILE]\n\
         \n\
         Print lines from FILE (or stdin) that match the regex PATTERN.\n\
         \n\
         Flags:\n  \
           -i  case-insensitive\n  \
           -v  invert match (print non-matching lines)\n  \
           -c  print the count instead of the matching lines\n\
         \n\
         Flags can be combined (e.g. `-ivc`). Exit 0 if any line matched \
         (or the count is non-zero under `-c`), 1 if no matches, 2 on \
         usage/input errors.\n"
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
        let (flags, positional) = match parse_flags(&input.args) {
            Ok(v) => v,
            Err(msg) => {
                return Ok(CommandOutput::failed(
                    2,
                    error_line("grep", msg, "Use", "grep (no args) for supported flags")
                        .into_bytes(),
                ));
            }
        };
        if positional.is_empty() {
            return Ok(CommandOutput {
                stdout: self.help().into_bytes(),
                stderr: Vec::new(),
                exit_code: 2,
                attachments: Vec::new(),
            });
        }

        let pattern = &positional[0];
        let re = match RegexBuilder::new(pattern)
            .case_insensitive(flags.case_insensitive)
            .build()
        {
            Ok(r) => r,
            Err(e) => {
                return Ok(CommandOutput::failed(
                    2,
                    error_line(
                        "grep",
                        format_args!("bad regex pattern: {e}"),
                        "Check",
                        "escape regex metachars; run grep for usage",
                    )
                    .into_bytes(),
                ));
            }
        };

        let content: Vec<u8> = if positional.len() > 1 {
            let path = &positional[1];
            match tokio::fs::read(path).await {
                Ok(b) => b,
                Err(e) => {
                    return Ok(CommandOutput::failed(
                        2,
                        io_error_nav("grep", path, &e).into_bytes(),
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
                    error_line(
                        "grep",
                        "input is not valid UTF-8",
                        "Try",
                        "grep on a text file or pipe from cat",
                    )
                    .into_bytes(),
                ));
            }
        };

        let mut matched_lines = Vec::new();
        let mut count: usize = 0;
        for line in text.split_inclusive('\n') {
            let hit = re.is_match(line) ^ flags.invert;
            if hit {
                count += 1;
                if !flags.count_only {
                    matched_lines.extend_from_slice(line.as_bytes());
                }
            }
        }

        let stdout = if flags.count_only {
            format!("{count}\n").into_bytes()
        } else {
            matched_lines
        };
        Ok(CommandOutput {
            stdout,
            stderr: Vec::new(),
            exit_code: if count > 0 { 0 } else { 1 },
            attachments: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn run_grep(args: &[&str], stdin: &[u8]) -> CommandOutput {
        GrepCommand
            .run(CommandInput {
                args: args.iter().map(|s| s.to_string()).collect(),
                stdin: stdin.to_vec(),
            })
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn matches_from_stdin() {
        let out = run_grep(&["ERROR"], b"INFO ok\nERROR boom\nINFO also\n").await;
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, b"ERROR boom\n");
    }

    #[tokio::test]
    async fn no_match_exits_1() {
        let out = run_grep(&["ZZZ"], b"nothing here\n").await;
        assert_eq!(out.exit_code, 1);
        assert!(out.stdout.is_empty());
    }

    #[tokio::test]
    async fn missing_pattern_errors() {
        let out = run_grep(&[], b"").await;
        assert_eq!(out.exit_code, 2);
    }

    #[tokio::test]
    async fn i_flag_matches_case_insensitively() {
        let out = run_grep(&["-i", "error"], b"ERROR boom\nnope\n").await;
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, b"ERROR boom\n");
    }

    #[tokio::test]
    async fn v_flag_inverts_match() {
        let out = run_grep(&["-v", "ERROR"], b"INFO ok\nERROR boom\n").await;
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, b"INFO ok\n");
    }

    #[tokio::test]
    async fn c_flag_returns_count() {
        let out = run_grep(&["-c", "ERROR"], b"ERROR a\nINFO\nERROR b\n").await;
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, b"2\n");
    }

    #[tokio::test]
    async fn ic_combined_case_insensitive_count() {
        let out = run_grep(&["-ic", "error"], b"ERROR a\ninfo\nError b\nnothing\n").await;
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, b"2\n");
    }

    #[tokio::test]
    async fn ivc_all_three_flags_together() {
        let out = run_grep(&["-ivc", "error"], b"ERROR\ninfo\nError\nok\n").await;
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, b"2\n");
    }

    #[tokio::test]
    async fn c_with_no_matches_exits_1_but_prints_zero() {
        let out = run_grep(&["-c", "ZZZ"], b"a\nb\n").await;
        assert_eq!(out.exit_code, 1);
        assert_eq!(out.stdout, b"0\n");
    }

    #[tokio::test]
    async fn unknown_flag_errors() {
        let out = run_grep(&["-x", "foo"], b"").await;
        assert_eq!(out.exit_code, 2);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("[error] grep: unknown flag '-x'"),
            "{stderr}"
        );
        assert!(
            stderr.contains("Use: grep (no args) for supported flags"),
            "{stderr}"
        );
    }
}
