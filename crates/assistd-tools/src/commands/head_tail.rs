//! `head` and `tail`: take lines from one end of stdin. They share a
//! flag parser because the only thing that differs between them is which
//! end of the stream they keep.

use anyhow::Result;
use async_trait::async_trait;

use crate::command::{Command, CommandInput, CommandOutput, error_line};

/// Lines emitted when no count flag is given, matching coreutils.
const DEFAULT_LINES: usize = 10;

/// `head [-n N]`: emit the first `N` lines read from stdin.
pub struct HeadCommand;

/// `tail [-n N]`: emit the last `N` lines read from stdin.
pub struct TailCommand;

/// Why an argument list could not be read as a line count.
struct CountError {
    what: String,
    recovery: String,
}

/// Parse the line count out of argv. Accepts `-n N`, `-nN` and the bare
/// `-N` shorthand. A non-flag argument is rejected rather than ignored:
/// silently reading empty stdin because a filename was mistaken for a
/// flag is the worst failure this command has.
fn parse_line_count(cmd: &str, argv: &[String]) -> Result<usize, CountError> {
    let mut count = DEFAULT_LINES;
    let mut i = 0;
    while i < argv.len() {
        let arg = &argv[i];
        let Some(rest) = arg.strip_prefix('-') else {
            return Err(CountError {
                what: format!("unexpected argument: {arg}"),
                recovery: format!("cat {arg} | {cmd}"),
            });
        };
        let digits = match rest.strip_prefix('n') {
            Some("") => {
                i += 1;
                argv.get(i).map(String::as_str).ok_or_else(|| CountError {
                    what: "-n needs a line count".to_string(),
                    recovery: format!("{cmd} -n 20"),
                })?
            }
            Some(glued) => glued,
            None => rest,
        };
        count = digits.parse().map_err(|_| CountError {
            what: format!("not a line count: '-{digits}'"),
            recovery: format!("{cmd} -n 20"),
        })?;
        i += 1;
    }
    Ok(count)
}

fn count_error(cmd: &str, e: CountError) -> CommandOutput {
    CommandOutput::failed(2, error_line(cmd, e.what, "Use", e.recovery).into_bytes())
}

fn first_lines(stdin: &[u8], count: usize) -> Vec<u8> {
    stdin
        .split_inclusive(|b| *b == b'\n')
        .take(count)
        .collect::<Vec<_>>()
        .concat()
}

fn last_lines(stdin: &[u8], count: usize) -> Vec<u8> {
    let lines: Vec<&[u8]> = stdin.split_inclusive(|b| *b == b'\n').collect();
    lines[lines.len().saturating_sub(count)..].concat()
}

#[async_trait]
impl Command for HeadCommand {
    fn name(&self) -> &str {
        "head"
    }

    fn summary(&self) -> &'static str {
        "emit the first N lines of stdin (default 10; -n N to change)"
    }

    fn help(&self) -> String {
        format!(
            "usage: head [-n N]\n\
             \n\
             Emit the first N lines read from stdin, {DEFAULT_LINES} by \
             default. `-n N`, `-nN` and `-N` are all accepted.\n\
             \n\
             Reads stdin only; pipe a file in with `cat FILE | head`.\n"
        )
    }

    async fn run(&self, input: CommandInput) -> Result<CommandOutput> {
        match parse_line_count("head", &input.args) {
            Ok(n) => Ok(CommandOutput::ok(first_lines(&input.stdin, n))),
            Err(e) => Ok(count_error("head", e)),
        }
    }
}

#[async_trait]
impl Command for TailCommand {
    fn name(&self) -> &str {
        "tail"
    }

    fn summary(&self) -> &'static str {
        "emit the last N lines of stdin (default 10; -n N to change)"
    }

    fn help(&self) -> String {
        format!(
            "usage: tail [-n N]\n\
             \n\
             Emit the last N lines read from stdin, {DEFAULT_LINES} by \
             default. `-n N`, `-nN` and `-N` are all accepted.\n\
             \n\
             Reads stdin only; pipe a file in with `cat FILE | tail`.\n"
        )
    }

    async fn run(&self, input: CommandInput) -> Result<CommandOutput> {
        match parse_line_count("tail", &input.args) {
            Ok(n) => Ok(CommandOutput::ok(last_lines(&input.stdin, n))),
            Err(e) => Ok(count_error("tail", e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn run(cmd: impl Command, args: &[&str], stdin: &[u8]) -> CommandOutput {
        cmd.run(CommandInput {
            args: args.iter().map(|s| s.to_string()).collect(),
            stdin: stdin.to_vec(),
        })
        .await
        .expect("run returns Ok")
    }

    const FIVE: &[u8] = b"one\ntwo\nthree\nfour\nfive\n";

    #[tokio::test]
    async fn head_defaults_to_ten_lines() {
        let stdin: Vec<u8> = (1..=12)
            .map(|i| format!("line{i}\n"))
            .collect::<String>()
            .into();
        let out = run(HeadCommand, &[], &stdin).await;
        assert_eq!(out.exit_code, 0);
        assert_eq!(String::from_utf8_lossy(&out.stdout).lines().count(), 10);
    }

    #[tokio::test]
    async fn head_accepts_every_count_spelling() {
        for args in [vec!["-n", "2"], vec!["-n2"], vec!["-2"]] {
            let out = run(HeadCommand, &args, FIVE).await;
            assert_eq!(out.stdout, b"one\ntwo\n", "args={args:?}");
        }
    }

    #[tokio::test]
    async fn tail_takes_from_the_end() {
        let out = run(TailCommand, &["-2"], FIVE).await;
        assert_eq!(out.stdout, b"four\nfive\n");
    }

    #[tokio::test]
    async fn count_larger_than_input_yields_everything() {
        assert_eq!(run(HeadCommand, &["-99"], FIVE).await.stdout, FIVE);
        assert_eq!(run(TailCommand, &["-99"], FIVE).await.stdout, FIVE);
    }

    #[tokio::test]
    async fn unterminated_last_line_is_preserved() {
        let out = run(TailCommand, &["-1"], b"a\nb").await;
        assert_eq!(out.stdout, b"b");
    }

    #[tokio::test]
    async fn empty_stdin_is_empty_output() {
        assert!(run(HeadCommand, &[], b"").await.stdout.is_empty());
        assert!(run(TailCommand, &[], b"").await.stdout.is_empty());
    }

    #[tokio::test]
    async fn filename_argument_points_at_the_pipe() {
        let out = run(HeadCommand, &["notes.md"], FIVE).await;
        assert_eq!(out.exit_code, 2);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("[error] head: unexpected argument: notes.md"),
            "{stderr}"
        );
        assert!(stderr.contains("Use: cat notes.md | head"), "{stderr}");
    }

    #[tokio::test]
    async fn non_numeric_count_errors() {
        let out = run(TailCommand, &["-n", "lots"], FIVE).await;
        assert_eq!(out.exit_code, 2);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("[error] tail: not a line count"),
            "{stderr}"
        );
        assert!(stderr.contains("Use: tail -n 20"), "{stderr}");
    }

    #[tokio::test]
    async fn dangling_n_errors() {
        let out = run(HeadCommand, &["-n"], FIVE).await;
        assert_eq!(out.exit_code, 2);
        assert!(
            String::from_utf8_lossy(&out.stderr).contains("-n needs a line count"),
            "{out:?}"
        );
    }
}
