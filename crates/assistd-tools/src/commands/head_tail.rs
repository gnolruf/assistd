//! `head` and `tail`: take lines from one end of the named files or stdin.

use async_trait::async_trait;

use crate::command::{Command, CommandInput, CommandOutput};
use crate::commands::collect_input;

/// Lines emitted when no count flag is given, matching coreutils.
const DEFAULT_LINES: usize = 10;

/// `head [-n N] [FILE]...`: emit the first `N` lines of the named
/// files, or of stdin when none are given.
pub struct HeadCommand;

#[async_trait]
impl Command for HeadCommand {
    fn name(&self) -> &str {
        "head"
    }

    fn summary(&self) -> &'static str {
        "emit the first N lines of FILE or stdin (default 10; -n N to change)"
    }

    fn help(&self) -> String {
        format!(
            "usage: head [-n N] [FILE]...\n\
             \n\
             Emit the first N lines of the named files, or of stdin when \
             none are given, {DEFAULT_LINES} by default. `-n N`, `-nN` \
             and `-N` are all accepted.\n\
             \n\
             Several files concatenate, exactly as `cat FILE... | head` \
             would; there are no `==> FILE <==` banners. Binary files are \
             refused — use `cat -b FILE` for those.\n"
        )
    }

    async fn run(&self, input: CommandInput) -> CommandOutput {
        let (count, files) = match parse_flags("head", &input.args) {
            Ok(v) => v,
            Err(e) => return count_error("head", e),
        };
        let data = match collect_input("head", &files, input.stdin).await {
            Ok(Some(bytes)) => bytes,
            Ok(None) => return CommandOutput::usage(self.help()),
            Err(failure) => return failure,
        };
        CommandOutput::ok(first_lines(&data, count))
    }
}

/// `tail [-n N] [FILE]...`: emit the last `N` lines of the named
/// files, or of stdin when none are given.
pub struct TailCommand;

#[async_trait]
impl Command for TailCommand {
    fn name(&self) -> &str {
        "tail"
    }

    fn summary(&self) -> &'static str {
        "emit the last N lines of FILE or stdin (default 10; -n N to change)"
    }

    fn help(&self) -> String {
        format!(
            "usage: tail [-n N] [FILE]...\n\
             \n\
             Emit the last N lines of the named files, or of stdin when \
             none are given, {DEFAULT_LINES} by default. `-n N`, `-nN` \
             and `-N` are all accepted.\n\
             \n\
             Several files concatenate, exactly as `cat FILE... | tail` \
             would; there are no `==> FILE <==` banners. Binary files are \
             refused — use `cat -b FILE` for those.\n"
        )
    }

    async fn run(&self, input: CommandInput) -> CommandOutput {
        let (count, files) = match parse_flags("tail", &input.args) {
            Ok(v) => v,
            Err(e) => return count_error("tail", e),
        };
        let data = match collect_input("tail", &files, input.stdin).await {
            Ok(Some(bytes)) => bytes,
            Ok(None) => return CommandOutput::usage(self.help()),
            Err(failure) => return failure,
        };
        CommandOutput::ok(last_lines(&data, count))
    }
}

/// Why an argument list could not be read as a line count.
struct CountError {
    what: String,
    recovery: String,
}

/// Split argv into the line count and the files to read. A bare `-` is a
/// file (stdin, as in coreutils), not a flag.
fn parse_flags(cmd: &str, argv: &[String]) -> Result<(usize, Vec<String>), CountError> {
    let mut count = DEFAULT_LINES;
    let mut files = Vec::new();
    let mut pos = 0;
    while pos < argv.len() {
        let arg = &argv[pos];
        let Some(rest) = arg.strip_prefix('-').filter(|r| !r.is_empty()) else {
            files.push(arg.clone());
            pos += 1;
            continue;
        };
        let digits = match rest.strip_prefix('n') {
            Some("") => {
                pos += 1;
                argv.get(pos)
                    .map(String::as_str)
                    .ok_or_else(|| CountError {
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
        pos += 1;
    }
    Ok((count, files))
}

fn count_error(cmd: &str, e: CountError) -> CommandOutput {
    CommandOutput::usage_error(cmd, e.what, e.recovery)
}

fn first_lines(text: &[u8], count: usize) -> Vec<u8> {
    let end: usize = text
        .split_inclusive(|b| *b == b'\n')
        .take(count)
        .map(<[u8]>::len)
        .sum();
    text[..end].to_vec()
}

fn last_lines(text: &[u8], count: usize) -> Vec<u8> {
    let lines: Vec<&[u8]> = text.split_inclusive(|b| *b == b'\n').collect();
    lines[lines.len().saturating_sub(count)..].concat()
}

#[cfg(test)]
mod tests {
    use std::ops::RangeInclusive;

    use super::*;

    async fn run(cmd: &dyn Command, args: &[&str], stdin: &[u8]) -> CommandOutput {
        cmd.run(CommandInput {
            args: args.iter().map(|s| s.to_string()).collect(),
            stdin: Some(stdin.to_vec()),
        })
        .await
    }

    const FIVE: &[u8] = b"one\ntwo\nthree\nfour\nfive\n";

    #[tokio::test]
    async fn head_defaults_to_ten_lines() {
        let lines = |range: RangeInclusive<u32>| -> String {
            range.map(|i| format!("line{i}\n")).collect()
        };
        let out = run(&HeadCommand, &[], lines(1..=12).as_bytes()).await;
        assert_eq!(out.exit_code, 0);
        assert_eq!(String::from_utf8_lossy(&out.stdout), lines(1..=10));
    }

    #[tokio::test]
    async fn head_accepts_every_count_spelling() {
        for args in [vec!["-n", "2"], vec!["-n2"], vec!["-2"]] {
            let out = run(&HeadCommand, &args, FIVE).await;
            assert_eq!(out.stdout, b"one\ntwo\n", "args={args:?}");
        }
    }

    #[tokio::test]
    async fn takes_lines_from_the_right_end() {
        const FIVE_TEXT: &str = "one\ntwo\nthree\nfour\nfive\n";
        let cases: [(&dyn Command, &[&str], &str, &str); 6] = [
            (&TailCommand, &["-2"], FIVE_TEXT, "four\nfive\n"),
            (&HeadCommand, &["-99"], FIVE_TEXT, FIVE_TEXT),
            (&TailCommand, &["-99"], FIVE_TEXT, FIVE_TEXT),
            (&TailCommand, &["-1"], "a\nb", "b"),
            (&HeadCommand, &[], "", ""),
            (&TailCommand, &[], "", ""),
        ];
        for (cmd, args, stdin, expected) in cases {
            let label = format!("{} {args:?} {stdin:?}", cmd.name());
            let out = run(cmd, args, stdin.as_bytes()).await;
            assert_eq!(out.exit_code, 0, "{label}");
            assert_eq!(String::from_utf8_lossy(&out.stdout), expected, "{label}");
        }
    }

    #[tokio::test]
    async fn no_stdin_emits_usage() {
        for (cmd, name) in [
            (&HeadCommand as &dyn Command, "head"),
            (&TailCommand, "tail"),
        ] {
            let out = cmd
                .run(CommandInput {
                    args: vec!["-n".into(), "2".into()],
                    stdin: None,
                })
                .await;
            assert_eq!(out.exit_code, 2, "{name}");
            let usage = format!("usage: {name}");
            assert!(out.stdout.starts_with(usage.as_bytes()), "{name}: {out:?}");
        }
    }

    #[tokio::test]
    async fn named_files_are_read_beat_stdin_and_concatenate() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.txt");
        let b = dir.path().join("b.txt");
        std::fs::write(&a, b"one\ntwo\n").unwrap();
        std::fs::write(&b, b"three\nfour\n").unwrap();
        let (a, b) = (a.to_str().unwrap(), b.to_str().unwrap());

        let out = run(&HeadCommand, &[a], b"ignored\n").await;
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, b"one\ntwo\n");

        let both = run(&HeadCommand, &["-2", a, b], b"").await;
        assert_eq!(both.stdout, b"one\ntwo\n");
        let tail = run(&TailCommand, &["-2", a, b], b"").await;
        assert_eq!(tail.stdout, b"three\nfour\n");
    }

    #[tokio::test]
    async fn missing_file_reports_navigation_error() {
        let out = run(&HeadCommand, &["/nope/missing.txt"], b"").await;
        assert_eq!(out.exit_code, 1);
        assert_eq!(
            String::from_utf8_lossy(&out.stderr),
            "[error] head: file not found: /nope/missing.txt. Use: ls /nope to see what is there\n"
        );
    }

    #[tokio::test]
    async fn binary_file_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("blob.bin");
        std::fs::write(&bin, b"\x00\x01\x02binary\x00").unwrap();
        let out = run(&TailCommand, &[bin.to_str().unwrap()], b"").await;
        assert_eq!(out.exit_code, 1);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("[error] tail: binary"), "{stderr}");
        assert!(stderr.contains("Use: cat -b "), "{stderr}");
    }

    #[tokio::test]
    async fn non_numeric_count_errors() {
        let out = run(&TailCommand, &["-n", "lots"], FIVE).await;
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
        let out = run(&HeadCommand, &["-n"], FIVE).await;
        assert_eq!(out.exit_code, 2);
        assert_eq!(
            String::from_utf8_lossy(&out.stderr),
            "[error] head: -n needs a line count. Use: head -n 20\n"
        );
    }
}
