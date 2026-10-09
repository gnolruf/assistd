//! `head` and `tail`: take lines from one end of the named files or stdin.

use async_trait::async_trait;

use crate::command::{Command, CommandInput, CommandOutput};
use crate::commands::collect_input;

/// Lines emitted when no count flag is given, matching coreutils.
const DEFAULT_LINES: usize = 10;

/// `head [-n N] [FILE]...`: emit the first `N` lines of the named
/// files, or of stdin when none are given.
#[derive(Debug)]
pub struct HeadCommand;

#[async_trait]
impl Command for HeadCommand {
    fn name(&self) -> &'static str {
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
#[derive(Debug)]
pub struct TailCommand;

#[async_trait]
impl Command for TailCommand {
    fn name(&self) -> &'static str {
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

    use super::*;

    async fn run(cmd: &dyn Command, args: &[&str], stdin: &[u8]) -> CommandOutput {
        cmd.run(CommandInput {
            args: args.iter().map(ToString::to_string).collect(),
            stdin: Some(stdin.to_vec()),
        })
        .await
    }

    const FIVE: &[u8] = b"one\ntwo\nthree\nfour\nfive\n";

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
        let cases: [(&dyn Command, &[&str], &str, &str); 3] = [
            (&TailCommand, &["-2"], FIVE_TEXT, "four\nfive\n"),
            (&TailCommand, &["-99"], FIVE_TEXT, FIVE_TEXT),
            (&TailCommand, &["-1"], "a\nb", "b"),
        ];
        for (cmd, args, stdin, expected) in cases {
            let label = format!("{} {args:?} {stdin:?}", cmd.name());
            let out = run(cmd, args, stdin.as_bytes()).await;
            assert_eq!(out.exit_code, 0, "{label}");
            assert_eq!(String::from_utf8_lossy(&out.stdout), expected, "{label}");
        }
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
}
