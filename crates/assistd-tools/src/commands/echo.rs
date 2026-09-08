use anyhow::Result;
use async_trait::async_trait;

use crate::command::{Command, CommandInput, CommandOutput};

/// `echo [-ne] [ARGS...]`: write args joined by spaces, followed by a
/// newline unless `-n` is given.
///
/// Flags:
/// - `-n` omit the trailing newline
/// - `-e` interpret `\n`, `\t`, `\r`, `\0` and `\\` in the arguments
pub struct EchoCommand;

#[derive(Default)]
struct Flags {
    no_newline: bool,
    escapes: bool,
}

fn split_flags(argv: &[String]) -> (Flags, &[String]) {
    let mut flags = Flags::default();
    let mut i = 0;
    while let Some(arg) = argv.get(i) {
        let Some(letters) = arg.strip_prefix('-').filter(|l| !l.is_empty()) else {
            break;
        };
        if !letters.chars().all(|c| matches!(c, 'n' | 'e' | 'E')) {
            break;
        }
        for c in letters.chars() {
            match c {
                'n' => flags.no_newline = true,
                'e' => flags.escapes = true,
                _ => flags.escapes = false,
            }
        }
        i += 1;
    }
    (flags, &argv[i..])
}

fn unescape(text: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(text.len());
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            let mut buf = [0u8; 4];
            out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            continue;
        }
        match chars.next() {
            Some('n') => out.push(b'\n'),
            Some('t') => out.push(b'\t'),
            Some('r') => out.push(b'\r'),
            Some('0') => out.push(0),
            Some('\\') => out.push(b'\\'),
            Some(other) => {
                let mut buf = [0u8; 4];
                out.push(b'\\');
                out.extend_from_slice(other.encode_utf8(&mut buf).as_bytes());
            }
            None => out.push(b'\\'),
        }
    }
    out
}

#[async_trait]
impl Command for EchoCommand {
    fn name(&self) -> &str {
        "echo"
    }

    fn summary(&self) -> &'static str {
        "write arguments joined by spaces (-n no newline, -e escapes)"
    }

    fn help(&self) -> String {
        "usage: echo [-ne] [ARGS...]\n\
         \n\
         Write the argument list joined by single spaces, followed by a \
         newline.\n\
         \n\
         Flags:\n  \
           -n  omit the trailing newline\n  \
           -e  interpret \\n, \\t, \\r, \\0 and \\\\ in the arguments\n  \
           -E  disable -e (the default)\n\
         \n\
         Note that a backslash inside double quotes reaches echo intact, \
         so `echo -e \"a\\nb\"` prints two lines.\n"
            .to_string()
    }

    async fn run(&self, input: CommandInput) -> Result<CommandOutput> {
        if input.args.is_empty() {
            return Ok(CommandOutput::usage(self.help()));
        }
        let (flags, words) = split_flags(&input.args);
        let joined = words.join(" ");
        let mut out = if flags.escapes {
            unescape(&joined)
        } else {
            joined.into_bytes()
        };
        if !flags.no_newline {
            out.push(b'\n');
        }
        Ok(CommandOutput::ok(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn run_echo(args: &[&str]) -> CommandOutput {
        EchoCommand
            .run(CommandInput {
                args: args.iter().map(|s| s.to_string()).collect(),
                stdin: None,
            })
            .await
            .expect("run returns Ok")
    }

    #[tokio::test]
    async fn echo_joins_args_with_space_and_newline() {
        let out = run_echo(&["hello", "world"]).await;
        assert_eq!(out.stdout, b"hello world\n");
        assert_eq!(out.exit_code, 0);
    }

    #[tokio::test]
    async fn echo_no_args_emits_usage() {
        let out = run_echo(&[]).await;
        assert_eq!(out.exit_code, 2);
        assert!(out.stdout.starts_with(b"usage: echo"), "{out:?}");
    }

    #[tokio::test]
    async fn n_flag_omits_trailing_newline() {
        assert_eq!(run_echo(&["-n", "hi"]).await.stdout, b"hi");
    }

    #[tokio::test]
    async fn e_flag_expands_escapes() {
        assert_eq!(run_echo(&["-e", r"a\nb\tc"]).await.stdout, b"a\nb\tc\n");
    }

    #[tokio::test]
    async fn escapes_are_literal_without_e() {
        assert_eq!(run_echo(&[r"a\nb"]).await.stdout, b"a\\nb\n");
    }

    #[tokio::test]
    async fn combined_ne_flag() {
        assert_eq!(run_echo(&["-ne", r"a\nb"]).await.stdout, b"a\nb");
    }

    #[tokio::test]
    async fn capital_e_disables_escapes() {
        assert_eq!(run_echo(&["-e", "-E", r"a\nb"]).await.stdout, b"a\\nb\n");
    }

    #[tokio::test]
    async fn unknown_escape_keeps_both_characters() {
        assert_eq!(run_echo(&["-e", r"a\qb"]).await.stdout, b"a\\qb\n");
    }

    #[tokio::test]
    async fn non_flag_argument_ends_flag_parsing() {
        // `-n` after a word is data, not a flag, so the newline stays.
        assert_eq!(run_echo(&["hi", "-n"]).await.stdout, b"hi -n\n");
    }

    #[tokio::test]
    async fn lone_dash_is_data() {
        assert_eq!(run_echo(&["-"]).await.stdout, b"-\n");
    }
}
