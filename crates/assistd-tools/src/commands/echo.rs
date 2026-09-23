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

fn parse_flags(argv: &[String]) -> (Flags, &[String]) {
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

    async fn run(&self, input: CommandInput) -> CommandOutput {
        if input.args.is_empty() {
            return CommandOutput::usage(self.help());
        }
        let (flags, words) = parse_flags(&input.args);
        let joined = words.join(" ");
        let mut out = if flags.escapes {
            unescape(&joined)
        } else {
            joined.into_bytes()
        };
        if !flags.no_newline {
            out.push(b'\n');
        }
        CommandOutput::ok(out)
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
    }

    #[tokio::test]
    async fn echo_no_args_emits_usage() {
        let out = run_echo(&[]).await;
        assert_eq!(out.exit_code, 2);
        assert!(out.stdout.starts_with(b"usage: echo"), "{out:?}");
    }

    #[tokio::test]
    async fn echo_output_per_flags() {
        let cases: [(&[&str], &[u8]); 9] = [
            (&["hello", "world"], b"hello world\n"),
            (&["-n", "hi"], b"hi"),
            (&["-e", r"a\nb\tc"], b"a\nb\tc\n"),
            (&[r"a\nb"], b"a\\nb\n"),
            (&["-ne", r"a\nb"], b"a\nb"),
            (&["-e", "-E", r"a\nb"], b"a\\nb\n"),
            (&["-e", r"a\qb"], b"a\\qb\n"),
            // A flag after a word is data, so the newline stays.
            (&["hi", "-n"], b"hi -n\n"),
            (&["-"], b"-\n"),
        ];
        for (args, expected) in cases {
            let out = run_echo(args).await;
            assert_eq!(out.exit_code, 0, "{args:?}");
            assert_eq!(
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(expected),
                "{args:?}"
            );
        }
    }
}
