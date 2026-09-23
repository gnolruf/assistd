use async_trait::async_trait;

use crate::command::{Command, CommandInput, CommandOutput};
use crate::commands::collect_input;

/// `wc [-lwc] [FILE]...`: count what the named files, or stdin, hold.
/// With no flags the output is
/// `<lines> <words> <bytes>`; each flag narrows it to that one count.
/// Flags can be combined, and the selected counts print in the
/// canonical lines-words-bytes order regardless of how they were given.
pub struct WcCommand;

/// Which counts to print. Nothing selected means all three, matching
/// bare `wc`.
#[derive(Default)]
struct Selected {
    lines: bool,
    words: bool,
    bytes: bool,
}

impl Selected {
    fn any(&self) -> bool {
        self.lines || self.words || self.bytes
    }
}

#[async_trait]
impl Command for WcCommand {
    fn name(&self) -> &str {
        "wc"
    }

    fn summary(&self) -> &'static str {
        "count lines/words/bytes of FILE or stdin (-l, -w, -c to pick one)"
    }

    fn help(&self) -> String {
        "usage: wc [-lwc] [FILE]...\n\
         \n\
         Count newlines, whitespace-separated words, and bytes in the \
         named files, or in stdin when none are given. With no flags the \
         output is `<lines> <words> <bytes>` on one line; with flags, \
         only the counts you ask for, always in that order.\n\
         \n\
         Flags:\n  \
           -l  print the line count\n  \
           -w  print the word count\n  \
           -c  print the byte count\n\
         \n\
         Several files count as one stream, exactly as \
         `cat FILE... | wc` would; there is no per-file breakdown. \
         Binary files are refused — use `cat -b FILE` for those.\n"
            .to_string()
    }

    async fn run(&self, input: CommandInput) -> CommandOutput {
        let mut selected = Selected::default();
        let mut files = Vec::new();
        for arg in &input.args {
            let letters = arg.strip_prefix('-').filter(|l| !l.is_empty());
            let Some(letters) = letters else {
                files.push(arg.clone());
                continue;
            };
            for c in letters.chars() {
                match c {
                    'l' => selected.lines = true,
                    'w' => selected.words = true,
                    'c' => selected.bytes = true,
                    other => return unsupported(&format!("-{other}")),
                }
            }
        }

        let stdin = match collect_input("wc", &files, input.stdin).await {
            Ok(Some(bytes)) => bytes,
            Ok(None) => return CommandOutput::usage(self.help()),
            Err(failure) => return failure,
        };
        let lines = stdin.iter().filter(|b| **b == b'\n').count();
        let words = stdin
            .split(u8::is_ascii_whitespace)
            .filter(|w| !w.is_empty())
            .count();
        let bytes = stdin.len();

        let show_all = !selected.any();
        let counts = [
            (selected.lines || show_all, lines),
            (selected.words || show_all, words),
            (selected.bytes || show_all, bytes),
        ];
        let out: Vec<String> = counts
            .iter()
            .filter(|(wanted, _)| *wanted)
            .map(|(_, n)| n.to_string())
            .collect();
        CommandOutput::ok(format!("{}\n", out.join(" ")).into_bytes())
    }
}

fn unsupported(flag: &str) -> CommandOutput {
    CommandOutput::usage_error(
        "wc",
        format_args!("flag '{flag}' not supported"),
        "wc, wc -l, wc -w or wc -c",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn run_wc(args: &[&str], stdin: Option<&[u8]>) -> CommandOutput {
        WcCommand
            .run(CommandInput {
                args: args.iter().map(|s| s.to_string()).collect(),
                stdin: stdin.map(<[u8]>::to_vec),
            })
            .await
    }

    #[tokio::test]
    async fn counts_per_flags() {
        let cases: [(&[&str], &[u8], &str); 6] = [
            (&[], b"hello world\nagain\n", "2 3 18\n"),
            (&["-l"], b"a\nb\nc\n", "3\n"),
            (&["-w"], b"a b\nc\n", "3\n"),
            (&["-c"], b"abc\n", "4\n"),
            (&["-w"], b"a \xff b\n", "3\n"),
            // Asked for words then lines; printed lines then words.
            (&["-wl"], b"a b\nc\n", "2 3\n"),
        ];
        for (args, stdin, expected) in cases {
            let out = run_wc(args, Some(stdin)).await;
            assert_eq!(out.exit_code, 0, "{args:?}");
            assert_eq!(String::from_utf8_lossy(&out.stdout), expected, "{args:?}");
        }
    }

    #[tokio::test]
    async fn wc_no_stdin_emits_usage() {
        let out = run_wc(&[], None).await;
        assert_eq!(out.exit_code, 2);
        assert!(out.stdout.starts_with(b"usage: wc"), "{out:?}");
    }

    #[tokio::test]
    async fn wc_unknown_flag_errors() {
        let out = run_wc(&["-q"], Some(b"")).await;
        assert_eq!(out.exit_code, 2);
        assert_eq!(
            String::from_utf8_lossy(&out.stderr),
            "[error] wc: flag '-q' not supported. Use: wc, wc -l, wc -w or wc -c\n"
        );
    }

    #[tokio::test]
    async fn named_files_count_as_one_stream() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.txt");
        let b = dir.path().join("b.txt");
        std::fs::write(&a, b"one two\n").unwrap();
        std::fs::write(&b, b"three\n").unwrap();
        let out = run_wc(
            &["-l", a.to_str().unwrap(), b.to_str().unwrap()],
            Some(b"ignored\n"),
        )
        .await;
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, b"2\n");
    }

    #[tokio::test]
    async fn missing_file_reports_navigation_error() {
        let out = run_wc(&["/nope/missing.txt"], Some(b"")).await;
        assert_eq!(out.exit_code, 1);
        assert_eq!(
            String::from_utf8_lossy(&out.stderr),
            "[error] wc: file not found: /nope/missing.txt. Use: ls /nope to see what is there\n"
        );
    }

    #[tokio::test]
    async fn wc_refuses_a_device_file() {
        let out = run_wc(&["/dev/null"], None).await;
        assert_eq!(out.exit_code, 1);
        assert_eq!(
            String::from_utf8_lossy(&out.stderr),
            "[error] wc: /dev/null: not a regular file (device, pipe, or socket). Check: ls -l /dev/null\n"
        );
    }
}
