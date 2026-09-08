use anyhow::Result;
use async_trait::async_trait;

use crate::command::{Command, CommandInput, CommandOutput, error_line};

/// `wc [-lwc]`: count what stdin holds. With no flags the output is
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
        "count lines/words/bytes on stdin (-l, -w, -c to pick one)"
    }

    fn help(&self) -> String {
        "usage: wc [-lwc]\n\
         \n\
         Count newlines, whitespace-separated words, and bytes read from \
         stdin. With no flags the output is `<lines> <words> <bytes>` on \
         one line; with flags, only the counts you ask for, always in \
         that order.\n\
         \n\
         Flags:\n  \
           -l  print the line count\n  \
           -w  print the word count\n  \
           -c  print the byte count\n"
            .to_string()
    }

    async fn run(&self, input: CommandInput) -> Result<CommandOutput> {
        let mut selected = Selected::default();
        for arg in &input.args {
            let letters = arg.strip_prefix('-').filter(|l| !l.is_empty());
            let Some(letters) = letters else {
                return Ok(unsupported(arg));
            };
            for c in letters.chars() {
                match c {
                    'l' => selected.lines = true,
                    'w' => selected.words = true,
                    'c' => selected.bytes = true,
                    other => return Ok(unsupported(&format!("-{other}"))),
                }
            }
        }

        let Some(stdin) = input.stdin else {
            return Ok(CommandOutput::usage(self.help()));
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
        Ok(CommandOutput::ok(
            format!("{}\n", out.join(" ")).into_bytes(),
        ))
    }
}

fn unsupported(flag: &str) -> CommandOutput {
    CommandOutput::failed(
        2,
        error_line(
            "wc",
            format_args!("flag '{flag}' not supported"),
            "Use",
            "wc, wc -l, wc -w or wc -c",
        )
        .into_bytes(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn wc_l_counts_newlines() {
        let out = WcCommand
            .run(CommandInput {
                args: vec!["-l".into()],
                stdin: Some(b"a\nb\nc\n".to_vec()),
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
                stdin: Some(b"hello world\nagain\n".to_vec()),
            })
            .await
            .unwrap();
        // 2 lines, 3 words, 18 bytes
        assert_eq!(out.stdout, b"2 3 18\n");
    }

    async fn run_wc(args: &[&str], stdin: &[u8]) -> CommandOutput {
        WcCommand
            .run(CommandInput {
                args: args.iter().map(|s| s.to_string()).collect(),
                stdin: Some(stdin.to_vec()),
            })
            .await
            .expect("run returns Ok")
    }

    #[tokio::test]
    async fn wc_w_counts_words() {
        assert_eq!(run_wc(&["-w"], b"a b\nc\n").await.stdout, b"3\n");
    }

    #[tokio::test]
    async fn wc_c_counts_bytes() {
        assert_eq!(run_wc(&["-c"], b"abc\n").await.stdout, b"4\n");
    }

    #[tokio::test]
    async fn wc_no_stdin_emits_usage() {
        let out = WcCommand
            .run(CommandInput {
                args: Vec::new(),
                stdin: None,
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 2);
        assert!(out.stdout.starts_with(b"usage: wc"), "{out:?}");
    }

    #[tokio::test]
    async fn wc_counts_words_in_non_utf8_input() {
        assert_eq!(run_wc(&["-w"], b"a \xff b\n").await.stdout, b"3\n");
    }

    #[tokio::test]
    async fn wc_combined_flags_keep_canonical_order() {
        // Asked for words then lines; printed lines then words.
        assert_eq!(run_wc(&["-wl"], b"a b\nc\n").await.stdout, b"2 3\n");
    }

    #[tokio::test]
    async fn wc_unknown_flag_errors() {
        let out = run_wc(&["-q"], b"").await;
        assert_eq!(out.exit_code, 2);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("[error] wc: flag '-q' not supported"),
            "{stderr}"
        );
        assert!(
            stderr.contains("Use: wc, wc -l, wc -w or wc -c"),
            "{stderr}"
        );
    }

    #[tokio::test]
    async fn wc_positional_argument_errors() {
        let out = run_wc(&["notes.md"], b"").await;
        assert_eq!(out.exit_code, 2);
        assert!(
            String::from_utf8_lossy(&out.stderr).contains("'notes.md' not supported"),
            "{out:?}"
        );
    }
}
