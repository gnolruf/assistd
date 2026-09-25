use async_trait::async_trait;

use crate::command::{Command, CommandInput, CommandOutput};
use crate::commands::collect_input;

/// `sort [-fnr] [FILE]...`: sort the lines of the named files or stdin
/// byte-wise (`LC_ALL=C`). Every emitted line is newline-terminated.
pub struct SortCommand;

#[derive(Default)]
struct Flags {
    numeric: bool,
    reverse: bool,
    fold_case: bool,
}

#[async_trait]
impl Command for SortCommand {
    fn name(&self) -> &str {
        "sort"
    }

    fn summary(&self) -> &'static str {
        "sort lines of FILE or stdin (-n numeric, -r reverse, -f fold case)"
    }

    fn help(&self) -> String {
        "usage: sort [-fnr] [FILE]...\n\
         \n\
         Sort the lines of the named files, or of stdin when none are \
         given, and write them back out one per line. Ordering is \
         byte-wise (`LC_ALL=C sort`).\n\
         \n\
         Flags:\n  \
           -n  compare by leading integer instead of bytes\n  \
           -r  reverse the result\n  \
           -f  fold case, so `Beta` and `beta` sort together\n\
         \n\
         Several files sort together as one stream, exactly as \
         `cat FILE... | sort` would. Binary files are refused — use \
         `cat -b FILE` for those.\n"
            .to_string()
    }

    async fn run(&self, input: CommandInput) -> CommandOutput {
        let (flags, files) = match parse_flags(&input.args) {
            Ok(v) => v,
            Err(msg) => {
                return CommandOutput::usage_error(
                    "sort",
                    msg,
                    "sort (no args) for supported flags",
                );
            }
        };

        let text = match collect_input("sort", &files, input.stdin).await {
            Ok(Some(bytes)) => bytes,
            Ok(None) => return CommandOutput::usage(self.help()),
            Err(failure) => return failure,
        };
        let mut lines: Vec<&[u8]> = text.split(|b| *b == b'\n').collect();
        if lines.last().is_some_and(|l| l.is_empty()) {
            lines.pop();
        }
        if flags.numeric {
            lines.sort_by_cached_key(|l| numeric_key(l));
        } else if flags.fold_case {
            lines.sort_by_cached_key(|l| l.to_ascii_lowercase());
        } else {
            lines.sort_unstable();
        }
        if flags.reverse {
            lines.reverse();
        }

        let mut out = Vec::with_capacity(text.len());
        for line in lines {
            out.extend_from_slice(line);
            out.push(b'\n');
        }
        CommandOutput::ok(out)
    }
}

fn parse_flags(argv: &[String]) -> Result<(Flags, Vec<String>), String> {
    let mut flags = Flags::default();
    let mut files = Vec::new();
    for arg in argv {
        let Some(rest) = arg.strip_prefix('-').filter(|r| !r.is_empty()) else {
            files.push(arg.clone());
            continue;
        };
        for ch in rest.chars() {
            match ch {
                'n' => flags.numeric = true,
                'r' => flags.reverse = true,
                'f' => flags.fold_case = true,
                other => return Err(format!("unknown flag '-{other}'")),
            }
        }
    }
    Ok((flags, files))
}

fn numeric_key(line: &[u8]) -> i64 {
    let trimmed = line.trim_ascii_start();
    let len = trimmed
        .iter()
        .enumerate()
        .take_while(|(i, b)| b.is_ascii_digit() || (*i == 0 && **b == b'-'))
        .count();
    std::str::from_utf8(&trimmed[..len])
        .ok()
        .and_then(|digits| digits.parse().ok())
        .unwrap_or(i64::MIN)
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn run_sort(args: &[&str], stdin: &[u8]) -> CommandOutput {
        SortCommand
            .run(CommandInput {
                args: args.iter().map(|s| s.to_string()).collect(),
                stdin: Some(stdin.to_vec()),
            })
            .await
    }

    #[tokio::test]
    async fn orders_lines_per_flags() {
        let cases: [(&[&str], &str, &str); 8] = [
            (&[], "pear\napple\nfig\n", "apple\nfig\npear\n"),
            (&["-r"], "apple\npear\nfig\n", "pear\nfig\napple\n"),
            (&["-n"], "9\n10\n2\n", "2\n9\n10\n"),
            (&["-nr"], "3\n10\n7\n", "10\n7\n3\n"),
            (&["-nr"], "2\tbeta\n11\talpha\n", "11\talpha\n2\tbeta\n"),
            (&["-f"], "beta\nAlpha\ngamma\n", "Alpha\nbeta\ngamma\n"),
            (&[], "b\na", "a\nb\n"),
            (&[], "", ""),
        ];
        for (args, stdin, expected) in cases {
            let label = format!("{args:?} {stdin:?}");
            let out = run_sort(args, stdin.as_bytes()).await;
            assert_eq!(out.exit_code, 0, "{label}");
            assert_eq!(String::from_utf8_lossy(&out.stdout), expected, "{label}");
        }
    }

    #[tokio::test]
    async fn no_stdin_emits_usage() {
        let out = SortCommand
            .run(CommandInput {
                args: Vec::new(),
                stdin: None,
            })
            .await;
        assert_eq!(out.exit_code, 2);
        assert!(out.stdout.starts_with(b"usage: sort"), "{out:?}");
    }

    #[tokio::test]
    async fn unknown_flag_errors() {
        let out = run_sort(&["-q"], b"a\n").await;
        assert_eq!(out.exit_code, 2);
        assert_eq!(
            String::from_utf8_lossy(&out.stderr),
            "[error] sort: unknown flag '-q'. Use: sort (no args) for supported flags\n"
        );
    }

    #[tokio::test]
    async fn named_files_sort_as_one_stream() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.txt");
        let b = dir.path().join("b.txt");
        std::fs::write(&a, b"pear\nfig\n").unwrap();
        std::fs::write(&b, b"apple\n").unwrap();
        let out = run_sort(&[a.to_str().unwrap(), b.to_str().unwrap()], b"ignored\n").await;
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, b"apple\nfig\npear\n");
    }

    #[tokio::test]
    async fn missing_file_reports_navigation_error() {
        let out = run_sort(&["/nope/missing.txt"], b"").await;
        assert_eq!(out.exit_code, 1);
        assert_eq!(
            String::from_utf8_lossy(&out.stderr),
            "[error] sort: file not found: /nope/missing.txt. Use: ls /nope to see what is there\n"
        );
    }
}
