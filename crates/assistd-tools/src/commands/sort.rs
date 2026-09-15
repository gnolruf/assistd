use anyhow::Result;
use async_trait::async_trait;

use crate::command::{Command, CommandInput, CommandOutput, error_line};
use crate::commands::collect_input;

/// `sort [-fnr] [FILE]...`: sort the lines of the named files, or of
/// stdin when none are given.
///
/// Flags:
/// - `-n` compare by leading integer instead of bytes
/// - `-r` reverse the result
/// - `-f` fold case, so `Beta` and `beta` sort together
///
/// Comparison is byte-wise, matching `LC_ALL=C sort`, so the ordering
/// doesn't depend on the daemon's locale. Every emitted line is
/// newline-terminated even when the input's last line was not, because
/// reordering an unterminated line would otherwise glue it to its new
/// neighbour.
pub struct SortCommand;

#[derive(Default)]
struct Flags {
    numeric: bool,
    reverse: bool,
    fold_case: bool,
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

    async fn run(&self, input: CommandInput) -> Result<CommandOutput> {
        let (flags, files) = match parse_flags(&input.args) {
            Ok(v) => v,
            Err(msg) => {
                return Ok(CommandOutput::failed(
                    2,
                    error_line("sort", msg, "Use", "sort (no args) for supported flags")
                        .into_bytes(),
                ));
            }
        };

        let stdin = match collect_input("sort", &files, input.stdin).await {
            Ok(Some(bytes)) => bytes,
            Ok(None) => return Ok(CommandOutput::usage(self.help())),
            Err(failure) => return Ok(failure),
        };
        let mut lines: Vec<&[u8]> = stdin.split(|b| *b == b'\n').collect();
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

        let mut out = Vec::with_capacity(stdin.len());
        for line in lines {
            out.extend_from_slice(line);
            out.push(b'\n');
        }
        Ok(CommandOutput::ok(out))
    }
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
            .expect("run returns Ok")
    }

    #[tokio::test]
    async fn sorts_lexicographically() {
        let out = run_sort(&[], b"pear\napple\nfig\n").await;
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, b"apple\nfig\npear\n");
    }

    #[tokio::test]
    async fn r_flag_reverses() {
        let out = run_sort(&["-r"], b"apple\npear\nfig\n").await;
        assert_eq!(out.stdout, b"pear\nfig\napple\n");
    }

    #[tokio::test]
    async fn n_flag_orders_numerically() {
        // Byte order would put "10" before "9"; -n must not.
        let out = run_sort(&["-n"], b"9\n10\n2\n").await;
        assert_eq!(out.stdout, b"2\n9\n10\n");
    }

    #[tokio::test]
    async fn nr_combined_is_descending_numeric() {
        let out = run_sort(&["-nr"], b"3\n10\n7\n").await;
        assert_eq!(out.stdout, b"10\n7\n3\n");
    }

    #[tokio::test]
    async fn numeric_key_reads_leading_count_of_a_uniq_c_line() {
        let out = run_sort(&["-nr"], b"2\tbeta\n11\talpha\n").await;
        assert_eq!(out.stdout, b"11\talpha\n2\tbeta\n");
    }

    #[tokio::test]
    async fn f_flag_folds_case() {
        // Byte order puts every capital before every lowercase letter.
        let out = run_sort(&["-f"], b"beta\nAlpha\ngamma\n").await;
        assert_eq!(out.stdout, b"Alpha\nbeta\ngamma\n");
    }

    #[tokio::test]
    async fn unterminated_last_line_gets_a_newline() {
        let out = run_sort(&[], b"b\na").await;
        assert_eq!(out.stdout, b"a\nb\n");
    }

    #[tokio::test]
    async fn empty_stdin_is_empty_output() {
        assert!(run_sort(&[], b"").await.stdout.is_empty());
    }

    #[tokio::test]
    async fn no_stdin_emits_usage() {
        let out = SortCommand
            .run(CommandInput {
                args: Vec::new(),
                stdin: None,
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 2);
        assert!(out.stdout.starts_with(b"usage: sort"), "{out:?}");
    }

    #[tokio::test]
    async fn unknown_flag_errors() {
        let out = run_sort(&["-q"], b"a\n").await;
        assert_eq!(out.exit_code, 2);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("[error] sort: unknown flag '-q'"),
            "{stderr}"
        );
        assert!(stderr.contains("Use: "), "{stderr}");
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
        assert!(
            String::from_utf8_lossy(&out.stderr).contains("[error] sort: file not found"),
            "{out:?}"
        );
    }
}
