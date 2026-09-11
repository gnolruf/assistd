use anyhow::Result;
use async_trait::async_trait;

use crate::command::{Command, CommandInput, CommandOutput, error_line};
use crate::commands::collect_input;

/// `uniq [-c] [FILE]...`: collapse runs of identical adjacent lines
/// from the named files, or from stdin when none are given. Only
/// *adjacent* duplicates collapse, so the usual spelling is
/// `sort | uniq`.
///
/// Flags:
/// - `-c` prefix each line with `<count>\t`
pub struct UniqCommand;

#[async_trait]
impl Command for UniqCommand {
    fn name(&self) -> &str {
        "uniq"
    }

    fn summary(&self) -> &'static str {
        "collapse adjacent duplicate lines of FILE or stdin (-c to count)"
    }

    fn help(&self) -> String {
        "usage: uniq [-c] [FILE]...\n\
         \n\
         Collapse runs of identical adjacent lines from the named files, \
         or from stdin when none are given. Only adjacent duplicates \
         collapse, so sort first: `sort FILE | uniq`.\n\
         \n\
         Flags:\n  \
           -c  prefix each line with its repeat count and a tab\n\
         \n\
         The `-c` output is `<count>\\t<line>`, which `sort -nr` orders \
         by frequency. Binary files are refused — use `cat -b FILE` for \
         those.\n"
            .to_string()
    }

    async fn run(&self, input: CommandInput) -> Result<CommandOutput> {
        let mut count_runs = false;
        let mut files = Vec::new();
        for arg in &input.args {
            match arg.as_str() {
                "-c" => count_runs = true,
                flag if flag.starts_with('-') && flag.len() > 1 => {
                    return Ok(CommandOutput::failed(
                        2,
                        error_line(
                            "uniq",
                            format_args!("unknown flag '{flag}'"),
                            "Use",
                            "uniq or uniq -c",
                        )
                        .into_bytes(),
                    ));
                }
                file => files.push(file.to_string()),
            }
        }

        let stdin = match collect_input("uniq", &files, input.stdin).await {
            Ok(Some(bytes)) => bytes,
            Ok(None) => return Ok(CommandOutput::usage(self.help())),
            Err(failure) => return Ok(failure),
        };
        let mut lines: Vec<&[u8]> = stdin.split(|b| *b == b'\n').collect();
        // `split` on newline-terminated input leaves a trailing empty
        // element that is not a line; blank lines in the middle are.
        if lines.last().is_some_and(|l| l.is_empty()) {
            lines.pop();
        }

        let mut out = Vec::with_capacity(stdin.len());
        for run in lines.chunk_by(|a, b| a == b) {
            emit(&mut out, run[0], run.len(), count_runs);
        }
        Ok(CommandOutput::ok(out))
    }
}

fn emit(out: &mut Vec<u8>, line: &[u8], count: usize, count_runs: bool) {
    if count_runs {
        out.extend_from_slice(format!("{count}\t").as_bytes());
    }
    out.extend_from_slice(line);
    out.push(b'\n');
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn run_uniq(args: &[&str], stdin: &[u8]) -> CommandOutput {
        UniqCommand
            .run(CommandInput {
                args: args.iter().map(|s| s.to_string()).collect(),
                stdin: Some(stdin.to_vec()),
            })
            .await
            .expect("run returns Ok")
    }

    #[tokio::test]
    async fn collapses_adjacent_duplicates() {
        let out = run_uniq(&[], b"a\na\nb\na\n").await;
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, b"a\nb\na\n");
    }

    #[tokio::test]
    async fn c_flag_prefixes_run_lengths() {
        let out = run_uniq(&["-c"], b"a\na\nb\n").await;
        assert_eq!(out.stdout, b"2\ta\n1\tb\n");
    }

    #[tokio::test]
    async fn unterminated_last_line_gets_a_newline() {
        let out = run_uniq(&[], b"a\nb").await;
        assert_eq!(out.stdout, b"a\nb\n");
    }

    #[tokio::test]
    async fn empty_stdin_is_empty_output() {
        assert!(run_uniq(&[], b"").await.stdout.is_empty());
    }

    #[tokio::test]
    async fn no_stdin_emits_usage() {
        let out = UniqCommand
            .run(CommandInput {
                args: Vec::new(),
                stdin: None,
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 2);
        assert!(out.stdout.starts_with(b"usage: uniq"), "{out:?}");
    }

    #[tokio::test]
    async fn blank_lines_are_lines() {
        let out = run_uniq(&["-c"], b"a\n\n\nb\n").await;
        assert_eq!(out.stdout, b"1\ta\n2\t\n1\tb\n");
    }

    #[tokio::test]
    async fn named_file_is_read_instead_of_stdin() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("lines.txt");
        std::fs::write(&f, b"a\na\nb\n").unwrap();
        let out = run_uniq(&["-c", f.to_str().unwrap()], b"ignored\n").await;
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, b"2\ta\n1\tb\n");
    }

    #[tokio::test]
    async fn unknown_flag_still_errors() {
        let out = run_uniq(&["-q"], b"a\n").await;
        assert_eq!(out.exit_code, 2);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("[error] uniq: unknown flag '-q'"),
            "{stderr}"
        );
        assert!(stderr.contains("Use: uniq or uniq -c"), "{stderr}");
    }
}
