use async_trait::async_trait;

use crate::command::{Command, CommandInput, CommandOutput};
use crate::commands::collect_input;

/// `uniq [-c] [FILE]...`: collapse runs of identical adjacent lines from
/// the named files or stdin; `-c` prefixes each with `<count>\t`.
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

    async fn run(&self, input: CommandInput) -> CommandOutput {
        let mut count_runs = false;
        let mut files = Vec::new();
        for arg in &input.args {
            match arg.as_str() {
                "-c" => count_runs = true,
                flag if flag.starts_with('-') && flag.len() > 1 => {
                    return CommandOutput::usage_error(
                        "uniq",
                        format_args!("unknown flag '{flag}'"),
                        "uniq or uniq -c",
                    );
                }
                file => files.push(file.to_string()),
            }
        }

        let text = match collect_input("uniq", &files, input.stdin).await {
            Ok(Some(bytes)) => bytes,
            Ok(None) => return CommandOutput::usage(self.help()),
            Err(failure) => return failure,
        };
        let mut lines: Vec<&[u8]> = text.split(|b| *b == b'\n').collect();
        if lines.last().is_some_and(|l| l.is_empty()) {
            lines.pop();
        }

        let mut out = Vec::with_capacity(text.len());
        for run in lines.chunk_by(|a, b| a == b) {
            emit(&mut out, run[0], count_runs.then_some(run.len()));
        }
        CommandOutput::ok(out)
    }
}

fn emit(out: &mut Vec<u8>, line: &[u8], count: Option<usize>) {
    if let Some(count) = count {
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
    }

    #[tokio::test]
    async fn collapses_adjacent_runs() {
        let cases: [(&[&str], &str, &str); 5] = [
            (&[], "a\na\nb\na\n", "a\nb\na\n"),
            (&["-c"], "a\na\nb\n", "2\ta\n1\tb\n"),
            (&[], "a\nb", "a\nb\n"),
            (&[], "", ""),
            (&["-c"], "a\n\n\nb\n", "1\ta\n2\t\n1\tb\n"),
        ];
        for (args, stdin, expected) in cases {
            let label = format!("{args:?} {stdin:?}");
            let out = run_uniq(args, stdin.as_bytes()).await;
            assert_eq!(out.exit_code, 0, "{label}");
            assert_eq!(String::from_utf8_lossy(&out.stdout), expected, "{label}");
        }
    }

    #[tokio::test]
    async fn no_stdin_emits_usage() {
        let out = UniqCommand
            .run(CommandInput {
                args: Vec::new(),
                stdin: None,
            })
            .await;
        assert_eq!(out.exit_code, 2);
        assert!(out.stdout.starts_with(b"usage: uniq"), "{out:?}");
    }

    #[tokio::test]
    async fn named_file_is_read_instead_of_stdin() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("lines.txt");
        std::fs::write(&file, b"a\na\nb\n").unwrap();
        let out = run_uniq(&["-c", file.to_str().unwrap()], b"ignored\n").await;
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, b"2\ta\n1\tb\n");
    }

    #[tokio::test]
    async fn unknown_flag_errors() {
        let out = run_uniq(&["-q"], b"a\n").await;
        assert_eq!(out.exit_code, 2);
        assert_eq!(
            String::from_utf8_lossy(&out.stderr),
            "[error] uniq: unknown flag '-q'. Use: uniq or uniq -c\n"
        );
    }
}
