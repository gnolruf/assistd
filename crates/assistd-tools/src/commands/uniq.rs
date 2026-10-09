use async_trait::async_trait;

use crate::command::{Command, CommandInput, CommandOutput};
use crate::commands::collect_input;

/// `uniq [-c] [FILE]...`: collapse runs of identical adjacent lines from
/// the named files or stdin; `-c` prefixes each with `<count>\t`.
#[derive(Debug)]
pub struct UniqCommand;

#[async_trait]
impl Command for UniqCommand {
    fn name(&self) -> &'static str {
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
                args: args.iter().map(ToString::to_string).collect(),
                stdin: Some(stdin.to_vec()),
            })
            .await
    }

    #[tokio::test]
    async fn collapses_adjacent_runs() {
        let cases: [(&[&str], &str, &str); 2] = [
            (&[], "a\na\nb\na\n", "a\nb\na\n"),
            (&["-c"], "a\n\n\nb\n", "1\ta\n2\t\n1\tb\n"),
        ];
        for (args, stdin, expected) in cases {
            let label = format!("{args:?} {stdin:?}");
            let out = run_uniq(args, stdin.as_bytes()).await;
            assert_eq!(out.exit_code, 0, "{label}");
            assert_eq!(String::from_utf8_lossy(&out.stdout), expected, "{label}");
        }
    }
}
