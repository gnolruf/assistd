//! Walks a [`Chain`], dispatching each stage through a [`CommandRegistry`]
//! and combining results with Unix pipeline semantics.

use std::future::Future;
use std::pin::Pin;

use super::expand::{ExpandError, expand_args};
use super::{Chain, Word};
use crate::command::{CommandInput, CommandOutput, CommandRegistry, Hint, error_line};

/// Most bytes of stdout, or of stderr, that one stage or a whole chain may
/// hold. Overflow exits 141, the SIGPIPE code, so `||` fallbacks still fire.
pub const OUTPUT_MAX: usize = 10 * 1024 * 1024;

/// Execute a parsed chain. Pipes run sequentially, the left stage's stdout
/// becoming the right's stdin; every stage's stderr lines are kept, prefixed
/// `[name]\t`. Short-circuited stages emit nothing. A pipe stage whose stdout
/// passes [`OUTPUT_MAX`] stops the pipe; any other output past it, including
/// the streams `;`, `&&` and `||` join, is replaced by an overflow failure.
pub fn execute<'a>(
    chain: &'a Chain,
    registry: &'a CommandRegistry,
    stdin: Option<Vec<u8>>,
) -> Pin<Box<dyn Future<Output = CommandOutput> + Send + 'a>> {
    Box::pin(async move { within_output_max(execute_uncapped(chain, registry, stdin).await) })
}

fn execute_uncapped<'a>(
    chain: &'a Chain,
    registry: &'a CommandRegistry,
    stdin: Option<Vec<u8>>,
) -> Pin<Box<dyn Future<Output = CommandOutput> + Send + 'a>> {
    Box::pin(async move {
        match chain {
            Chain::Command(argv) => run_command(argv, registry, stdin).await,
            Chain::Pipe(l, r) => {
                let mut left = execute_uncapped(l, registry, stdin).await;
                let piped = std::mem::take(&mut left.stdout);
                if piped.len() > OUTPUT_MAX {
                    return left.then(pipe_overflow());
                }
                let right = execute_uncapped(r, registry, Some(piped)).await;
                left.then(right)
            }
            Chain::And(l, r) | Chain::Or(l, r) => {
                let left = execute(l, registry, stdin.clone()).await;
                let run_right = matches!(chain, Chain::And(..)) == (left.exit_code == 0);
                if !run_right {
                    return left;
                }
                let right = execute(r, registry, stdin).await;
                left.then(right)
            }
            Chain::Seq(l, r) => {
                let left = execute(l, registry, stdin.clone()).await;
                let right = execute(r, registry, stdin).await;
                left.then(right)
            }
        }
    })
}

fn within_output_max(mut out: CommandOutput) -> CommandOutput {
    if out.stdout.len() <= OUTPUT_MAX && out.stderr.len() <= OUTPUT_MAX {
        return out;
    }
    out.stdout.clear();
    out.stderr.truncate(last_line_end_within_max(&out.stderr));
    out.then(CommandOutput::failed(
        141,
        error_line(
            "run",
            format_args!("output exceeded {OUTPUT_MAX} bytes"),
            Hint::Try,
            "fewer files per command, or grep -l / grep -c to find what matters first",
        )
        .into_bytes(),
    ))
}

fn last_line_end_within_max(stream: &[u8]) -> usize {
    stream[..stream.len().min(OUTPUT_MAX)]
        .iter()
        .rposition(|&b| b == b'\n')
        .map_or(0, |newline| newline + 1)
}

fn pipe_overflow() -> CommandOutput {
    CommandOutput::failed(
        141,
        error_line(
            "pipe",
            format_args!("stage output exceeded {OUTPUT_MAX} bytes"),
            Hint::Try,
            "pipe through wc -l or head first to shrink the stream",
        )
        .into_bytes(),
    )
}

/// Dispatch one stage. `--help` anywhere in its args prints usage, even for
/// commands whose bare form does real work.
async fn run_command(
    words: &[Word],
    registry: &CommandRegistry,
    stdin: Option<Vec<u8>>,
) -> CommandOutput {
    let name = words.first().map(|w| w.text.as_str()).unwrap_or_default();
    if name.is_empty() {
        return CommandOutput::failed(
            2,
            error_line(
                "run",
                "empty command",
                Hint::Use,
                "run <cmd> (see tool description for available commands)",
            )
            .into_bytes(),
        );
    }

    let Some(cmd) = registry.get(name) else {
        let avail = registry.sorted_names().join(", ");
        let msg = format!("[error] unknown command: {name}. Available: {avail}\n");
        return CommandOutput::failed(127, msg.into_bytes());
    };

    let args = if cmd.expands_args() {
        match expand_args(&words[1..]).await {
            Ok(args) => args,
            Err(err) => return glob_failure(name, &err),
        }
    } else {
        words[1..].iter().map(|word| word.text.clone()).collect()
    };
    if args.iter().any(|a| a == "--help") {
        return CommandOutput::usage(cmd.help());
    }

    let out = cmd.run(CommandInput { args, stdin }).await;

    let stderr = if out.stderr.is_empty() {
        Vec::new()
    } else {
        prefix_stderr(name, &out.stderr)
    };
    CommandOutput {
        stdout: out.stdout,
        stderr,
        exit_code: out.exit_code,
        attachments: out.attachments,
    }
}

fn glob_failure(name: &str, err: &ExpandError) -> CommandOutput {
    CommandOutput::failed(
        1,
        error_line(
            name,
            err,
            Hint::Try,
            "narrow the pattern, or ls the directory first",
        )
        .into_bytes(),
    )
}

fn prefix_stderr(name: &str, raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len() + name.len() + 4);
    let mut line_start = true;
    for &b in raw {
        if line_start {
            out.push(b'[');
            out.extend_from_slice(name.as_bytes());
            out.push(b']');
            out.push(b'\t');
            line_start = false;
        }
        out.push(b);
        if b == b'\n' {
            line_start = true;
        }
    }
    out
}

#[cfg(test)]
mod tests;
