//! Walks a [`Chain`], dispatching each stage through a [`CommandRegistry`]
//! and combining results with Unix pipeline semantics.

use std::future::Future;
use std::pin::Pin;

use super::expand::expand_args;
use super::{Chain, Word};
use crate::command::{CommandInput, CommandOutput, CommandRegistry, Hint, error_line};

/// Maximum bytes buffered between pipe stages. Overflow exits 141, the
/// SIGPIPE code, so `||` fallbacks still fire.
pub const PIPE_BUF_MAX: usize = 10 * 1024 * 1024;

/// Execute a parsed chain. Pipes run sequentially, the left stage's stdout
/// (capped at [`PIPE_BUF_MAX`]) becoming the right's stdin; every stage's
/// stderr lines are kept, prefixed `[name]\t`. Short-circuited stages emit nothing.
pub fn execute<'a>(
    chain: &'a Chain,
    registry: &'a CommandRegistry,
    stdin: Option<Vec<u8>>,
) -> Pin<Box<dyn Future<Output = CommandOutput> + Send + 'a>> {
    Box::pin(async move {
        match chain {
            Chain::Command(argv) => run_command(argv, registry, stdin).await,
            Chain::Pipe(l, r) => {
                let mut left = execute(l, registry, stdin).await;
                let piped = std::mem::take(&mut left.stdout);
                if piped.len() > PIPE_BUF_MAX {
                    return left.then(pipe_overflow());
                }
                let right = execute(r, registry, Some(piped)).await;
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

fn pipe_overflow() -> CommandOutput {
    CommandOutput::failed(
        141,
        error_line(
            "pipe",
            format_args!("stage output exceeded {PIPE_BUF_MAX} bytes"),
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
        expand_args(&words[1..])
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
