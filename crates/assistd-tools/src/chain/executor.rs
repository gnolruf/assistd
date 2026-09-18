//! Walks a [`super::Chain`] AST, dispatching each stage through a
//! [`crate::CommandRegistry`] and gluing the results together according to
//! Unix pipeline semantics.
//!
//! Pipelining is sequential: the left stage runs to completion and its
//! stdout becomes the right stage's stdin. [`PIPE_BUF_MAX`] caps that
//! buffer so a runaway stage can't exhaust daemon memory.

use anyhow::Result;
use std::future::Future;
use std::pin::Pin;

use super::expand::expand_args;
use super::{Chain, Word};
use crate::command::{CommandInput, CommandOutput, CommandRegistry, error_line};

/// Maximum bytes buffered between pipe stages. Overflow exits 141, the
/// SIGPIPE code, so `||` fallbacks still fire.
pub const PIPE_BUF_MAX: usize = 10 * 1024 * 1024;

/// Execute a parsed command chain.
///
/// The returned `CommandOutput`'s `stderr` is the concatenation of every
/// stage's stderr, with each command's output prefixed `"[name]\t"` so
/// the caller can tell which stage spoke. Short-circuited branches
/// (right side of `&&` on failure, right side of `||` on success) emit
/// nothing.
pub fn execute<'a>(
    chain: &'a Chain,
    registry: &'a CommandRegistry,
    stdin: Option<Vec<u8>>,
) -> Pin<Box<dyn Future<Output = Result<CommandOutput>> + Send + 'a>> {
    Box::pin(async move {
        match chain {
            Chain::Command(argv) => run_command(argv, registry, stdin).await,
            Chain::Pipe(l, r) => {
                let mut left = execute(l, registry, stdin).await?;
                let piped = std::mem::take(&mut left.stdout);
                if piped.len() > PIPE_BUF_MAX {
                    let overflow = CommandOutput::failed(
                        141,
                        error_line(
                            "pipe",
                            format_args!("stage output exceeded {PIPE_BUF_MAX} bytes"),
                            "Try",
                            "pipe through wc -l or head first to shrink the stream",
                        )
                        .into_bytes(),
                    );
                    return Ok(left.then(overflow));
                }
                let right = execute(r, registry, Some(piped)).await?;
                Ok(left.then(right))
            }
            Chain::And(l, r) | Chain::Or(l, r) => {
                let left = execute(l, registry, stdin.clone()).await?;
                let run_right = matches!(chain, Chain::And(..)) == (left.exit_code == 0);
                if !run_right {
                    return Ok(left);
                }
                let right = execute(r, registry, stdin).await?;
                Ok(left.then(right))
            }
            Chain::Seq(l, r) => {
                let left = execute(l, registry, stdin.clone()).await?;
                let right = execute(r, registry, stdin).await?;
                Ok(left.then(right))
            }
        }
    })
}

async fn run_command(
    words: &[Word],
    registry: &CommandRegistry,
    stdin: Option<Vec<u8>>,
) -> Result<CommandOutput> {
    let name = words.first().map(|w| w.text.as_str()).unwrap_or_default();
    if name.is_empty() {
        return Ok(CommandOutput::failed(
            2,
            error_line(
                "run",
                "empty command",
                "Use",
                "run <cmd> (see tool description for available commands)",
            )
            .into_bytes(),
        ));
    }

    let Some(cmd) = registry.get(name) else {
        let avail = registry.sorted_names().join(", ");
        let msg = format!("[error] unknown command: {name}. Available: {avail}\n");
        return Ok(CommandOutput::failed(127, msg.into_bytes()));
    };

    // Every command answers `--help`, including the ones whose no-arg
    // form does real work (`ls`, `echo`) and so never reaches their own
    // usage text.
    let args = expand_args(&words[1..]);
    if args.iter().any(|a| a == "--help") {
        return Ok(CommandOutput::usage(cmd.help()));
    }

    let out = cmd.run(CommandInput { args, stdin }).await?;

    let stderr = if out.stderr.is_empty() {
        Vec::new()
    } else {
        prefix_stderr(name, &out.stderr)
    };
    Ok(CommandOutput {
        stdout: out.stdout,
        stderr,
        exit_code: out.exit_code,
        attachments: out.attachments,
    })
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
