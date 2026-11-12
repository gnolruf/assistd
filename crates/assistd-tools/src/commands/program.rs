//! `grep`, `head`, `tail`, `sort`, `uniq` and `wc`: the system's own programs, run by name
//! under `bash`'s policy and sandbox.

use std::sync::Arc;

use async_trait::async_trait;

use super::bash::{ScriptRun, run_script};
use crate::chain::Word;
use crate::command::{Command, CommandError, CommandInput, CommandOutput, Hint, Usage};
use crate::policy::{BashPolicyCfg, ConfirmationGate, SandboxInfo, SubprocessPolicy};

const PROGRAMS: [Program; 6] = [
    Program {
        name: "grep",
        usage: "[OPTION]... PATTERN [FILE|DIR]...",
        summary: "GNU grep: lines matching PATTERN (-E -F -P, -i -n -v -o -l -c -r, -A/-B/-C N)",
    },
    Program {
        name: "head",
        usage: "[OPTION]... [FILE]...",
        summary: "GNU head: first lines of FILE or stdin (default 10; -n N, -c BYTES)",
    },
    Program {
        name: "tail",
        usage: "[OPTION]... [FILE]...",
        summary: "GNU tail: last lines of FILE or stdin (default 10; -n N, -n +N from line N)",
    },
    Program {
        name: "sort",
        usage: "[OPTION]... [FILE]...",
        summary: "GNU sort: sort lines of FILE or stdin (-n -r -u -f, -k FIELD, -t SEP)",
    },
    Program {
        name: "uniq",
        usage: "[OPTION]... [FILE]",
        summary: "GNU uniq: collapse adjacent duplicate lines of FILE or stdin (-c -d -u -i)",
    },
    Program {
        name: "wc",
        usage: "[OPTION]... [FILE]...",
        summary: "GNU wc: count lines, words and bytes of FILE or stdin (-l -w -c -m)",
    },
];

#[derive(Debug)]
struct Program {
    name: &'static str,
    usage: &'static str,
    summary: &'static str,
}

/// One of the system's text programs as a `run` command: its arguments are quoted into a
/// script that runs as a `bash` script would, so a write option or a guarded path asks.
#[derive(Debug)]
pub struct ProgramCommand {
    program: &'static Program,
    policy: SubprocessPolicy,
}

impl ProgramCommand {
    /// Every program command, each running under `cfg`, inside `sandbox`, asking `gate`.
    pub fn all(
        cfg: &Arc<BashPolicyCfg>,
        sandbox: &Arc<SandboxInfo>,
        gate: &Arc<dyn ConfirmationGate>,
    ) -> impl Iterator<Item = Self> {
        PROGRAMS.iter().map(move |program| Self {
            program,
            policy: SubprocessPolicy {
                cfg: Arc::clone(cfg),
                sandbox: Arc::clone(sandbox),
                gate: Arc::clone(gate),
            },
        })
    }
}

#[async_trait]
impl Command for ProgramCommand {
    fn name(&self) -> &str {
        self.program.name
    }

    fn summary(&self) -> &'static str {
        self.program.summary
    }

    fn usage(&self) -> Usage {
        let Program { name, usage, .. } = self.program;
        let body = format!(
            "The system's GNU {name}: every option it documents works; `bash \"{name} --help\"` \
             lists them. With no FILE it reads the piped input.\n\
             \n\
             It runs in the sandbox, where /tmp and /run are empty and private to each call. \
             For a file there, pipe it in: `cat FILE | {name} ...`.\n"
        );
        Usage::new(*usage, body)
    }

    async fn run(&self, input: CommandInput) -> Result<CommandOutput, CommandError> {
        let name = self.program.name;
        let words = std::iter::once(name).chain(input.args.iter().map(Word::as_str));
        let script = shlex::try_join(words).map_err(|_| {
            CommandError::usage(
                "an argument holds a NUL byte",
                format_args!("{name} with printable arguments"),
            )
        })?;
        let stdin = input.stdin.as_deref().unwrap_or_default();
        let ScriptRun { out, private_dirs } =
            run_script(&self.policy, &input.turn, name, script, stdin).await?;
        Ok(if private_dirs.is_empty() {
            out
        } else {
            with_private_dir_note(out, name, &private_dirs)
        })
    }
}

fn with_private_dir_note(mut out: CommandOutput, name: &str, dirs: &str) -> CommandOutput {
    if out.stderr.last().is_some_and(|last| *last != b'\n') {
        out.stderr.push(b'\n');
    }
    let note = format!(
        "[note] {name}: it runs in the sandbox, where {dirs} is empty and private to each \
         call. {}: cat FILE | {name} ... to read a file there\n",
        Hint::Use,
    );
    out.stderr.extend_from_slice(note.as_bytes());
    out
}

#[cfg(test)]
mod tests;
