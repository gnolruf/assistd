//! `grep`, `head`, `tail`, `sort`, `uniq` and `wc`: the system's own programs, run by name
//! under `bash`'s policy and sandbox.

use std::sync::Arc;

use async_trait::async_trait;

use super::bash::{ScriptRun, run_script};
use crate::chain::Word;
use crate::command::{Command, CommandError, CommandInput, CommandOutput, Hint, Usage};
use crate::commands::reads_of;
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
/// A run that exits 0 reports the files it named as read.
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
        let ScriptRun {
            out,
            private_dirs,
            reads,
        } = run_script(&self.policy, &input.turn, name, script, stdin).await?;
        let out = if out.exit_code == 0 {
            out.touching(reads_of(&input.scope, reads).await)
        } else {
            out
        };
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
mod tests {
    use super::*;
    use crate::TurnContext;
    use crate::chain::{execute, parse_chain};
    use crate::command::CommandRegistry;
    use crate::commands::{RecordingGate, register_readers, test_sandbox};
    use crate::exec::{POLICY_DENIED_EXIT, SPAWN_FAILED_EXIT};
    use crate::test_support::bash_policy;

    /// The readers over a sandbox that starts nothing: a line the policy lets through exits
    /// [`SPAWN_FAILED_EXIT`].
    fn registry_asking(gate: &Arc<RecordingGate>) -> CommandRegistry {
        let gate: Arc<dyn ConfirmationGate> = gate.clone();
        let mut registry = CommandRegistry::new();
        let cfg = Arc::new(bash_policy());
        register_readers(&mut registry, &cfg, &test_sandbox(), &gate);
        registry
    }

    async fn run(registry: &CommandRegistry, line: &str) -> CommandOutput {
        execute(
            &parse_chain(line).expect("parses"),
            registry,
            &TurnContext::default(),
        )
        .await
    }

    /// `touch` as a command would ask; quoted into `grep`'s pattern it is only a word.
    #[tokio::test]
    async fn arguments_reach_the_program_as_words_not_code() {
        let gate = RecordingGate::new(false);
        let registry = registry_asking(&gate);
        let line = "echo 'a; touch /nonexistent/x' | grep -c \"; touch /nonexistent/x\"";

        let out = run(&registry, line).await;

        assert_eq!(out.exit_code, SPAWN_FAILED_EXIT, "{out:?}");
        assert!(gate.prompts().is_empty(), "{:?}", gate.prompts());
    }

    #[tokio::test]
    async fn a_writing_option_asks_and_a_refusal_stops_the_run() {
        let dir = tempfile::tempdir().expect("tempdir");
        let source = dir.path().join("lines");
        std::fs::write(&source, b"b\na\n").expect("source file");
        let written = dir.path().join("sorted");
        let gate = RecordingGate::new(false);
        let registry = registry_asking(&gate);
        let line = format!("sort -o {} {}", written.display(), source.display());

        let out = run(&registry, &line).await;

        assert_eq!(out.exit_code, POLICY_DENIED_EXIT, "{out:?}");
        assert!(!written.exists());
        let prompts = gate.prompts();
        assert_eq!(prompts.len(), 1, "{prompts:?}");
        assert_eq!(
            (prompts[0].0.as_str(), prompts[0].1.as_str()),
            ("sort", line.as_str())
        );
    }
}
