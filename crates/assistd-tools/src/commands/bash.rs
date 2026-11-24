//! `bash SCRIPT`: a real `bash -c` subprocess behind the subprocess policy
//! (classifier, path guards, permission mode, sandbox, timeout).

use std::io;
use std::path::PathBuf;
use std::sync::Arc;

use assistd_utils::text::human_size;
use async_trait::async_trait;
use tracing::warn;

use crate::TurnContext;
use crate::chain::Word;
use crate::command::{Command, CommandError, CommandInput, CommandOutput, Hint, Usage, error_line};
use crate::exec::{SCRATCH_FULL_EXIT, SPAWN_FAILED_EXIT, supervise};
use crate::policy::{
    BashPolicyCfg, ConfirmationGate, Guarded, SandboxInfo, ScratchCap, ScriptReview,
    SubprocessPolicy,
};

/// `bash SCRIPT`: run a policy-gated `bash -c <script>` subprocess. Policy
/// refusals exit 126; a timeout kills the process group and exits 137.
#[derive(Debug)]
pub struct BashCommand {
    policy: SubprocessPolicy,
}

impl BashCommand {
    /// A `bash` command that runs scripts under `cfg`, inside `sandbox`,
    /// asking `gate` before any script the policy does not let through.
    pub fn new(
        cfg: Arc<BashPolicyCfg>,
        sandbox: Arc<SandboxInfo>,
        gate: Arc<dyn ConfirmationGate>,
    ) -> Self {
        Self {
            policy: SubprocessPolicy { cfg, sandbox, gate },
        }
    }
}

#[async_trait]
impl Command for BashCommand {
    fn name(&self) -> &'static str {
        "bash"
    }

    fn summary(&self) -> &'static str {
        "escape hatch: sandboxed bash -c <script> (policy-gated, private /tmp)"
    }

    fn usage(&self) -> Usage {
        let timeout_secs = self.policy.cfg.timeout.as_secs();
        let sharing_hint = self.policy.sandbox.sharing_hint();
        let body = format!(
            "Spawn a real `bash -c <script>` subprocess; a leading `-c` is \
             accepted and ignored. The escape hatch for \
             anything the other commands can't express: redirections, env \
             expansion, backgrounding, pipes the chain parser doesn't support.\n\
             \n\
             Stdin is forwarded to the script's stdin. Stdout/stderr/exit-code \
             are captured. Exit 137 on timeout ({timeout_secs}s default), 127 \
             if the spawn itself failed, 126 if the script is blocked by \
             policy (user-cancelled confirmation).\n\
             \n\
             The script runs sandboxed: /tmp, /run and any other tmpfs the \
             sandbox mounts are empty and private to each call, so files the other commands put there (including \
             `Full output:` spill files) are not visible, and files the \
             script puts there are gone when it exits. For a file both need, \
             use {sharing_hint}.\n"
        );
        Usage::new("[-c] \"<script>\"", body)
    }

    /// The words become shell source, so bash expands them itself: a file
    /// name matched by a glob must never be parsed as code.
    fn expands_args(&self) -> bool {
        false
    }

    async fn run(&self, input: CommandInput) -> Result<CommandOutput, CommandError> {
        let script_words = strip_dash_c(&input.args);
        if script_words.is_empty() {
            return Ok(CommandOutput::usage(self.help()));
        }
        let script = Word::joined_by_spaces(script_words);
        let stdin = input.stdin.as_deref().unwrap_or_default();
        let ScriptRun {
            out, private_dirs, ..
        } = run_script(&self.policy, &input.turn, "bash", script, stdin).await?;
        if private_dirs.is_empty() {
            return Ok(out);
        }
        warn!(
            target: "assistd::policy",
            dirs = %private_dirs,
            "bash script names a directory that is empty and private inside the sandbox"
        );
        Ok(with_private_dir_note(
            out,
            &private_dirs,
            &self.policy.sandbox.sharing_hint(),
        ))
    }
}

/// What [`run_script`] produced, the directories the script names that the sandbox gave
/// it empty and private, comma-separated, and the files its review found it names to read.
pub(crate) struct ScriptRun {
    pub(crate) out: CommandOutput,
    pub(crate) private_dirs: String,
    pub(crate) reads: Vec<String>,
}

/// Run `script` as `bash -c` under `policy` within `turn`, reporting as `tool`: reviewed and authorized,
/// then sandboxed with the home access its review gave it, fed `stdin`, timed and capped, and
/// killed if it outgrows the scratch cap.
pub(crate) async fn run_script(
    policy: &SubprocessPolicy,
    turn: &TurnContext,
    tool: &str,
    script: String,
    stdin: &[u8],
) -> Result<ScriptRun, CommandError> {
    let ScriptReview {
        review,
        home,
        reads,
    } = policy.review_script(script.clone()).await;
    policy
        .authorize(turn, tool, "command", &script, review)
        .await?;
    let sandboxed = policy
        .sandbox
        .command("bash", vec!["-c".to_string(), script.clone()], home)
        .await
        .map_err(|error| spawn_failed(&error))?;
    let private_dirs = display_list(&sandboxed.private_dirs_named(&script));
    let run = supervise(
        tool,
        sandboxed.command,
        policy.sandbox.confined_spawner(),
        stdin,
        policy.cfg.timeout,
    );
    let out = match &policy.sandbox.shared.scratch_cap {
        Some(cap) => scratch_capped(tool, cap, cap.guard(run).await),
        None => run.await,
    }
    .map_err(|error| spawn_failed(&error))?;
    Ok(ScriptRun {
        out,
        private_dirs,
        reads,
    })
}

/// The output of a `run` guarded by `cap`: an error in place of a run the scratch directory
/// outgrew `cap` under, and a note when files were removed to fit it.
fn scratch_capped(
    tool: &str,
    cap: &ScratchCap,
    run: Guarded<io::Result<CommandOutput>>,
) -> io::Result<CommandOutput> {
    let dir = cap.dir().display();
    let max = human_size(cap.max_bytes());
    let mut out = run.output.unwrap_or_else(|| {
        let killed = error_line(
            tool,
            format_args!("killed: {dir} grew past its {max} cap"),
            Hint::Try,
            "writing less there",
        );
        Ok(CommandOutput::failed(SCRATCH_FULL_EXIT, killed))
    })?;
    if run.removed > 0 {
        if out.stderr.last().is_some_and(|last| *last != b'\n') {
            out.stderr.push(b'\n');
        }
        let note = format!(
            "[note] {tool}: {dir} outgrew its {max} cap; removed {} of its oldest files\n",
            run.removed
        );
        out.stderr.extend_from_slice(note.as_bytes());
    }
    Ok(out)
}

fn spawn_failed(error: &io::Error) -> CommandError {
    CommandError::new(
        SPAWN_FAILED_EXIT,
        format_args!("spawn failed: {error}"),
        Hint::Check,
        "bash and (if configured) bwrap are on PATH",
    )
}

fn with_private_dir_note(mut out: CommandOutput, dirs: &str, sharing_hint: &str) -> CommandOutput {
    if out.stderr.last().is_some_and(|last| *last != b'\n') {
        out.stderr.push(b'\n');
    }
    let note = format!(
        "[note] bash: the sandbox gives each bash call its own empty {dirs}: files other \
         commands put there are not visible, and files a script puts there are gone when it \
         exits. {}: {sharing_hint}\n",
        Hint::Use,
    );
    out.stderr.extend_from_slice(note.as_bytes());
    out
}

fn display_list(dirs: &[PathBuf]) -> String {
    let dirs: Vec<_> = dirs.iter().map(|dir| dir.to_string_lossy()).collect();
    dirs.join(", ")
}

/// The script words without a leading `-c`, which the model writes out of
/// habit although this command already runs `bash -c`.
fn strip_dash_c(args: &[Word]) -> &[Word] {
    match args.split_first() {
        Some((flag, rest)) if flag.as_str() == "-c" => rest,
        _ => args,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::{execute, parse_chain};
    use crate::command::CommandRegistry;
    use crate::command::run_rendered;
    use crate::commands::{RecordingGate, test_sandbox};
    use crate::exec::POLICY_DENIED_EXIT;
    use crate::test_support::{DenyAllGate, bash_policy};

    /// Bash over a sandbox that starts nothing: a script the policy lets through exits
    /// [`SPAWN_FAILED_EXIT`].
    fn bash_asking(gate: Arc<dyn ConfirmationGate>) -> BashCommand {
        BashCommand::new(Arc::new(bash_policy()), test_sandbox(), gate)
    }

    async fn run(cmd: &BashCommand, script: &str) -> CommandOutput {
        run_rendered(
            cmd,
            CommandInput {
                args: vec![Word::quoted(script)],
                stdin: None,
                scope: crate::FileScope::Host,
                turn: TurnContext::default(),
            },
        )
        .await
    }

    /// The script `line`, an edit the gate refuses, asked the gate about.
    async fn script_asked_about(line: &str) -> String {
        let gate = RecordingGate::new(false);
        let mut registry = CommandRegistry::new();
        registry.register(bash_asking(gate.clone()));
        let chain = parse_chain(line).expect("parses");
        let out = execute(&chain, &registry, &TurnContext::default()).await;
        assert_eq!(out.exit_code, POLICY_DENIED_EXIT, "{out:?}");
        let prompts = gate.prompts();
        let [(_, script, _)] = prompts.as_slice() else {
            panic!("expected exactly one prompt, got {prompts:?}");
        };
        script.clone()
    }

    #[tokio::test]
    async fn a_glob_reaches_bash_unexpanded_so_a_matched_name_is_never_code() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("a$(echo injected).txt"), b"").expect("planted file");
        let glob = format!("{}/*.txt", dir.path().display());

        let script = script_asked_about(&format!("bash touch {glob}")).await;

        assert_eq!(script, format!("touch {glob}"));
    }

    #[tokio::test]
    async fn leading_dash_c_is_accepted_as_bash_would() {
        let script = script_asked_about("bash -c 'touch /nonexistent/assistd-test'").await;
        assert_eq!(script, "touch /nonexistent/assistd-test");
    }

    #[tokio::test]
    async fn edit_is_cancelled_when_denied() {
        let cmd = bash_asking(Arc::new(DenyAllGate));
        let out = run(&cmd, "rm -rf /tmp/this-directory-does-not-exist-XYZ").await;
        assert_eq!(out.exit_code, 126);
        assert_eq!(
            String::from_utf8_lossy(&out.stderr),
            "[error] bash: command cancelled by user. It edits `/tmp/this-directory-does-not-exist-XYZ`. Try: a different approach\n"
        );
    }

    #[tokio::test]
    async fn quoted_destructive_literal_does_not_prompt() {
        let gate = RecordingGate::new(false);
        let out = run(&bash_asking(gate.clone()), "echo \"rm -rf\"").await;
        assert_eq!(out.exit_code, SPAWN_FAILED_EXIT, "{out:?}");
        assert!(gate.prompts().is_empty(), "{:?}", gate.prompts());
    }
}
