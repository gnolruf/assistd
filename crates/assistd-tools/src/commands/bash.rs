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
        let ScriptRun { out, private_dirs } =
            run_script(&self.policy, &input.turn, "bash", script, stdin).await?;
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

/// What [`run_script`] produced, and the directories the script names that the sandbox gave
/// it empty and private, comma-separated.
pub(crate) struct ScriptRun {
    pub(crate) out: CommandOutput,
    pub(crate) private_dirs: String,
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
    let ScriptReview { review, home } = policy.review_script(script.clone()).await;
    policy
        .authorize(turn, tool, "command", &script, review)
        .await?;
    let cmd = policy
        .sandbox
        .command("bash", vec!["-c".to_string(), script.clone()], home)
        .await
        .map_err(|error| spawn_failed(&error))?;
    let private_dirs = display_list(&policy.sandbox.private_dirs_named(&cmd, &script));
    let run = supervise(
        tool,
        cmd,
        policy.sandbox.confined_spawner(),
        stdin,
        policy.cfg.timeout,
    );
    let out = match &policy.sandbox.shared.scratch_cap {
        Some(cap) => scratch_capped(tool, cap, cap.guard(run).await),
        None => run.await,
    }
    .map_err(|error| spawn_failed(&error))?;
    Ok(ScriptRun { out, private_dirs })
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
    use std::path::Path;
    use std::process::Stdio;
    use std::time::Duration;

    use super::*;
    use crate::chain::{execute, parse_chain};
    use crate::command::CommandRegistry;
    use crate::command::run_rendered;
    use crate::commands::{RecordingGate, test_bash};
    use crate::exec::{OUTPUT_BUF_MAX, OUTPUT_OVERFLOW_EXIT};
    use crate::policy::{AlwaysAllowGate, DenyAllGate};

    fn bash_with_cfg(cfg: BashPolicyCfg, gate: Arc<dyn ConfirmationGate>) -> BashCommand {
        BashCommand::new(Arc::new(cfg), SandboxInfo::none(), gate)
    }

    fn with_timeout(timeout: Duration) -> BashCommand {
        bash_with_cfg(
            BashPolicyCfg {
                timeout,
                ..Default::default()
            },
            Arc::new(AlwaysAllowGate),
        )
    }

    fn bash_asking(gate: Arc<dyn ConfirmationGate>) -> BashCommand {
        bash_with_cfg(BashPolicyCfg::default(), gate)
    }

    async fn run(cmd: &BashCommand, script: &str, stdin: Option<Vec<u8>>) -> CommandOutput {
        run_rendered(
            cmd,
            CommandInput {
                args: vec![Word::quoted(script)],
                stdin,
                scope: crate::FileScope::Host,
                turn: TurnContext::default(),
            },
        )
        .await
    }

    #[tokio::test]
    async fn glob_matches_reach_bash_as_file_names_not_code() {
        let dir = tempfile::tempdir().expect("tempdir");
        let planted = dir.path().join("a$(echo injected).txt");
        std::fs::write(&planted, b"").expect("planted file");
        let mut registry = CommandRegistry::new();
        registry.register(test_bash());
        let line = format!("bash echo {}/*.txt", dir.path().display());

        let out = execute(
            &parse_chain(&line).expect("parses"),
            &registry,
            &TurnContext::default(),
        )
        .await;

        assert_eq!(out.exit_code, 0, "{out:?}");
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            format!("{}\n", planted.display())
        );
    }

    #[tokio::test]
    async fn leading_dash_c_is_accepted_as_bash_would() {
        let mut registry = CommandRegistry::new();
        registry.register(test_bash());
        let line = "bash -c 'n=$((1+4)); echo RESULT_$n | tr A-Z a-z'";

        let out = execute(
            &parse_chain(line).expect("parses"),
            &registry,
            &TurnContext::default(),
        )
        .await;

        assert_eq!(out.exit_code, 0, "{out:?}");
        assert_eq!(out.stdout, b"result_5\n");
    }

    #[tokio::test]
    async fn bash_timeout_returns_137_with_timeout_message() {
        let out = run(&with_timeout(Duration::from_millis(100)), "sleep 5", None).await;
        assert_eq!(out.exit_code, 137);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.starts_with("[error] bash: timed out after 0s [exit:137 | "),
            "{stderr}"
        );
        assert!(stderr.ends_with("s]\n"), "{stderr}");
    }

    #[tokio::test]
    async fn edit_is_cancelled_when_denied() {
        let cmd = bash_asking(Arc::new(DenyAllGate));
        let out = run(&cmd, "rm -rf /tmp/this-directory-does-not-exist-XYZ", None).await;
        assert_eq!(out.exit_code, 126);
        assert_eq!(
            String::from_utf8_lossy(&out.stderr),
            "[error] bash: command cancelled by user. It edits `/tmp/this-directory-does-not-exist-XYZ`. Try: a different approach\n"
        );
    }

    #[tokio::test]
    async fn quoted_destructive_literal_does_not_prompt() {
        let gate = RecordingGate::new(false);
        let out = run(&bash_asking(gate.clone()), "echo \"rm -rf\"", None).await;
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, b"rm -rf\n");
        assert!(gate.prompts().is_empty(), "{:?}", gate.prompts());
    }

    /// `yes` must be killed at the capture cap (141), not buffered until the
    /// timeout (137).
    #[tokio::test]
    async fn bash_output_overflow_kills_child_and_returns_141() {
        let out = run(&with_timeout(Duration::from_secs(10)), "yes", None).await;
        assert_eq!(out.exit_code, OUTPUT_OVERFLOW_EXIT);
        assert!(
            out.stdout.len() <= OUTPUT_BUF_MAX,
            "stdout was {} bytes, expected <= {OUTPUT_BUF_MAX}",
            out.stdout.len()
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("output exceeded"), "{stderr}");
    }

    /// Poll `kill -0` for up to two seconds; returns whether `pid` exited.
    #[cfg(unix)]
    async fn waited_for_exit(pid: &str) -> bool {
        for _ in 0..40 {
            let alive = std::process::Command::new("kill")
                .args(["-0", pid])
                .stderr(Stdio::null())
                .status()
                .is_ok_and(|s| s.success());
            if !alive {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        false
    }

    /// Run `script`, which must write a background child's PID to `pidfile`,
    /// and return the output plus that PID. Bounded, since a surviving
    /// grandchild would hold the output pipe open forever.
    #[cfg(unix)]
    async fn run_leaving_background_child(
        cfg: BashPolicyCfg,
        script: impl Fn(&Path) -> String,
    ) -> (CommandOutput, String) {
        let dir = tempfile::tempdir().unwrap();
        let pidfile = dir.path().join("child.pid");
        let cmd = bash_with_cfg(cfg, Arc::new(AlwaysAllowGate));
        let out = tokio::time::timeout(Duration::from_secs(10), run(&cmd, &script(&pidfile), None))
            .await
            .expect("bash did not return: a grandchild kept the output pipe open");
        let pid = std::fs::read_to_string(&pidfile)
            .unwrap()
            .trim()
            .to_string();
        assert!(!pid.is_empty(), "script did not record a background PID");
        (out, pid)
    }

    /// `sleep 300 &` outlives a leader-only SIGKILL, so this checks the whole
    /// process group is killed.
    #[cfg(unix)]
    #[tokio::test]
    async fn timeout_kills_backgrounded_grandchild() {
        let cfg = BashPolicyCfg {
            timeout: Duration::from_millis(200),
            ..Default::default()
        };
        let (out, pid) = run_leaving_background_child(cfg, |pidfile| {
            format!("sleep 300 & echo $! > {}; wait", pidfile.display())
        })
        .await;
        assert_eq!(out.exit_code, 137);
        assert!(
            waited_for_exit(&pid).await,
            "grandchild {pid} survived the timeout"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn normal_exit_kills_backgrounded_grandchild() {
        let (out, pid) = run_leaving_background_child(BashPolicyCfg::default(), |pidfile| {
            format!("sleep 300 & echo $! > {}; echo done", pidfile.display())
        })
        .await;
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, b"done\n");
        assert!(
            waited_for_exit(&pid).await,
            "grandchild {pid} survived the script's exit"
        );
    }

    #[tokio::test]
    async fn unread_stdin_does_not_block_the_timeout() {
        let cmd = with_timeout(Duration::from_millis(200));
        let out = tokio::time::timeout(
            Duration::from_secs(10),
            run(&cmd, "sleep 300", Some(vec![b'x'; 1024 * 1024])),
        )
        .await
        .expect("stdin write blocked before the timeout armed");
        assert_eq!(out.exit_code, 137);
    }

    #[tokio::test]
    async fn stdin_larger_than_a_pipe_buffer_arrives_whole() {
        let out = run(&test_bash(), "wc -c", Some(vec![b'x'; 1024 * 1024])).await;
        assert_eq!(out.exit_code, 0);
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "1048576");
    }
}
