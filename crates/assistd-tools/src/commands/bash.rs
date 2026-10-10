//! `bash SCRIPT`: a real `bash -c` subprocess behind the subprocess policy
//! (denylist, allowlist, destructive-pattern confirmation, sandbox, timeout).

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use tracing::warn;

use crate::command::{Command, CommandInput, CommandOutput, Hint, error_line};
use crate::exec::{SPAWN_FAILED_EXIT, supervise};
use crate::policy::{BashPolicyCfg, ConfirmationGate, SandboxInfo, SubprocessPolicy};

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

#[cfg(test)]
impl Default for BashCommand {
    fn default() -> Self {
        Self::new(
            Arc::new(BashPolicyCfg::default()),
            SandboxInfo::none(),
            Arc::new(crate::policy::AlwaysAllowGate),
        )
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

    fn help(&self) -> String {
        let timeout_secs = self.policy.cfg.timeout.as_secs();
        let sharing_hint = self.policy.sandbox.sharing_hint();
        format!(
            "usage: bash [-c] \"<script>\"\n\
             \n\
             Spawn a real `bash -c <script>` subprocess; a leading `-c` is \
             accepted and ignored. The escape hatch for \
             anything the in-process commands can't express: redirections, env \
             expansion, backgrounding, pipes the chain parser doesn't support.\n\
             \n\
             Stdin is forwarded to the script's stdin. Stdout/stderr/exit-code \
             are captured. Exit 137 on timeout ({timeout_secs}s default), 127 \
             if the spawn itself failed, 126 if the script is blocked by \
             policy (denylist match or user-cancelled confirmation).\n\
             \n\
             The script runs sandboxed: /tmp, /run and any other tmpfs the \
             sandbox mounts are empty and private to each call, so files the other commands put there (including \
             `Full output:` spill files) are not visible, and files the \
             script puts there are gone when it exits. For a file both need, \
             use {sharing_hint}.\n"
        )
    }

    /// The words become shell source, so bash expands them itself: a file
    /// name matched by a glob must never be parsed as code.
    fn expands_args(&self) -> bool {
        false
    }

    async fn run(&self, input: CommandInput) -> CommandOutput {
        let script_words = strip_dash_c(&input.args);
        if script_words.is_empty() {
            return CommandOutput::usage(self.help());
        }
        let script = script_words.join(" ");
        let confirmation = self.policy.review_script(script.clone()).await;
        if let Err(denied) = self
            .policy
            .authorize("bash", "command", &script, confirmation)
            .await
        {
            return denied;
        }

        let cmd = self
            .policy
            .sandbox
            .command("bash", vec!["-c".to_string(), script.clone()])
            .await;
        let private_dirs = display_list(&self.policy.sandbox.private_dirs_named(&cmd, &script));
        let out = supervise(
            "bash",
            cmd,
            input.stdin.as_deref().unwrap_or_default(),
            self.policy.cfg.timeout,
        )
        .await
        .unwrap_or_else(|e| {
            CommandOutput::failed(
                SPAWN_FAILED_EXIT,
                error_line(
                    "bash",
                    format_args!("spawn failed: {e}"),
                    Hint::Check,
                    "bash and (if configured) bwrap are on PATH",
                )
                .into_bytes(),
            )
        });
        if private_dirs.is_empty() {
            return out;
        }
        warn!(
            target: "assistd::policy",
            dirs = %private_dirs,
            "bash script names a directory that is empty and private inside the sandbox"
        );
        with_private_dir_note(out, &private_dirs, &self.policy.sandbox.sharing_hint())
    }
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
fn strip_dash_c(args: &[String]) -> &[String] {
    match args.split_first() {
        Some((flag, rest)) if flag == "-c" => rest,
        _ => args,
    }
}

#[cfg(test)]
mod tests;
