//! `bash SCRIPT`: a real `bash -c` subprocess behind the subprocess policy
//! (denylist, allowlist, destructive-pattern confirmation, sandbox, timeout).

use std::sync::Arc;

use async_trait::async_trait;

use crate::command::{Command, CommandInput, CommandOutput, Hint, error_line};
use crate::exec::{SPAWN_FAILED_EXIT, supervise};
use crate::policy::{
    BashPolicyCfg, ConfirmationGate, SandboxAccess, SandboxInfo, SubprocessPolicy, check_script,
};

/// `bash SCRIPT`: run a policy-gated `bash -c <script>` subprocess. Policy
/// refusals exit 126; a timeout kills the process group and exits 137.
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
    fn name(&self) -> &str {
        "bash"
    }

    fn summary(&self) -> &'static str {
        "escape hatch: run a bash -c <script> subprocess (policy-gated)"
    }

    fn help(&self) -> String {
        let timeout_secs = self.policy.cfg.timeout.as_secs();
        format!(
            "usage: bash \"<script>\"\n\
             \n\
             Spawn a real `bash -c <script>` subprocess. The escape hatch for \
             anything the in-process commands can't express: redirections, env \
             expansion, backgrounding, pipes the chain parser doesn't support.\n\
             \n\
             Stdin is forwarded to the script's stdin. Stdout/stderr/exit-code \
             are captured. Exit 137 on timeout ({timeout_secs}s default), 127 \
             if the spawn itself failed, 126 if the script is blocked by \
             policy (denylist match or user-cancelled confirmation).\n"
        )
    }

    /// The words become shell source, so bash expands them itself: a file
    /// name matched by a glob must never be parsed as code.
    fn expands_args(&self) -> bool {
        false
    }

    async fn run(&self, input: CommandInput) -> CommandOutput {
        if input.args.is_empty() {
            return CommandOutput::usage(self.help());
        }
        let script = input.args.join(" ");
        let confirmation = check_script(&script, &self.policy.cfg.rules());
        if let Err(denied) = self
            .policy
            .authorize("bash", "command", &script, confirmation)
            .await
        {
            return denied;
        }

        let cmd =
            self.policy
                .sandbox
                .command(SandboxAccess::Default, "bash", ["-c", script.as_str()]);
        supervise(
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
        })
    }
}

#[cfg(test)]
mod tests;
