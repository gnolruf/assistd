//! `bash SCRIPT`: spawn a real `bash -c <script>` subprocess behind the
//! denylist, destructive-pattern confirmation, sandbox, and timeout
//! policy. Denylist hits exit 126 without prompting; destructive hits
//! prompt through the gate and exit 126 when refused; a timeout kills
//! the process group and exits 137.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;

use crate::command::{Command, CommandInput, CommandOutput, error_line};
use crate::exec::{SPAWN_FAILED_EXIT, supervise};
use crate::policy::{
    BashPolicyCfg, ConfirmationGate, SandboxAccess, SandboxInfo, SubprocessPolicy,
    matches_destructive,
};

/// `bash SCRIPT`: spawn a real `bash -c <script>` subprocess, policy-gated.
pub struct BashCommand {
    policy: SubprocessPolicy,
}

impl BashCommand {
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
        use crate::policy::AlwaysAllowGate;
        Self::new(
            Arc::new(BashPolicyCfg::default()),
            SandboxInfo::none(),
            Arc::new(AlwaysAllowGate),
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

    async fn run(&self, input: CommandInput) -> Result<CommandOutput> {
        if input.args.is_empty() {
            return Ok(CommandOutput::usage(self.help()));
        }
        let script = input.args.join(" ");
        let destructive = matches_destructive(&script, &self.policy.cfg.destructive_patterns);
        if let Err(denied) = self
            .policy
            .authorize("bash", "command", &script, destructive)
            .await
        {
            return Ok(denied);
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
        .or_else(|e| {
            Ok(CommandOutput::failed(
                SPAWN_FAILED_EXIT,
                error_line(
                    "bash",
                    format_args!("spawn failed: {e}"),
                    "Check",
                    "bash and (if configured) bwrap are on PATH",
                )
                .into_bytes(),
            ))
        })
    }
}

#[cfg(test)]
mod tests;
