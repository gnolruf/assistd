//! `bash SCRIPT`: spawn a real `bash -c <script>` subprocess behind the
//! denylist, destructive-pattern confirmation, sandbox, and timeout
//! policy. Denylist hits exit 126 without prompting; destructive hits
//! prompt through the gate and exit 126 when refused; a timeout kills
//! the process group and exits 137.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use tracing::warn;

use crate::command::{Command, CommandInput, CommandOutput, error_line};
use crate::exec::{POLICY_DENIED_EXIT, SPAWN_FAILED_EXIT, supervise};
use crate::policy::{
    ConfirmationGate, ConfirmationRequest, SandboxAccess, SandboxInfo, matches_denylist,
    matches_destructive,
};

/// Policy for the commands that spawn subprocesses. Destructive
/// patterns are pre-tokenized so no invocation re-parses them.
#[derive(Debug, Clone)]
pub struct BashPolicyCfg {
    pub timeout: Duration,
    pub denylist: Vec<String>,
    pub destructive_patterns: Vec<Vec<String>>,
}

impl Default for BashPolicyCfg {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            denylist: Vec::new(),
            destructive_patterns: Vec::new(),
        }
    }
}

/// `bash SCRIPT`: spawn a real `bash -c <script>` subprocess, policy-gated.
pub struct BashCommand {
    cfg: Arc<BashPolicyCfg>,
    sandbox: Arc<SandboxInfo>,
    gate: Arc<dyn ConfirmationGate>,
}

impl BashCommand {
    pub fn new(
        cfg: Arc<BashPolicyCfg>,
        sandbox: Arc<SandboxInfo>,
        gate: Arc<dyn ConfirmationGate>,
    ) -> Self {
        Self { cfg, sandbox, gate }
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
        let timeout_secs = self.cfg.timeout.as_secs();
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

        if let Some(pat) = matches_denylist(&script, &self.cfg.denylist) {
            warn!(
                target: "assistd::policy",
                script = %script,
                matched = %pat,
                "bash denied by denylist"
            );
            return Ok(CommandOutput::failed(
                POLICY_DENIED_EXIT,
                error_line(
                    "bash",
                    format_args!("command denied by policy. Matched denylist pattern: {pat}"),
                    "Try",
                    "a non-destructive alternative",
                )
                .into_bytes(),
            ));
        }

        if let Some(matched) = matches_destructive(&script, &self.cfg.destructive_patterns) {
            let pattern_display = matched.join(" ");
            let approved = self
                .gate
                .confirm(ConfirmationRequest {
                    tool: "bash".to_string(),
                    script: script.clone(),
                    matched_pattern: pattern_display.clone(),
                })
                .await;
            if !approved {
                return Ok(CommandOutput::failed(
                    POLICY_DENIED_EXIT,
                    error_line(
                        "bash",
                        format_args!(
                            "cancelled by user. Matched destructive pattern: {pattern_display}"
                        ),
                        "Try",
                        "a different approach",
                    )
                    .into_bytes(),
                ));
            }
        }

        let cmd = self
            .sandbox
            .command(SandboxAccess::Default, "bash", ["-c", script.as_str()]);
        supervise(
            "bash",
            cmd,
            input.stdin.as_deref().unwrap_or_default(),
            self.cfg.timeout,
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
