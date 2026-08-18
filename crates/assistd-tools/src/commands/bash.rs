//! The escape hatch: spawns a real `bash -c <script>` subprocess, gated by
//! a configurable policy (denylist, destructive-pattern confirmation,
//! timeout) and optionally wrapped in a bubblewrap sandbox.
//!
//! Spawn, output capture, and timeout live in [`crate::exec`], shared
//! with `wm open` (which uses that module's detached counterpart).
//!
//! ## Policy layering
//!
//! 1. **Denylist** (synchronous, before spawn): literal substring match.
//!    On match, return `exit 126` with the matched pattern in the error
//!    line so the LLM can pick a different approach. Never consults the
//!    confirmation gate; these patterns are too dangerous to prompt for.
//! 2. **Destructive patterns** (awaits `ConfirmationGate::confirm`):
//!    shlex-tokenized word-prefix match against each command segment. On
//!    match, the gate decides. If the gate returns `false`, return `exit
//!    126` with a cancellation message.
//! 3. **Sandbox wrap**: if the resolved sandbox mode is `Bwrap`, prefix
//!    the argv with `bwrap <default-flags> <extra-args> -- bash -c <script>`.
//! 4. **Timeout**: the spawn itself is wrapped in `tokio::time::timeout`.
//!    Exceeding the limit kills the process group (via `kill_on_drop`)
//!    and returns `exit 137` with the AC-specified format.
//!
//! ## Honest scope note
//!
//! Any syntactic check here (denylist / destructive patterns) can be
//! defeated by a sufficiently clever caller: variable expansion,
//! `$(echo rm) -rf /`, here-docs, base64 decoding. The sandbox is the
//! real defense; the pattern checks are a backstop that catches the
//! *obvious* cases the user expects blocked.

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

/// Policy bundle for the commands that spawn subprocesses: timeout,
/// denylist substrings, tokenized destructive patterns. Caller (e.g.
/// `assistd-core::build_tools`) shlex-tokenizes the config's destructive
/// patterns once at startup and passes the result here as
/// `Vec<Vec<String>>` to avoid re-parsing on every invocation.
///
/// Shared with [`crate::commands::WmCommand`], whose `open` subcommand
/// spawns model-chosen argv and is gated by the same `[tools.bash]`
/// policy.
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
    /// Construct a `BashCommand` with the given policy, sandbox, and confirmation gate.
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
    /// Test-only default: 30s timeout, no denylist, no destructive
    /// patterns, no sandbox, allow-all gate. Production paths always go
    /// through `BashCommand::new` with real config.
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
            return Ok(CommandOutput {
                stdout: self.help().into_bytes(),
                stderr: Vec::new(),
                exit_code: 2,
                attachments: Vec::new(),
            });
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
        supervise("bash", cmd, &input.stdin, self.cfg.timeout)
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
mod tests {
    use super::*;
    use crate::exec::{OUTPUT_BUF_MAX, OUTPUT_OVERFLOW_EXIT};
    use crate::policy::{AlwaysAllowGate, DenyAllGate};

    fn bash_with_cfg(cfg: BashPolicyCfg, gate: Arc<dyn ConfirmationGate>) -> BashCommand {
        BashCommand::new(Arc::new(cfg), SandboxInfo::none(), gate)
    }

    #[tokio::test]
    async fn bash_runs_echo() {
        let out = BashCommand::default()
            .run(CommandInput {
                args: vec!["echo hi".into()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, b"hi\n");
    }

    #[tokio::test]
    async fn bash_propagates_nonzero_exit() {
        let out = BashCommand::default()
            .run(CommandInput {
                args: vec!["exit 3".into()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 3);
    }

    #[tokio::test]
    async fn bash_receives_stdin() {
        let out = BashCommand::default()
            .run(CommandInput {
                args: vec!["tr a-z A-Z".into()],
                stdin: b"hello".to_vec(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, b"HELLO");
    }

    /// A timed-out script returns exit 137 with the byte-exact message
    /// `[error] bash: timed out after 30s [exit:137 | 30.0s]`. The
    /// timeout is 100ms here for speed; the format is unchanged.
    #[tokio::test]
    async fn bash_timeout_returns_137_with_ac_format() {
        let cfg = BashPolicyCfg {
            timeout: Duration::from_millis(100),
            ..Default::default()
        };
        let cmd = bash_with_cfg(cfg, Arc::new(AlwaysAllowGate));
        let out = cmd
            .run(CommandInput {
                args: vec!["sleep 5".into()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 137);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.starts_with("[error] bash: timed out after 0s [exit:137 | "),
            "{stderr}"
        );
        assert!(stderr.ends_with("s]\n"), "{stderr}");
    }

    #[tokio::test]
    async fn bash_missing_script_errors() {
        let out = BashCommand::default()
            .run(CommandInput {
                args: Vec::new(),
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 2);
    }

    /// Acceptance: `bash "<nonexistent-dep>"` must forward the subprocess's
    /// own stderr ("command not found") so the LLM sees *which* dependency
    /// is missing, not a bare exit:127.
    #[tokio::test]
    async fn bash_missing_dependency_forwards_subprocess_stderr() {
        let out = BashCommand::default()
            .run(CommandInput {
                args: vec!["assistd-definitely-not-a-real-binary-xyz".into()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 127);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("command not found"), "{stderr}");
        assert!(
            stderr.contains("assistd-definitely-not-a-real-binary-xyz"),
            "{stderr}"
        );
    }

    /// A script matching a denylist pattern is rejected before spawn,
    /// with the byte-exact error message
    /// `[error] bash: command denied by policy. Matched denylist pattern:
    /// rm -rf /. Try: a non-destructive alternative\n`.
    #[tokio::test]
    async fn ac1_bash_rm_rf_root_rejected() {
        let cfg = BashPolicyCfg {
            denylist: vec!["rm -rf /".into()],
            ..Default::default()
        };
        let cmd = bash_with_cfg(cfg, Arc::new(AlwaysAllowGate));
        let out = cmd
            .run(CommandInput {
                args: vec!["rm -rf /".into()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 126);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(
            stderr,
            "[error] bash: command denied by policy. Matched denylist pattern: rm -rf /. Try: a non-destructive alternative\n"
        );
    }

    #[tokio::test]
    async fn bash_denylist_is_case_insensitive() {
        let cfg = BashPolicyCfg {
            denylist: vec!["mkfs".into()],
            ..Default::default()
        };
        let cmd = bash_with_cfg(cfg, Arc::new(AlwaysAllowGate));
        let out = cmd
            .run(CommandInput {
                args: vec!["MKFS.ext4 /dev/sda1".into()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 126);
    }

    /// When a destructive pattern is matched and the gate approves, the
    /// command runs normally. Uses `true` as a no-op script so the test
    /// doesn't actually delete anything.
    #[tokio::test]
    async fn destructive_pattern_invokes_gate_and_proceeds_when_approved() {
        let cfg = BashPolicyCfg {
            destructive_patterns: vec![vec!["true".into()]],
            ..Default::default()
        };
        let cmd = bash_with_cfg(cfg, Arc::new(AlwaysAllowGate));
        let out = cmd
            .run(CommandInput {
                args: vec!["true".into()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 0);
    }

    /// When a destructive pattern is matched and the gate denies, the
    /// command does NOT run and returns exit 126 with a cancellation
    /// message.
    #[tokio::test]
    async fn destructive_pattern_invokes_gate_and_cancels_when_denied() {
        let cfg = BashPolicyCfg {
            destructive_patterns: vec![vec!["rm".into(), "-rf".into()]],
            ..Default::default()
        };
        // DenyAllGate is the production default for IPC-connected clients;
        // verify it blocks destructive commands as documented.
        let cmd = bash_with_cfg(cfg, Arc::new(DenyAllGate));
        let out = cmd
            .run(CommandInput {
                // Use /tmp/nonexistent so that even if the gate is buggy
                // and allows execution, no real data is lost.
                args: vec!["rm -rf /tmp/this-directory-does-not-exist-XYZ".into()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 126);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("cancelled by user"),
            "expected cancellation message, got {stderr}"
        );
        assert!(
            stderr.contains("Matched destructive pattern: rm -rf"),
            "expected matched pattern in message, got {stderr}"
        );
    }

    /// A destructive pattern substring inside a quoted literal must not
    /// trigger the gate; `echo "rm -rf"` is harmless and legitimate.
    #[tokio::test]
    async fn destructive_matcher_ignores_quoted_literals() {
        // Counter-gate that panics if called; asserts the gate is NOT
        // invoked for this script.
        struct PanicGate;
        #[async_trait]
        impl ConfirmationGate for PanicGate {
            async fn confirm(&self, _req: ConfirmationRequest) -> bool {
                panic!("gate should not be invoked for quoted literal");
            }
        }
        let cfg = BashPolicyCfg {
            destructive_patterns: vec![vec!["rm".into(), "-rf".into()]],
            ..Default::default()
        };
        let cmd = bash_with_cfg(cfg, Arc::new(PanicGate));
        let out = cmd
            .run(CommandInput {
                args: vec!["echo \"rm -rf\"".into()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, b"rm -rf\n");
    }

    /// A runaway script (`yes`) emits gigabytes per second. Without an
    /// execution-time cap, `wait_with_output()` would buffer it all into
    /// memory before the chain executor's PIPE_BUF_MAX check fires. With
    /// the cap, the child is killed at OUTPUT_BUF_MAX and we return exit
    /// 141 with a bounded `stdout`. We use a short timeout so the test
    /// is fast even if the overflow path is broken — but the test only
    /// passes if overflow (141) fires *before* timeout (137).
    #[tokio::test]
    async fn bash_output_overflow_kills_child_and_returns_141() {
        let cfg = BashPolicyCfg {
            timeout: Duration::from_secs(10),
            ..Default::default()
        };
        let cmd = bash_with_cfg(cfg, Arc::new(AlwaysAllowGate));
        let out = cmd
            .run(CommandInput {
                args: vec!["yes".into()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, OUTPUT_OVERFLOW_EXIT);
        assert!(
            out.stdout.len() <= OUTPUT_BUF_MAX,
            "stdout was {} bytes, expected <= {OUTPUT_BUF_MAX}",
            out.stdout.len()
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("output exceeded"),
            "expected overflow message in stderr, got {stderr}"
        );
    }

    /// Stderr is bounded too: a script that floods only stderr is killed
    /// once it crosses the cap. Uses `bash -c` redirection inside the
    /// script so the bytes land on fd 2.
    #[tokio::test]
    async fn bash_stderr_overflow_also_caps() {
        let cfg = BashPolicyCfg {
            timeout: Duration::from_secs(10),
            ..Default::default()
        };
        let cmd = bash_with_cfg(cfg, Arc::new(AlwaysAllowGate));
        let out = cmd
            .run(CommandInput {
                args: vec!["yes 1>&2".into()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, OUTPUT_OVERFLOW_EXIT);
        assert!(
            out.stderr.len() <= OUTPUT_BUF_MAX + 256,
            "stderr was {} bytes, expected <= ~{OUTPUT_BUF_MAX} + overflow message",
            out.stderr.len()
        );
    }

    /// A script that writes well under the cap and exits cleanly must
    /// still return exit 0 with its full stdout — i.e. the streaming
    /// path doesn't drop bytes or false-positive on overflow.
    #[tokio::test]
    async fn bash_below_cap_returns_full_output() {
        let out = BashCommand::default()
            .run(CommandInput {
                // ~50 KiB, comfortably below 10 MiB.
                args: vec!["printf '%.0sx' {1..51200}".into()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout.len(), 51200);
    }

    /// A script killed by a signal (here SIGSEGV via `kill -SEGV $$`) must
    /// report `128 + signum` (139), not the timeout sentinel 137. Before
    /// the fix, `status.code()` returning `None` collapsed both the
    /// signal-death and timeout cases to 137.
    #[cfg(unix)]
    #[tokio::test]
    async fn bash_signal_death_reports_128_plus_signum_not_timeout() {
        let out = BashCommand::default()
            .run(CommandInput {
                args: vec!["kill -SEGV $$".into()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(
            out.exit_code, 139,
            "SIGSEGV should surface as 128+11=139, not the timeout sentinel"
        );
    }

    /// Sandbox mode `None` executes bash directly with no wrapper.
    #[tokio::test]
    async fn bash_sandbox_none_runs_unsandboxed() {
        let cfg = BashPolicyCfg::default();
        let cmd = BashCommand::new(
            Arc::new(cfg),
            SandboxInfo::none(),
            Arc::new(AlwaysAllowGate),
        );
        let out = cmd
            .run(CommandInput {
                args: vec!["echo sandboxed".into()],
                stdin: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, b"sandboxed\n");
    }
}
