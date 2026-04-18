//! The escape hatch: spawns a real `bash -c <script>` subprocess, gated by
//! a configurable policy (denylist, destructive-pattern confirmation,
//! timeout) and optionally wrapped in a bubblewrap sandbox.
//!
//! The spawn pattern (`kill_on_drop(true)` + `process_group(0)` on Unix)
//! mirrors `assistd-llm/src/llama_server/process.rs` — any mid-command
//! daemon shutdown kills the whole process group, preventing grandchild
//! leaks if the script itself forked.
//!
//! ## Policy layering
//!
//! 1. **Denylist** (synchronous, before spawn): literal substring match.
//!    On match, return `exit 126` with the matched pattern in the error
//!    line so the LLM can pick a different approach. Never consults the
//!    confirmation gate — these patterns are too dangerous to prompt for.
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
//! defeated by a sufficiently clever caller — variable expansion,
//! `$(echo rm) -rf /`, here-docs, base64 decoding. The sandbox is the
//! real defense; the pattern checks are a backstop that catches the
//! *obvious* cases the user expects blocked.

use std::ffi::OsStr;
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use async_trait::async_trait;
use tokio::io::AsyncWriteExt;
use tokio::process::Command as ProcCommand;
use tokio::time::timeout;
use tracing::warn;

use crate::command::{Command, CommandInput, CommandOutput, error_line};
use crate::policy::{
    ConfirmationGate, ConfirmationRequest, ResolvedSandboxMode, SandboxInfo, matches_denylist,
    matches_destructive,
};

/// Exit code for policy denial. POSIX "command found but not executable" is
/// the closest semantic match to "we recognize the command but refuse it".
const POLICY_DENIED_EXIT: i32 = 126;

/// Exit code for timeout (128 + SIGKILL=9). The process group is killed
/// when the `tokio::time::timeout` future fires and the `tokio::Child`
/// handle's `kill_on_drop(true)` destructor runs.
const TIMEOUT_EXIT: i32 = 137;

/// Bash-policy bundle: timeout, denylist substrings, tokenized destructive
/// patterns. Caller (e.g. `assistd-core::build_tools`) shlex-tokenizes
/// the config's destructive patterns once at startup and passes the result
/// here as `Vec<Vec<String>>` to avoid re-parsing on every invocation.
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

        // 1. Denylist: synchronous, bypasses the gate.
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

        // 2. Destructive patterns: require gate approval.
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

        // 3. Build the subprocess argv, wrapping in bwrap if configured.
        let start = Instant::now();
        let mut cmd = build_command(&self.sandbox, &script);
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        #[cfg(unix)]
        cmd.process_group(0);

        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                return Ok(CommandOutput::failed(
                    127,
                    error_line(
                        "bash",
                        format_args!("spawn failed: {e}"),
                        "Check",
                        "bash and (if configured) bwrap are on PATH",
                    )
                    .into_bytes(),
                ));
            }
        };

        if let Some(mut stdin) = child.stdin.take() {
            if !input.stdin.is_empty() {
                let _ = stdin.write_all(&input.stdin).await;
            }
            // Drop closes the pipe, signaling EOF to the child.
            drop(stdin);
        }

        // 4. Timeout. On expiry, the tokio::Child is dropped, which sends
        //    SIGKILL to the process group thanks to kill_on_drop + the
        //    process_group(0) we set above.
        let output = match timeout(self.cfg.timeout, child.wait_with_output()).await {
            Ok(Ok(o)) => o,
            Ok(Err(e)) => {
                return Ok(CommandOutput::failed(
                    1,
                    error_line(
                        "bash",
                        format_args!("wait failed: {e}"),
                        "Try",
                        "re-running the command",
                    )
                    .into_bytes(),
                ));
            }
            Err(_) => {
                // AC #2: exact one-line format including the `[exit:N | Ms]`
                // suffix inline. This is a deliberate divergence from the
                // `error_line` / presentation-footer convention because
                // the acceptance criterion pins the exact form.
                let elapsed = start.elapsed();
                let secs = self.cfg.timeout.as_secs();
                let elapsed_secs = elapsed.as_secs_f64();
                let msg = format!(
                    "[error] bash: timed out after {secs}s [exit:{TIMEOUT_EXIT} | {elapsed_secs:.1}s]\n"
                );
                return Ok(CommandOutput::failed(TIMEOUT_EXIT, msg.into_bytes()));
            }
        };

        let exit_code = output.status.code().unwrap_or(TIMEOUT_EXIT);
        Ok(CommandOutput {
            stdout: output.stdout,
            stderr: output.stderr,
            exit_code,
            attachments: Vec::new(),
        })
    }
}

/// Default bubblewrap flags applied before any user `bwrap_extra_args`.
/// Read-only root, writable `$HOME` + `/tmp`, standard `/dev` and `/proc`,
/// isolated pid/ipc/uts namespaces, dies with the daemon. Crucially *not*
/// `--unshare-net`: the assistant legitimately needs curl/pip/etc., so
/// network isolation is opt-in via `bwrap_extra_args`.
fn default_bwrap_flags() -> Vec<String> {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/".to_string());
    vec![
        "--ro-bind".into(),
        "/".into(),
        "/".into(),
        "--bind".into(),
        home.clone(),
        home.clone(),
        "--bind".into(),
        "/tmp".into(),
        "/tmp".into(),
        "--dev".into(),
        "/dev".into(),
        "--proc".into(),
        "/proc".into(),
        "--tmpfs".into(),
        "/run".into(),
        "--unshare-pid".into(),
        "--unshare-ipc".into(),
        "--unshare-uts".into(),
        "--new-session".into(),
        "--die-with-parent".into(),
        "--setenv".into(),
        "HOME".into(),
        home,
        "--setenv".into(),
        "PATH".into(),
        "/usr/local/bin:/usr/bin:/bin".into(),
    ]
}

fn build_command(sandbox: &SandboxInfo, script: &str) -> ProcCommand {
    match &sandbox.mode {
        ResolvedSandboxMode::None => {
            let mut cmd = ProcCommand::new("bash");
            cmd.arg("-c").arg(script);
            cmd
        }
        ResolvedSandboxMode::Bwrap { path } => {
            let mut cmd = ProcCommand::new(path);
            cmd.args(default_bwrap_flags().iter().map(OsStr::new));
            cmd.args(sandbox.extra_args.iter().map(OsStr::new));
            cmd.arg("--");
            cmd.arg("bash").arg("-c").arg(script);
            cmd
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    /// Acceptance for AC #2: a timed-out bash script returns exit 137 with
    /// the byte-exact message format `[error] bash: timed out after 30s
    /// [exit:137 | 30.0s]`. The timeout is parameterized to 100ms for
    /// test speed, but the format is otherwise identical.
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
    /// is missing — not a bare exit:127.
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

    /// Acceptance for AC #1: a script matching a denylist pattern is
    /// rejected before spawn, with the byte-exact error message
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
    /// trigger the gate — `echo "rm -rf"` is harmless and legitimate.
    #[tokio::test]
    async fn destructive_matcher_ignores_quoted_literals() {
        // Counter-gate that panics if called — asserts the gate is NOT
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
