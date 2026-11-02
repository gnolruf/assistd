//! Command-execution policy: command review, confirmation gates, and the
//! bwrap sandbox.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use assistd_ipc::ConfirmSource;

use crate::command::{CommandOutput, Hint, error_line};
use crate::exec::POLICY_DENIED_EXIT;

mod allowlist;
mod approvals;
mod confirm;
mod egress;
mod file_scope;
mod judge;
mod path_class;
mod review;
mod sandbox;
mod shell;

pub use allowlist::{APPROVALS_FILE, Allowlist, AllowlistError, SearchPath};
pub use approvals::{
    APPROVED_APPS_FILE, APPROVED_HOSTS_FILE, APPROVED_MCP_TOOLS_FILE, ApprovalGate, Approvals,
};
#[cfg(any(test, feature = "test-support"))]
pub use confirm::{AlwaysAllowGate, DenyAllGate};
pub use confirm::{
    Approval, CONFIRM_ROUTER, CONFIRM_TIMEOUT, ConfirmRouter, ConfirmationGate,
    ConfirmationRequest, IpcConfirmationGate, MAX_PENDING_CONFIRMS, NoPendingConfirm, Risk,
    inherit_confirm_router,
};
pub(crate) use egress::{EgressPolicy, NonPublic, is_public, reachable_addrs};
pub use file_scope::{ConfinedRoot, FileScope};
pub use judge::{JUDGE_TIMEOUT, JudgeModel, JudgedCommand, SafetyJudge, Verdict};
pub(crate) use path_class::split_at_existing;
pub use path_class::{GuardRoots, PathGuards, PathVerdict};
pub use review::{
    Action, Analysis, Class, Confirmation, Delete, DestructivePattern, Effect, Rules, analyze,
    check_script, classify,
};
pub(crate) use sandbox::{ConfinedSpawner, EgressRoute, InputBind, is_within_visible_home_entry};
pub use sandbox::{
    Protected, ResolvedSandboxMode, SandboxError, SandboxInfo, SandboxRequest, SharedDirs,
    ToolSandbox, ToolsDisabled, probe_sandbox,
};

/// Policy for the commands that spawn subprocesses. A command runs
/// unasked only when [`check_script`] clears it.
#[derive(Debug, Clone)]
pub struct BashPolicyCfg {
    pub timeout: Duration,
    pub destructive_patterns: Vec<DestructivePattern>,
    /// Programs that run without confirmation, shared by every
    /// subprocess-spawning command so an approval applies to all of them.
    pub allowlist: Arc<Allowlist>,
    /// Directories a command may not name without confirmation.
    pub protected: Vec<PathBuf>,
    /// The sandbox's shared scratch directory, where a redirection may
    /// write without confirmation.
    pub scratch: Option<PathBuf>,
}

#[cfg(any(test, feature = "test-support"))]
impl Default for BashPolicyCfg {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            destructive_patterns: Vec::new(),
            allowlist: Arc::new(Allowlist::unsaved(
                assistd_config::defaults::default_bash_allowed_programs(),
                SandboxInfo::none().search_path(),
            )),
            protected: Vec::new(),
            scratch: None,
        }
    }
}

impl BashPolicyCfg {
    /// The rules [`check_script`] applies.
    pub fn rules(&self) -> Rules<'_> {
        Rules {
            patterns: &self.destructive_patterns,
            allowlist: &self.allowlist,
            protected: &self.protected,
            scratch: self.scratch.as_deref(),
        }
    }
}

/// What the policy decided about a script before it runs.
#[derive(Debug)]
pub(crate) enum Review {
    Clear,
    Ask(Confirmation),
    /// It would destroy a catastrophic path; holds what and where.
    Refuse(String),
}

/// The policy, sandbox and gate a command runs model-chosen argv under.
#[derive(Debug)]
pub(crate) struct SubprocessPolicy {
    pub(crate) cfg: Arc<BashPolicyCfg>,
    pub(crate) sandbox: Arc<SandboxInfo>,
    pub(crate) gate: Arc<dyn ConfirmationGate>,
}

impl SubprocessPolicy {
    /// The path guards, then [`check_script`] when they find nothing guarded, on the blocking
    /// pool, since both read the filesystem.
    pub(crate) async fn review_script(&self, script: String) -> Review {
        let cfg = Arc::clone(&self.cfg);
        let sandbox = Arc::clone(&self.sandbox);
        on_blocking_pool(move || {
            let analysis = analyze(&script, cfg.scratch.as_deref());
            match sandbox.path_guards().verdict(analysis) {
                PathVerdict::Refused(why) => Review::Refuse(why),
                PathVerdict::Sensitive(why) => Review::Ask(Confirmation::Sensitive(why)),
                PathVerdict::Ordinary(_) => {
                    check_script(&script, &cfg.rules()).map_or(Review::Clear, Review::Ask)
                }
            }
        })
        .await
    }

    /// Refuse `script` when `review` refuses it, or asks and the gate declines. `tool` and `op`
    /// name the command and operation in the error line.
    pub(crate) async fn authorize(
        &self,
        tool: &str,
        op: &str,
        script: &str,
        review: Review,
    ) -> Result<(), CommandOutput> {
        match review {
            Review::Clear => Ok(()),
            Review::Refuse(why) => Err(refused(tool, op, &why)),
            Review::Ask(confirmation) if self.confirmed(tool, script, &confirmation).await => {
                Ok(())
            }
            Review::Ask(confirmation) => Err(cancelled(tool, op, &confirmation)),
        }
    }

    /// Ask the gate; an "always" answer also adds the offered programs to
    /// the allowlist.
    async fn confirmed(&self, tool: &str, script: &str, confirmation: &Confirmation) -> bool {
        let offered = confirmation.always_allow();
        let request = ConfirmationRequest {
            tool: tool.to_string(),
            script: script.to_string(),
            matched_pattern: confirmation.to_string(),
            always_allow: offered.iter().map(ToString::to_string).collect(),
            risk: confirmation.risk(),
            source: ConfirmSource::Main,
        };
        approvals::confirm_remembering(
            self.gate.as_ref(),
            request,
            self.cfg.allowlist.approve(offered),
        )
        .await
    }
}

fn cancelled(tool: &str, op: &str, confirmation: &Confirmation) -> CommandOutput {
    let reason = match confirmation {
        Confirmation::Pattern(pattern) => format!("Matched destructive pattern: {pattern}"),
        Confirmation::Unverifiable(why) => {
            format!("Could not rule out a destructive command: {why}")
        }
        Confirmation::Unlisted { programs, .. } => {
            format!("Not on the allowlist: {}", programs.join(", "))
        }
        Confirmation::Sensitive(why) => format!("It {why}"),
    };
    CommandOutput::failed(
        POLICY_DENIED_EXIT,
        error_line(
            tool,
            format_args!("{op} cancelled by user. {reason}"),
            Hint::Try,
            "a different approach",
        )
        .into_bytes(),
    )
}

fn refused(tool: &str, op: &str, why: &str) -> CommandOutput {
    CommandOutput::failed(
        POLICY_DENIED_EXIT,
        error_line(
            tool,
            format_args!("{op} refused in every mode: it {why}"),
            Hint::Try,
            "deleting a specific subtree instead, such as a build or cache directory",
        )
        .into_bytes(),
    )
}

/// Run filesystem-bound `work` on the blocking pool so a hung mount cannot
/// pin an async worker; a panic in `work` resumes in the caller.
pub async fn on_blocking_pool<T, F>(work: F) -> T
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    tokio::task::spawn_blocking(work)
        .await
        .unwrap_or_else(|err| std::panic::resume_unwind(err.into_panic()))
}
