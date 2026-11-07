//! Command-execution policy: the command classifier, path guards, confirmation gates, and the
//! bwrap sandbox.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use assistd_ipc::ConfirmSource;

use crate::command::{CommandOutput, Hint, error_line};
use crate::exec::POLICY_DENIED_EXIT;

mod actions;
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

pub use actions::{ActionCheck, StandingApprovals};
pub use allowlist::{APPROVALS_FILE, Allowlist, AllowlistError, SearchPath};
pub use approvals::{
    APPROVED_APPS_FILE, APPROVED_HOSTS_FILE, APPROVED_MCP_TOOLS_FILE, ApprovalGate, Approvals,
};
#[cfg(any(test, feature = "test-support"))]
pub use confirm::{AlwaysAllowGate, DenyAllGate};
pub use confirm::{
    Approval, CONFIRM_ROUTER, CONFIRM_TIMEOUT, ConfirmRouter, ConfirmationGate,
    ConfirmationRequest, IpcConfirmationGate, JudgeContext, JudgedClass, MAX_PENDING_CONFIRMS,
    NoPendingConfirm, NoPendingReview, PromptParking, Risk, Unattended, inherit_confirm_router,
};
pub(crate) use egress::{EgressPolicy, NonPublic, is_public, reachable_addrs};
pub use file_scope::{ConfinedRoot, FileScope};
pub use judge::{JUDGE_TIMEOUT, JudgeModel, JudgedCommand, SafetyJudge, Verdict};
pub(crate) use path_class::split_at_existing;
pub use path_class::{GuardRoots, PathGuards, PathVerdict};
pub use review::{Action, Analysis, Class, Confirmation, Delete, Effect, analyze, classify};
pub(crate) use sandbox::{ConfinedSpawner, InputBind, is_within_visible_home_entry};
pub use sandbox::{
    Protected, ResolvedSandboxMode, SandboxError, SandboxInfo, SandboxRequest, SharedDirs,
    ToolSandbox, ToolsDisabled, probe_sandbox,
};

/// Policy for the commands that spawn subprocesses: read-only scripts run unasked, edits ask
/// unless the permission mode accepts them, and unvetted scripts ask.
#[derive(Debug, Clone)]
pub struct BashPolicyCfg {
    pub timeout: Duration,
    /// "Always allow" approvals, which vet the programs they cover; shared by every
    /// subprocess-spawning command so an approval applies to all of them.
    pub allowlist: Arc<Allowlist>,
    /// Directories a command may not name without confirmation.
    pub protected: Vec<PathBuf>,
    /// The sandbox's shared scratch directory, where a redirection may
    /// write without confirmation.
    pub scratch: Option<PathBuf>,
    /// Canonical directories below which an edit of ordinary paths runs unasked in
    /// [`assistd_ipc::PermissionMode::AcceptEdits`] and Auto.
    pub edit_scope: Vec<PathBuf>,
}

#[cfg(any(test, feature = "test-support"))]
impl Default for BashPolicyCfg {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            allowlist: Arc::new(Allowlist::unsaved(SandboxInfo::none().search_path())),
            protected: Vec::new(),
            scratch: None,
            edit_scope: Vec::new(),
        }
    }
}

/// What the policy decided about a script before it runs.
#[derive(Debug)]
pub(crate) enum Review {
    Clear,
    Ask(Confirmation, Unattended),
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
    /// The path guards, then the script's class, with "always allow" approvals vetting the
    /// programs they cover: an edit is checked against the edit scope, and an unvetted script
    /// asks. Runs on the blocking pool, since both read the filesystem.
    pub(crate) async fn review_script(&self, script: String) -> Review {
        let cfg = Arc::clone(&self.cfg);
        let sandbox = Arc::clone(&self.sandbox);
        on_blocking_pool(move || {
            let guards = sandbox.path_guards();
            let analysis = analyze(&script, cfg.scratch.as_deref(), Some(&cfg.allowlist));
            guarded_review(&guards, analysis, &cfg.edit_scope, &cfg.protected)
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
            Review::Ask(confirmation, unattended) => {
                if self
                    .confirmed(tool, script, &confirmation, unattended)
                    .await
                {
                    Ok(())
                } else {
                    Err(cancelled(tool, op, &confirmation))
                }
            }
        }
    }

    /// Ask the gate; an "always" answer also adds the offered programs to
    /// the allowlist.
    async fn confirmed(
        &self,
        tool: &str,
        script: &str,
        confirmation: &Confirmation,
        unattended: Unattended,
    ) -> bool {
        let offered = confirmation.always_allow();
        let request = ConfirmationRequest {
            tool: tool.to_string(),
            script: script.to_string(),
            matched_pattern: confirmation.to_string(),
            always_allow: offered.iter().map(ToString::to_string).collect(),
            risk: confirmation.risk(),
            unattended,
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

/// What the path guards make of `analysis`: an edit is checked against `edit_scope`, minus
/// `excluded`, and an unvetted script asks. Blocks on the filesystem.
pub(crate) fn guarded_review(
    guards: &PathGuards,
    analysis: Analysis,
    edit_scope: &[PathBuf],
    excluded: &[PathBuf],
) -> Review {
    let effects = analysis.effects.clone();
    let always_allow = analysis.always_allow.clone();
    match guards.verdict(analysis) {
        PathVerdict::Refused(why) => Review::Refuse(why),
        PathVerdict::Sensitive(why) => Review::Ask(Confirmation::Sensitive(why), Unattended::Never),
        PathVerdict::Ordinary(Class::ReadOnly) => Review::Clear,
        PathVerdict::Ordinary(Class::Edit { .. }) => {
            edit_review(guards, &effects, edit_scope, excluded)
        }
        PathVerdict::Ordinary(Class::Unvetted { why }) => Review::Ask(
            Confirmation::Unvetted {
                why: why.clone(),
                always_allow,
            },
            judged(JudgedClass::Unvetted, guards, why),
        ),
    }
}

/// An ask for an edit with `effects`: unattended in accept-edits mode when every change lands
/// inside `edit_scope` and outside `excluded`, else put to the judge in auto mode.
fn edit_review(
    guards: &PathGuards,
    effects: &[Effect],
    edit_scope: &[PathBuf],
    excluded: &[PathBuf],
) -> Review {
    let confirmation = Confirmation::Edit(
        effects
            .iter()
            .filter(|effect| effect.action != Action::Read)
            .map(|effect| effect.path.clone())
            .collect(),
    );
    let unattended = match guards.first_out_of_scope(effects, edit_scope, excluded) {
        None => Unattended::ScopedEdit,
        Some(path) => judged(
            JudgedClass::Edit,
            guards,
            format!("it changes `{path}`, outside the edit scope"),
        ),
    };
    Review::Ask(confirmation, unattended)
}

fn judged(class: JudgedClass, guards: &PathGuards, why: String) -> Unattended {
    Unattended::Judged(JudgeContext {
        class,
        cwd: guards.cwd().map(Path::to_path_buf),
        why,
    })
}

fn cancelled(tool: &str, op: &str, confirmation: &Confirmation) -> CommandOutput {
    let reason = match confirmation {
        Confirmation::Unvetted { why, .. } => format!("It could not be vetted: {why}"),
        Confirmation::Sensitive(why) => format!("It {why}"),
        Confirmation::Edit(_) => format!("It {confirmation}"),
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
