//! Command-execution policy: the command classifier, path guards, confirmation gates, and the
//! bwrap sandbox.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use assistd_ipc::{ConfirmSource, PlannedAction};

use crate::TurnContext;
use crate::chain::Word;
use crate::command::{CommandError, Hint};
use crate::exec::POLICY_DENIED_EXIT;

mod actions;
mod allowlist;
mod approvals;
mod confirm;
mod egress;
mod file_scope;
mod grant;
mod judge;
mod path_class;
mod review;
mod sandbox;
mod shell;

pub use actions::{ActionCheck, StandingApprovals};
use allowlist::ProgramApproval;
pub use allowlist::{APPROVALS_FILE, Allowlist, AllowlistError, SearchPath};
pub use approvals::{
    APPROVED_APPS_FILE, APPROVED_HOSTS_FILE, APPROVED_MCP_TOOLS_FILE, ApprovalGate, Approvals,
    PinnedApprovalGate, PinnedApprovals, Standing,
};
pub use confirm::{
    Answer, Approval, CONFIRM_TIMEOUT, ConfirmRouter, ConfirmationGate, ConfirmationRequest,
    IpcConfirmationGate, JudgeContext, JudgedClass, MAX_PENDING_CONFIRMS, NoPendingConfirm,
    NoPendingReview, PromptParking, Risk, SpokenAnswer, Summary, Unattended, VoiceGate,
};
pub(crate) use confirm::{spoken_list, spoken_path};
pub(crate) use egress::EgressPolicy;
pub use file_scope::{ConfinedRoot, FileScope};
pub use judge::{JUDGE_TIMEOUT, JudgeModel, JudgedCommand, SafetyJudge, Verdict};
pub(crate) use path_class::admits_edit;
pub use path_class::{GuardRoots, PathGuards, PathVerdict};
pub use review::{Action, Analysis, Class, Confirmation, Effect, analyze, classify};
#[cfg(any(test, feature = "test-support"))]
pub use sandbox::inert_sandbox;
pub(crate) use sandbox::{ConfinedSpawner, Guarded, InputBind, is_within_visible_home_entry};
pub use sandbox::{
    HomeAccess, Protected, ReaderProgram, ReaderSandbox, ReaderSandboxError, SCRATCH_MARKER,
    SandboxError, SandboxInfo, SandboxPrograms, SandboxRequest, SandboxedCommand, SandboxedServer,
    ScopeError, ScratchCap, ServerSandbox, ServerSandboxError, SharedDirs, ToolSandbox,
    ToolsDisabled, probe_sandbox,
};

/// Policy for the commands that spawn subprocesses: read-only scripts run unasked with `$HOME`
/// read-only, edits ask unless the permission mode accepts them, and unvetted scripts ask.
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

/// What the policy decided about a script before it runs.
#[derive(Debug)]
pub(crate) enum Review {
    Clear,
    Ask(Confirmation, Unattended),
    /// It would destroy a catastrophic path; holds what and where.
    Refuse(String),
}

/// A script's [`Review`], and what of `$HOME` it may change once that lets it run.
#[derive(Debug)]
pub(crate) struct ScriptReview {
    pub(crate) review: Review,
    pub(crate) home: HomeAccess,
    /// The files the script names to read, as written; a directory read throughout is
    /// left out.
    pub(crate) reads: Vec<String>,
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
    pub(crate) async fn review_script(&self, script: String) -> ScriptReview {
        let cfg = Arc::clone(&self.cfg);
        let sandbox = Arc::clone(&self.sandbox);
        on_blocking_pool(move || {
            let guards = sandbox.path_guards();
            let analysis = analyze(&script, cfg.scratch.as_deref(), Some(&cfg.allowlist));
            ScriptReview {
                home: home_access(&analysis),
                reads: files_read(&analysis),
                review: guarded_review(&guards, analysis, &cfg.edit_scope, &cfg.protected),
            }
        })
        .await
    }

    /// Refuse `script` when `review` refuses it, or asks and `turn`'s gate declines. `tool` names
    /// the command in the prompt and `op` the operation in the error.
    pub(crate) async fn authorize(
        &self,
        turn: &TurnContext,
        tool: &str,
        op: &str,
        script: &str,
        review: Review,
    ) -> Result<(), CommandError> {
        match review {
            Review::Clear => Ok(()),
            Review::Refuse(why) => Err(refused(op, &why)),
            Review::Ask(confirmation, unattended) => {
                if self
                    .confirmed(turn, tool, script, &confirmation, unattended)
                    .await
                {
                    Ok(())
                } else {
                    Err(cancelled(op, &confirmation))
                }
            }
        }
    }

    /// Ask the gate; an "always" answer also adds the offered programs to
    /// the allowlist.
    async fn confirmed(
        &self,
        turn: &TurnContext,
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
            summary: confirmation.summary(tool),
            always_allow: offered.iter().map(ToString::to_string).collect(),
            risk: confirmation.risk(),
            unattended,
            source: ConfirmSource::Main,
        };
        approvals::confirm_remembering(
            self.gate.as_ref(),
            turn,
            request,
            self.cfg.allowlist.approve(offered),
        )
        .await
    }
}

/// The path guards and gate the in-daemon commands read files under.
#[derive(Debug)]
pub(crate) struct ReadPolicy {
    pub(crate) sandbox: Arc<SandboxInfo>,
    pub(crate) gate: Arc<dyn ConfirmationGate>,
}

impl ReadPolicy {
    /// Refuse `tool`'s read of `paths`, as `scope` resolves them, when one is sensitive and
    /// `turn`'s gate declines; no mode reads a sensitive path unasked.
    pub(crate) async fn authorize(
        &self,
        turn: &TurnContext,
        tool: &str,
        scope: &FileScope,
        paths: &[&Word],
    ) -> Result<(), CommandError> {
        let Some(why) = self.first_sensitive(scope, paths).await? else {
            return Ok(());
        };
        let confirmation = Confirmation::Sensitive(why);
        let request = ConfirmationRequest {
            tool: tool.to_string(),
            script: read_as_shown(tool, paths),
            matched_pattern: confirmation.to_string(),
            summary: read_summary(paths, &confirmation),
            always_allow: Vec::new(),
            risk: confirmation.risk(),
            unattended: Unattended::Never,
            source: ConfirmSource::Main,
        };
        if self.gate.confirm(turn, request).await == Approval::Deny {
            return Err(cancelled("read", &confirmation));
        }
        Ok(())
    }

    /// What the path guards say of the first of `paths` whose read is sensitive, on the
    /// blocking pool; `Err` when a path cannot be made absolute.
    async fn first_sensitive(
        &self,
        scope: &FileScope,
        paths: &[&Word],
    ) -> Result<Option<String>, CommandError> {
        let sandbox = Arc::clone(&self.sandbox);
        let scope = scope.clone();
        let paths: Vec<Word> = paths.iter().copied().cloned().collect();
        on_blocking_pool(move || {
            let guards = sandbox.path_guards();
            paths
                .iter()
                .find_map(|path| match scope.absolute(Path::new(path.as_str())) {
                    Ok(absolute) => guards.sensitive_read(&absolute).map(Ok),
                    Err(error) => Some(Err(CommandError::io(path, &error))),
                })
                .transpose()
        })
        .await
    }
}

fn files_read(analysis: &Analysis) -> Vec<String> {
    analysis
        .effects
        .iter()
        .filter(|effect| effect.action == Action::Read && !effect.recursive)
        .map(|effect| effect.path.clone())
        .collect()
}

/// Read-only for a script that runs only programs on the read-only list, whatever the path
/// guards make of it; an edit, an unvetted script and an approved program may all write.
fn home_access(analysis: &Analysis) -> HomeAccess {
    if analysis.class == Class::ReadOnly && !analysis.runs_approved {
        HomeAccess::ReadOnly
    } else {
        HomeAccess::Writable
    }
}

/// `tool` and the `paths` it reads, each quoted, as a prompt shows them.
fn read_as_shown(tool: &str, paths: &[&Word]) -> String {
    paths.iter().fold(tool.to_string(), |shown, path| {
        format!("{shown} {:?}", path.as_str())
    })
}

/// A read of `paths` that asks because of `confirmation`, in a sentence.
fn read_summary(paths: &[&Word], confirmation: &Confirmation) -> Summary {
    let paths = paths
        .iter()
        .map(|path| spoken_path(Path::new(path.as_str())));
    Summary::of(format!("a read of {}", spoken_list(paths))).because(format!("It {confirmation}."))
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
        PathVerdict::Ordinary(Class::Edit) => edit_review(guards, &effects, edit_scope, excluded),
        PathVerdict::Ordinary(Class::Unvetted { why }) => {
            let grantable = always_allow
                .iter()
                .map(ProgramApproval::planned_action)
                .collect();
            Review::Ask(
                Confirmation::Unvetted {
                    why: why.clone(),
                    always_allow,
                },
                judged(JudgedClass::Unvetted, guards, why, grantable),
            )
        }
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
            guards
                .changes_outside_scope(effects, edit_scope, excluded)
                .unwrap_or_default()
                .into_iter()
                .map(|under| PlannedAction::Write { under })
                .collect(),
        ),
    };
    Review::Ask(confirmation, unattended)
}

fn judged(
    class: JudgedClass,
    guards: &PathGuards,
    why: String,
    grantable: Vec<PlannedAction>,
) -> Unattended {
    Unattended::Judged(JudgeContext {
        class,
        cwd: guards.cwd().map(Path::to_path_buf),
        why,
        grantable,
    })
}

fn cancelled(op: &str, confirmation: &Confirmation) -> CommandError {
    let reason = match confirmation {
        Confirmation::Unvetted { why, .. } => format!("It could not be vetted: {why}"),
        Confirmation::Sensitive(why) => format!("It {why}"),
        Confirmation::Edit(_) => format!("It {confirmation}"),
    };
    CommandError::new(
        POLICY_DENIED_EXIT,
        format_args!("{op} cancelled by user. {reason}"),
        Hint::Try,
        "a different approach",
    )
}

fn refused(op: &str, why: &str) -> CommandError {
    CommandError::new(
        POLICY_DENIED_EXIT,
        format_args!("{op} refused in every mode: it {why}"),
        Hint::Try,
        "deleting a specific subtree instead, such as a build or cache directory",
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
