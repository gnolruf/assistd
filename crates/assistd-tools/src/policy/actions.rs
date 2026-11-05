//! Checks on the actions a plan anticipates: whether each could work on this machine, and
//! whether a run would stop to ask about it.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use assistd_ipc::{ActionOutlook, HostName, PermissionMode, PlannedAction};
use assistd_utils::program::find_program;

use super::allowlist::{Allowlist, ApprovalScope, SearchPath, Verdict};
use super::approvals::{ApprovalGate, Approvals};
use super::path_class::canonical_existing;
use crate::commands::wm::unlaunchable_app;
use crate::{MCP_TOOL_NAME_PREFIX, ToolRegistry};

/// The approvals that let an action run without asking, whatever the mode.
#[derive(Debug, Clone)]
pub struct StandingApprovals {
    pub programs: Arc<Allowlist>,
    pub hosts: ApprovalGate,
    pub apps: Arc<Approvals>,
    pub mcp_tools: ApprovalGate,
    /// Canonical directories below which `AcceptEdits` and `Auto` approve an edit.
    pub edit_scope: Vec<PathBuf>,
}

/// Checks an action a plan anticipates before it is shown for approval.
#[derive(Debug, Clone)]
pub struct ActionCheck {
    search_path: SearchPath,
    approvals: Option<StandingApprovals>,
}

impl ActionCheck {
    /// Checks that look programs up on `search_path`, where commands find them, and read
    /// `approvals`; without them every action asks.
    pub fn new(search_path: SearchPath, approvals: Option<StandingApprovals>) -> Self {
        Self {
            search_path,
            approvals,
        }
    }

    /// Why `action` would fail here: a program not on the search path, a host that is not
    /// one, an MCP tool `tools` lacks, an app with no launchable desktop entry, or a write
    /// directory that is relative or not a directory. Blocks on the filesystem.
    pub fn problem(&self, action: &PlannedAction, tools: &ToolRegistry) -> Option<String> {
        match action {
            PlannedAction::Program { program, .. } => self.missing_program(program),
            PlannedAction::WebHost { host } => HostName::try_from(host.clone())
                .is_err()
                .then(|| format!("`{host}` is not a host name")),
            PlannedAction::McpTool { name } => (!name.starts_with(MCP_TOOL_NAME_PREFIX)
                || tools.get(name).is_none())
            .then(|| format!("no connected MCP server offers `{name}`")),
            PlannedAction::App { desktop_entry } => unlaunchable_app(desktop_entry),
            PlannedAction::Write { under } => unwritable_dir(under),
        }
    }

    /// Whether work under `mode`, also fetching from `allowed_hosts` unasked, would run `action`
    /// without asking. Blocks on the filesystem.
    pub fn outlook(
        &self,
        action: &PlannedAction,
        mode: PermissionMode,
        allowed_hosts: &[HostName],
    ) -> ActionOutlook {
        let listed = match action {
            PlannedAction::WebHost { host } => allowed_hosts
                .iter()
                .any(|allowed| allowed.as_str().eq_ignore_ascii_case(host)),
            _ => false,
        };
        let approved = self
            .approvals
            .as_ref()
            .is_some_and(|approvals| approvals.cover(action, mode));
        let judged = matches!(
            action,
            PlannedAction::Program { .. } | PlannedAction::McpTool { .. }
        );
        if listed || approved {
            ActionOutlook::Unasked
        } else if mode == PermissionMode::Auto && judged {
            ActionOutlook::JudgeDecides
        } else {
            ActionOutlook::Asks
        }
    }

    fn missing_program(&self, program: &str) -> Option<String> {
        if program.is_empty() || program.contains('/') {
            return Some(format!("`{program}` is not a program name"));
        }
        let path_env = std::env::join_paths(&self.search_path.dirs).ok()?;
        find_program(Path::new(program), &path_env)
            .err()
            .map(|e| format!("`{program}`: {e}"))
    }
}

impl StandingApprovals {
    /// Whether an approval, or `mode` for an edit inside the edit scope, covers `action`.
    fn cover(&self, action: &PlannedAction, mode: PermissionMode) -> bool {
        match action {
            PlannedAction::Program {
                program,
                subcommand,
            } => {
                let scope = subcommand
                    .clone()
                    .map_or(ApprovalScope::OptionsOnly, ApprovalScope::Subcommand);
                self.programs.verdict(program, Some(&scope)) == Verdict::Approved
            }
            PlannedAction::WebHost { host } => self.hosts.contains(host),
            PlannedAction::McpTool { name } => self.mcp_tools.contains(name),
            PlannedAction::App { desktop_entry } => self.apps.contains(desktop_entry),
            PlannedAction::Write { under } => {
                mode != PermissionMode::Manual
                    && canonical_existing(under).is_some_and(|dir| {
                        self.edit_scope.iter().any(|scope| dir.starts_with(scope))
                    })
            }
        }
    }
}

/// Why nothing can be written under `under`, if nothing can.
fn unwritable_dir(under: &Path) -> Option<String> {
    if !under.is_absolute() {
        return Some(format!("`{}` is not an absolute path", under.display()));
    }
    (under.exists() && !under.is_dir()).then(|| format!("`{}` is not a directory", under.display()))
}
