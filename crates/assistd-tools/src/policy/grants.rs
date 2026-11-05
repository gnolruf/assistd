//! Grants: approvals a task run holds in place of asking, and what a confirmation needs
//! from them before it may run unasked.

use std::path::{Path, PathBuf};

use assistd_ipc::TaskGrant;
use assistd_utils::program::find_program;

use super::allowlist::{ApprovalScope, ProgramApproval, SearchPath};
use super::path_class::canonical_existing;
use crate::commands::wm::unlaunchable_app;
use crate::{MCP_TOOL_NAME_PREFIX, ToolRegistry};

/// What grants must cover for a confirmation to run without asking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Grantable {
    /// It always asks: no grant covers it.
    Never,
    /// A grant must cover each of these; never empty.
    Needs(Vec<GrantNeed>),
}

impl Grantable {
    /// [`Self::Needs`] of `needs`, or [`Self::Never`] when there are none.
    pub fn needs(needs: Vec<GrantNeed>) -> Self {
        if needs.is_empty() {
            Self::Never
        } else {
            Self::Needs(needs)
        }
    }

    /// The grants that would cover it, as saved on a task; empty for [`Self::Never`].
    pub fn grants(&self) -> Vec<TaskGrant> {
        let mut grants = Vec::new();
        if let Self::Needs(needs) = self {
            for grant in needs.iter().map(GrantNeed::grant) {
                if !grants.contains(&grant) {
                    grants.push(grant);
                }
            }
        }
        grants
    }
}

/// One thing a confirmation would let happen that a grant must cover.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrantNeed {
    /// A run of a program that "always allow" could approve, in that approval's scope.
    Program(ProgramApproval),
    WebHost(String),
    /// An MCP tool's full registry name.
    McpTool(String),
    /// An app's desktop entry id.
    App(String),
    /// A change to this canonical path: a write, a delete or an attribute change.
    Write(PathBuf),
}

impl GrantNeed {
    /// The grant that covers it; a write's covers its parent directory.
    fn grant(&self) -> TaskGrant {
        match self {
            Self::Program(approval) => TaskGrant::Program {
                program: approval.name.clone(),
                subcommand: match &approval.scope {
                    ApprovalScope::OptionsOnly => None,
                    ApprovalScope::Subcommand(subcommand) => Some(subcommand.clone()),
                },
            },
            Self::WebHost(host) => TaskGrant::WebHost { host: host.clone() },
            Self::McpTool(name) => TaskGrant::McpTool { name: name.clone() },
            Self::App(entry) => TaskGrant::App {
                desktop_entry: entry.clone(),
            },
            Self::Write(target) => TaskGrant::Write {
                under: target.parent().unwrap_or(target).to_path_buf(),
            },
        }
    }

    /// Whether `grant` covers it exactly as an "always allow" answer would; a write must land
    /// strictly below the granted directory once that is resolved. Blocks on the filesystem.
    fn covered_by(&self, grant: &TaskGrant) -> bool {
        match (self, grant) {
            (
                Self::Program(approval),
                TaskGrant::Program {
                    program,
                    subcommand,
                },
            ) => {
                *program == approval.name
                    && match (&approval.scope, subcommand) {
                        (ApprovalScope::OptionsOnly, None) => true,
                        (ApprovalScope::Subcommand(asked), Some(granted)) => asked == granted,
                        _ => false,
                    }
            }
            (Self::WebHost(asked), TaskGrant::WebHost { host }) => asked.eq_ignore_ascii_case(host),
            (Self::McpTool(asked), TaskGrant::McpTool { name }) => asked == name,
            (Self::App(asked), TaskGrant::App { desktop_entry }) => asked == desktop_entry,
            (Self::Write(target), TaskGrant::Write { under }) => lands_below(target, under),
            _ => false,
        }
    }
}

/// The approvals one task run holds.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Grants(Vec<TaskGrant>);

impl Grants {
    pub fn new(grants: Vec<TaskGrant>) -> Self {
        Self(grants)
    }

    /// Hold each of `grants` not already held.
    pub fn extend(&mut self, grants: Vec<TaskGrant>) {
        for grant in grants {
            if !self.0.contains(&grant) {
                self.0.push(grant);
            }
        }
    }

    /// For each of `grantable`'s needs, a grant covering it; `None` unless every need is
    /// covered. Blocks on the filesystem.
    pub fn covering(&self, grantable: &Grantable) -> Option<Vec<TaskGrant>> {
        let Grantable::Needs(needs) = grantable else {
            return None;
        };
        if needs.is_empty() {
            return None;
        }
        needs
            .iter()
            .map(|need| self.0.iter().find(|grant| need.covered_by(grant)).cloned())
            .collect()
    }
}

/// Checks a grant a plan anticipates against this machine before it is offered for approval.
#[derive(Debug, Clone)]
pub struct GrantCheck {
    search_path: SearchPath,
}

impl GrantCheck {
    /// Checks that look programs up on `search_path`, where commands find them.
    pub fn new(search_path: SearchPath) -> Self {
        Self { search_path }
    }

    /// Why `grant` would cover nothing here: a program not on the search path, a host that is
    /// not one, an MCP tool `tools` lacks, an app with no launchable desktop entry, or a write
    /// directory that is relative or not a directory. Blocks on the filesystem.
    pub fn problem(&self, grant: &TaskGrant, tools: &ToolRegistry) -> Option<String> {
        match grant {
            TaskGrant::Program { program, .. } => self.missing_program(program),
            TaskGrant::WebHost { host } => url::Host::parse(host)
                .is_err()
                .then(|| format!("`{host}` is not a host name")),
            TaskGrant::McpTool { name } => (!name.starts_with(MCP_TOOL_NAME_PREFIX)
                || tools.get(name).is_none())
            .then(|| format!("no connected MCP server offers `{name}`")),
            TaskGrant::App { desktop_entry } => unlaunchable_app(desktop_entry),
            TaskGrant::Write { under } => unwritable_dir(under),
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

/// Why no write grant can be made under `under`, if none can.
fn unwritable_dir(under: &Path) -> Option<String> {
    if !under.is_absolute() {
        return Some(format!("`{}` is not an absolute path", under.display()));
    }
    (under.exists() && !under.is_dir()).then(|| format!("`{}` is not a directory", under.display()))
}

/// Whether canonical `target` lies strictly below `under` once `under` is resolved; a relative
/// `under` covers nothing.
fn lands_below(target: &Path, under: &Path) -> bool {
    if !under.is_absolute() {
        return false;
    }
    canonical_existing(under).is_some_and(|dir| *target != dir && target.starts_with(&dir))
}

#[cfg(test)]
mod tests;
