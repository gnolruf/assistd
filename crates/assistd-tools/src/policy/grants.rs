//! Grants: approvals a task run holds in place of asking, and what a confirmation needs
//! from them before it may run unasked.

use std::path::{Path, PathBuf};

use assistd_ipc::TaskGrant;

use super::allowlist::{ApprovalScope, ProgramApproval};
use super::path_class::canonical_existing;

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
