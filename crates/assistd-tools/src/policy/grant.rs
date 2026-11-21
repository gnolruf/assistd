//! A turn's grant: the actions the user of a headless turn approved together, out loud, which
//! then run without asking for the rest of that turn.

use assistd_ipc::PlannedAction;
use assistd_utils::path::canonical_existing;

use super::confirm::{ConfirmationRequest, Unattended, spoken_path};
use super::path_class::admits_edit;
use crate::MCP_TOOL_NAME_PREFIX;

const DESKTOP_ENTRY_SUFFIX: &str = ".desktop";

/// The actions granted to one turn. A program named without a subcommand covers every run of
/// it; a write directory covers what the edit scope would: files below it, none hidden.
#[derive(Debug, Default)]
pub(super) struct TurnGrant {
    actions: Vec<PlannedAction>,
}

impl TurnGrant {
    /// A grant of `actions`, its write directories canonical. Blocks on the filesystem.
    pub(super) fn canonical(actions: Vec<PlannedAction>) -> Self {
        let actions = actions
            .into_iter()
            .map(|action| match action {
                PlannedAction::Write { under } => PlannedAction::Write {
                    under: canonical_existing(&under).unwrap_or(under),
                },
                other => other,
            })
            .collect();
        Self { actions }
    }

    pub(super) fn merge(&mut self, more: Self) {
        self.actions.extend(more.actions);
    }

    /// Whether everything `req` needs approved was granted. A request no mode approves
    /// unasked, such as one touching a sensitive path, is never covered.
    pub(super) fn covers(&self, req: &ConfirmationRequest) -> bool {
        match &req.unattended {
            Unattended::WebHost(host) => {
                self.grants(&PlannedAction::WebHost { host: host.clone() })
            }
            Unattended::Judged(context) => {
                !context.grantable.is_empty()
                    && context.grantable.iter().all(|needed| self.grants(needed))
            }
            Unattended::Never | Unattended::ScopedEdit => false,
        }
    }

    /// The web hosts granted, as named.
    pub(super) fn hosts(&self) -> Vec<String> {
        self.actions
            .iter()
            .filter_map(|action| match action {
                PlannedAction::WebHost { host } => Some(host.clone()),
                _ => None,
            })
            .collect()
    }

    fn grants(&self, needed: &PlannedAction) -> bool {
        self.actions.iter().any(|granted| allows(granted, needed))
    }
}

/// Whether `granted` lets `needed` run, where a needed write names the path it changes.
fn allows(granted: &PlannedAction, needed: &PlannedAction) -> bool {
    match (granted, needed) {
        (
            PlannedAction::Program {
                program: granted,
                subcommand: only,
            },
            PlannedAction::Program {
                program,
                subcommand,
            },
        ) => granted.eq_ignore_ascii_case(program) && (only.is_none() || only == subcommand),
        (PlannedAction::WebHost { host: granted }, PlannedAction::WebHost { host }) => {
            granted.eq_ignore_ascii_case(host)
        }
        (PlannedAction::McpTool { name: granted }, PlannedAction::McpTool { name }) => {
            granted == name
        }
        (
            PlannedAction::App {
                desktop_entry: granted,
            },
            PlannedAction::App { desktop_entry },
        ) => entry_stem(granted) == entry_stem(desktop_entry),
        (PlannedAction::Write { under }, PlannedAction::Write { under: changed }) => {
            admits_edit(under, changed)
        }
        _ => false,
    }
}

fn entry_stem(desktop_entry: &str) -> &str {
    desktop_entry
        .strip_suffix(DESKTOP_ENTRY_SUFFIX)
        .unwrap_or(desktop_entry)
}

/// What `actions` ask for, to follow "I need to", with every one of them named.
pub(super) fn spoken_needs(actions: &[PlannedAction]) -> String {
    let named = |verb: &str, name: fn(&PlannedAction) -> Option<String>| {
        let names: Vec<String> = actions.iter().filter_map(name).collect();
        (!names.is_empty()).then(|| format!("{verb} {}", listed(&names, " and ")))
    };
    let needs: Vec<String> = [
        named("edit files in", |action| match action {
            PlannedAction::Write { under } => Some(spoken_path(under)),
            _ => None,
        }),
        named("run", |action| match action {
            PlannedAction::Program {
                program,
                subcommand: Some(subcommand),
            } => Some(format!("{program} {subcommand}")),
            PlannedAction::Program { program, .. } => Some(program.clone()),
            _ => None,
        }),
        named("reach", |action| match action {
            PlannedAction::WebHost { host } => Some(host.clone()),
            _ => None,
        }),
        named("call", |action| match action {
            PlannedAction::McpTool { name } => Some(spoken_mcp_tool(name)),
            _ => None,
        }),
        named("open", |action| match action {
            PlannedAction::App { desktop_entry } => Some(entry_stem(desktop_entry).to_string()),
            _ => None,
        }),
    ]
    .into_iter()
    .flatten()
    .collect();
    listed(&needs, ", and ")
}

/// `items` in a row, the last one joined by `last_joint`.
fn listed(items: &[String], last_joint: &str) -> String {
    match items {
        [] => String::new(),
        [only] => only.clone(),
        [rest @ .., last] => format!("{}{last_joint}{last}", rest.join(", ")),
    }
}

/// An MCP tool's full name as a sentence can say it: the tool, then the server it is from.
fn spoken_mcp_tool(name: &str) -> String {
    let plain = name.strip_prefix(MCP_TOOL_NAME_PREFIX).unwrap_or(name);
    match plain.split_once("__") {
        Some((server, tool)) => format!("{} from {server}", tool.replace('_', " ")),
        None => plain.replace('_', " "),
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use assistd_ipc::ConfirmSource;

    use super::*;
    use crate::policy::{JudgeContext, JudgedClass, Risk, Summary};

    fn program(program: &str, subcommand: Option<&str>) -> PlannedAction {
        PlannedAction::Program {
            program: program.into(),
            subcommand: subcommand.map(Into::into),
        }
    }

    fn write(under: &str) -> PlannedAction {
        PlannedAction::Write {
            under: PathBuf::from(under),
        }
    }

    fn host(host: &str) -> PlannedAction {
        PlannedAction::WebHost { host: host.into() }
    }

    fn request(unattended: Unattended) -> ConfirmationRequest {
        ConfirmationRequest {
            tool: "bash".into(),
            script: String::new(),
            matched_pattern: String::new(),
            summary: Summary::of("a shell command"),
            always_allow: Vec::new(),
            risk: Risk::Elevated,
            unattended,
            source: ConfirmSource::Main,
        }
    }

    fn needing(grantable: Vec<PlannedAction>) -> ConfirmationRequest {
        request(Unattended::Judged(JudgeContext {
            class: JudgedClass::Unvetted,
            cwd: None,
            why: String::new(),
            grantable,
        }))
    }

    #[test]
    fn grant_covers_only_requests_whose_every_need_was_granted() {
        let grant = TurnGrant {
            actions: vec![
                program("cargo", None),
                program("git", Some("commit")),
                write("/work/site"),
                host("docs.rs"),
            ],
        };

        assert!(grant.covers(&needing(vec![program("cargo", Some("build"))])));
        assert!(grant.covers(&needing(vec![program("git", Some("commit"))])));
        assert!(!grant.covers(&needing(vec![program("git", Some("push"))])));
        assert!(!grant.covers(&needing(vec![program("git", None)])));
        assert!(!grant.covers(&needing(vec![
            program("cargo", Some("build")),
            program("make", None)
        ])));

        assert!(grant.covers(&needing(vec![write("/work/site/pages/index.html")])));
        assert!(!grant.covers(&needing(vec![write("/work/site")])));
        assert!(!grant.covers(&needing(vec![write("/work/site/.git/hooks/pre-commit")])));
        assert!(!grant.covers(&needing(vec![write("/work/other/notes.txt")])));

        assert!(grant.covers(&request(Unattended::WebHost("docs.rs".into()))));
        assert!(!grant.covers(&request(Unattended::WebHost("static.docs.rs".into()))));

        assert!(!grant.covers(&needing(Vec::new())));
        assert!(!grant.covers(&request(Unattended::Never)));
    }

    #[test]
    fn spoken_needs_name_every_action() {
        let actions = [
            write("/work/site"),
            program("cargo", None),
            program("git", Some("commit")),
            host("docs.rs"),
            host("crates.io"),
            host("github.com"),
            PlannedAction::McpTool {
                name: "mcp__tracker__create_issue".into(),
            },
        ];
        assert_eq!(
            spoken_needs(&actions),
            "edit files in site in work, run cargo and git commit, reach docs.rs, crates.io and \
             github.com, and call create issue from tracker"
        );
    }
}
