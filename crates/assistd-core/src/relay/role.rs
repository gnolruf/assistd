//! The kinds of sub-agent a relay step runs as: named roles, each with the
//! permissions its sub-agent's tools, prompt and host-read rule follow from.

use std::sync::Arc;

use assistd_config::McpServerName;

use crate::agent::{McpAccess, Permissions};

/// A kind of sub-agent the main model can hand a step to.
#[derive(Debug)]
pub(super) struct Role {
    /// What the model passes as a step's `agent`, and clients show.
    pub(super) name: String,
    /// When to pick it, as the model is told.
    pub(super) summary: String,
    /// Its job, as its own system prompt states it.
    pub(super) brief: String,
    pub(super) permissions: Permissions,
}

/// The roles a relay offers, in the order the model is shown them; names
/// are unique.
#[derive(Debug)]
pub(super) struct Roles(Vec<Arc<Role>>);

impl Role {
    /// What a sub-agent in this role can touch, as the model is told.
    pub(super) fn reach_described(&self) -> String {
        let permissions = &self.permissions;
        let mut clauses = vec![
            if permissions.files {
                "sees the files and directories listed in `inputs`, read-only, and no other \
             file of this machine"
            } else {
                "gets no files, so `inputs` must be empty"
            }
            .to_string(),
        ];
        if permissions.index {
            clauses
                .push("can search the indexed documents, saved memories and past sessions".into());
        }
        if let Some(servers) = named_servers(&permissions.mcp) {
            clauses.push(format!("can call the tools of {servers}"));
        }
        clauses.push(
            match (permissions.web, permissions.reaches_host_data()) {
                (false, _) => "has no network",
                (true, false) => "can search the web and fetch pages",
                (true, true) => {
                    "can search the web and fetch pages, each fetch asking the user once it \
                     has read any of that"
                }
            }
            .to_string(),
        );
        clauses.join(", ")
    }
}

impl Roles {
    /// The roles assistd ships, which configuration cannot change.
    pub(super) fn built_in() -> Self {
        Self(vec![
            Arc::new(Role {
                name: "file_review".into(),
                summary: "For documents, logs and code too large to read here.".into(),
                brief: "You review local files handed to you: documents, logs and code.".into(),
                permissions: Permissions {
                    files: true,
                    ..Permissions::default()
                },
            }),
            Arc::new(Role {
                name: "web_research".into(),
                summary: "For research and reading web pages.".into(),
                brief: "You research on the web: search, read the pages that answer the task, \
                        and name the URLs your report relies on."
                    .into(),
                permissions: Permissions {
                    web: true,
                    ..Permissions::default()
                },
            }),
            Arc::new(Role {
                name: "file_web_research".into(),
                summary: "For a task that needs local files and the web together, such as \
                          checking a project's dependencies for advisories or a config against \
                          its upstream documentation."
                    .into(),
                brief: "You work from local files handed to you and from the web: read what \
                        the task needs from each, and name the paths and URLs your report \
                        relies on."
                    .into(),
                permissions: Permissions {
                    files: true,
                    web: true,
                    ..Permissions::default()
                },
            }),
        ])
    }

    pub(super) fn named(&self, name: &str) -> Option<&Arc<Role>> {
        self.0.iter().find(|role| role.name == name)
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = &Role> {
        self.0.iter().map(AsRef::as_ref)
    }

    /// The quoted names of the roles whose permissions `wanted` accepts, as a list for
    /// the model to read.
    pub(super) fn quoted_names(&self, wanted: impl Fn(&Permissions) -> bool) -> String {
        self.iter()
            .filter(|role| wanted(&role.permissions))
            .map(|role| format!("\"{}\"", role.name))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// The servers `mcp` allows as the model is told of them; `None` when it allows none.
fn named_servers(mcp: &McpAccess) -> Option<String> {
    match mcp {
        McpAccess::All => Some("every MCP server".to_string()),
        McpAccess::Servers(servers) if servers.is_empty() => None,
        McpAccess::Servers(servers) => {
            let names: Vec<&str> = servers.iter().map(McpServerName::as_str).collect();
            Some(format!("the MCP servers {}", names.join(", ")))
        }
    }
}
