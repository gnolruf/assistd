//! The kinds of sub-agent a relay step runs as: named roles, each a loaded definition
//! with the permissions its sub-agent's tools, prompt and host-read rule follow from.

use std::sync::Arc;

use assistd_config::{AgentCommand, AgentSpec, McpServerName};
use assistd_ipc::HostName;
use assistd_llm::ReplyLimits;

use crate::agent::{AgentLimits, McpAccess, Permissions};

/// A kind of sub-agent the main model can hand a step to.
#[derive(Debug)]
pub(super) struct Role {
    /// What the model passes as a step's `agent`, and clients show.
    pub(super) name: String,
    /// When to pick it, as the model is told.
    pub(super) summary: String,
    /// Its own instructions, which follow what every sub-agent's system prompt says.
    pub(super) brief: String,
    pub(super) permissions: Permissions,
    /// The only commands its `run` offers; `None` offers every one its permissions give.
    pub(super) commands: Option<Vec<AgentCommand>>,
    /// The web hosts every run is handed beside its step's own.
    pub(super) sites: Vec<HostName>,
    pub(super) limits: AgentLimits,
    pub(super) reply: ReplyLimits,
}

/// The roles a relay offers, in the order the model is shown them; names
/// are unique.
#[derive(Debug)]
pub(super) struct Roles(Vec<Arc<Role>>);

impl Role {
    /// Whether `run` offers a sub-agent in this role the command named `command`, when
    /// its permissions give it.
    pub(super) fn offers_command(&self, command: &str) -> bool {
        self.commands
            .as_ref()
            .is_none_or(|allowed| allowed.iter().any(|allowed| allowed.as_str() == command))
    }

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

impl From<AgentSpec> for Role {
    fn from(spec: AgentSpec) -> Self {
        let capabilities = spec.capabilities;
        Self {
            name: spec.name.as_str().to_owned(),
            summary: spec.description,
            brief: spec.prompt,
            permissions: Permissions {
                files: capabilities.files,
                web: capabilities.web,
                index: capabilities.index,
                mcp: McpAccess::Servers(capabilities.mcp_servers),
            },
            commands: spec.commands,
            sites: spec
                .sites
                .into_iter()
                .filter_map(|site| HostName::try_from(site).ok())
                .collect(),
            limits: AgentLimits {
                tool_steps: spec
                    .limits
                    .tool_steps
                    .map_or(AgentLimits::default().tool_steps, u32::from),
                ..AgentLimits::default()
            },
            reply: ReplyLimits {
                reply_tokens: spec.limits.reply_tokens,
                reasoning_tokens: spec.limits.thinking_tokens,
            },
        }
    }
}

impl Roles {
    /// One role per definition in `agents`, whose names are unique.
    pub(super) fn new(agents: Vec<AgentSpec>) -> Self {
        Self(
            agents
                .into_iter()
                .map(|agent| Arc::new(Role::from(agent)))
                .collect(),
        )
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
