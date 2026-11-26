//! What an agent is, apart from the history it runs over.

use std::sync::Arc;

use assistd_config::McpServerName;
use assistd_llm::SlotAssignment;
use assistd_tools::{MCP_TOOL_NAME_PREFIX, ToolRegistry};

/// Tool-calling steps a turn may take while it keeps making different calls.
pub(super) const MAX_TOOL_STEPS: u32 = 200;

/// Consecutive identical tool calls after which the model is treated as stuck.
pub(super) const DUPLICATE_CALL_LIMIT: usize = 3;

/// What an agent is: the prompt its history opens with, what it may reach, the tools it
/// is offered, the limits its loop holds it to, and the server slot its requests use.
#[derive(Debug, Clone)]
pub struct AgentDefinition {
    pub system_prompt: String,
    pub permissions: Permissions,
    pub tools: Arc<ToolRegistry>,
    /// Offered in place of `tools` while the turn plans.
    pub planning_tools: Option<Arc<ToolRegistry>>,
    pub limits: AgentLimits,
    pub slot: SlotAssignment,
}

impl AgentDefinition {
    /// An agent under `system_prompt` offered `tools` on `slot`, with every permission,
    /// the default limits and the same tools while it plans.
    pub fn new(system_prompt: String, tools: Arc<ToolRegistry>, slot: SlotAssignment) -> Self {
        Self {
            system_prompt,
            permissions: Permissions::all(),
            tools,
            planning_tools: None,
            limits: AgentLimits::default(),
            slot,
        }
    }
}

/// What an agent may reach beyond a directory of its own to work in. The default is
/// nothing.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Permissions {
    /// Read files: this machine's for the main agent, the inputs it is handed for a
    /// sub-agent.
    pub files: bool,
    /// Search the web and fetch pages.
    pub web: bool,
    /// Query the document index, saved memories and past sessions, read-only.
    pub index: bool,
    pub mcp: McpAccess,
}

impl Permissions {
    /// Everything, as the main agent holds.
    pub fn all() -> Self {
        Self {
            files: true,
            web: true,
            index: true,
            mcp: McpAccess::All,
        }
    }

    /// Whether what the agent reaches can hold data from this machine.
    pub fn reaches_host_data(&self) -> bool {
        self.files || self.index || self.mcp != McpAccess::default()
    }
}

/// The MCP servers whose tools an agent may call. The default is none.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum McpAccess {
    /// Only the servers named; each is a way out of a quarantine.
    Servers(Vec<McpServerName>),
    All,
}

impl McpAccess {
    /// Whether the tool named `tool` belongs to a server this allows.
    pub fn allows_tool(&self, tool: &str) -> bool {
        let Some(server_and_tool) = tool.strip_prefix(MCP_TOOL_NAME_PREFIX) else {
            return false;
        };
        match self {
            Self::All => true,
            Self::Servers(servers) => servers.iter().any(|server| {
                server_and_tool
                    .strip_prefix(server.as_str())
                    .is_some_and(|tool| tool.starts_with("__"))
            }),
        }
    }
}

impl Default for McpAccess {
    fn default() -> Self {
        Self::Servers(Vec::new())
    }
}

/// When an agent's loop withdraws its tools, leaving it to answer from what it has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AgentLimits {
    /// Tool-calling steps one turn may take.
    pub tool_steps: u32,
    /// Identical tool calls in a row that count as stuck.
    pub duplicate_calls: usize,
}

impl Default for AgentLimits {
    fn default() -> Self {
        Self {
            tool_steps: MAX_TOOL_STEPS,
            duplicate_calls: DUPLICATE_CALL_LIMIT,
        }
    }
}
