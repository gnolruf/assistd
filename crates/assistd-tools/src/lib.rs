//! Tool-use subsystem: [`Tool`]s the model calls with JSON, chiefly
//! [`RunTool`], which runs shell-style chains of byte-oriented [`Command`]s.

use async_trait::async_trait;
use serde_json::{Value, json};

pub mod attachment;
pub mod chain;
pub mod command;
pub mod commands;
mod exec;
pub mod memory;
pub mod memory_tools;
pub mod policy;
pub mod presentation;
pub mod run;
pub mod vision;

pub use attachment::{LoadImageError, load_image_attachment};
pub use command::{Attachment, Command, CommandInput, CommandOutput, CommandRegistry};
pub use memory::{DEFAULT_SEARCH_LIMIT, MemoryOps};
pub use memory_tools::{RecallTool, RememberTool, ReminisceTool};
pub use policy::{
    APPROVALS_FILE, Allowlist, AllowlistError, Approval, CONFIRM_ROUTER, CONFIRM_TIMEOUT,
    ConfirmRouter, ConfirmationGate, ConfirmationRequest, DestructivePattern, IpcConfirmationGate,
    NoPendingConfirm, Protected, SandboxError, SandboxInfo, SandboxRequest, SearchPath,
    inherit_confirm_router, probe_sandbox,
};
#[cfg(any(test, feature = "test-support"))]
pub use policy::{AlwaysAllowGate, DenyAllGate};
pub use run::RunTool;
pub use vision::VisionGate;

/// Prefix of every MCP-adapted tool's name (`mcp__<server>__<tool>`).
pub const MCP_TOOL_NAME_PREFIX: &str = "mcp__";

/// Why [`Tool::invoke`] produced no result. Failures the model can recover
/// from belong in the result envelope instead.
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    /// The model's arguments violate the tool's schema; the message names
    /// the offending argument.
    #[error("{0}")]
    InvalidArgs(String),
    /// A memory, conversation, or semantic store call failed.
    #[error(transparent)]
    Store(#[from] assistd_memory::MemoryError),
}

/// A single tool the LLM can invoke.
#[async_trait]
pub trait Tool: Send + Sync + 'static {
    /// Identifier the model calls this tool by.
    fn name(&self) -> &str;

    /// Description the model sees when deciding whether to call the tool.
    fn description(&self) -> &str;

    /// JSON Schema for the `arguments` object [`Tool::invoke`] accepts.
    fn parameters_schema(&self) -> Value;

    /// Execute the tool with JSON arguments and return a JSON result.
    async fn invoke(&self, args: Value) -> Result<Value, ToolError>;
}

/// Lookup table of registered tools.
#[derive(Default)]
pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
}

impl ToolRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tool by value.
    pub fn register<T: Tool>(&mut self, tool: T) {
        self.tools.push(Box::new(tool));
    }

    /// Register an already-boxed tool.
    pub fn register_boxed(&mut self, tool: Box<dyn Tool>) {
        self.tools.push(tool);
    }

    /// Look up a registered tool by its `name()`.
    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools
            .iter()
            .find(|t| t.name() == name)
            .map(|t| t.as_ref())
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Returns `true` when no tools have been registered.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Iterator over every registered tool's name.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tools.iter().map(|t| t.name())
    }

    /// Render the registry as an OpenAI chat-completions `tools` array, each
    /// entry `strict` so the model's arguments conform to its schema.
    pub fn openai_schemas(&self) -> Vec<Value> {
        self.tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.name(),
                        "description": t.description(),
                        "parameters": t.parameters_schema(),
                        "strict": true,
                    }
                })
            })
            .collect()
    }
}

/// Crate version.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Test fixtures shared across the crate's unit tests.
#[cfg(test)]
pub(crate) mod fixtures {
    /// A valid 1x1 RGBA PNG.
    pub(crate) const PNG_BYTES: &[u8] = &[
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44,
        0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1F,
        0x15, 0xC4, 0x89, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0x00,
        0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00, 0x00, 0x00, 0x00, 0x49,
        0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
    ];
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Noop;

    #[async_trait]
    impl Tool for Noop {
        fn name(&self) -> &str {
            "noop"
        }
        fn description(&self) -> &str {
            "does nothing"
        }
        fn parameters_schema(&self) -> Value {
            json!({"type": "object", "properties": {}, "additionalProperties": false})
        }
        async fn invoke(&self, _args: Value) -> Result<Value, ToolError> {
            Ok(Value::Null)
        }
    }

    #[test]
    fn registry_finds_tools_by_name() {
        let mut reg = ToolRegistry::new();
        reg.register(Noop);
        assert_eq!(reg.get("noop").map(|t| t.name()), Some("noop"));
        assert!(reg.get("missing").is_none());
    }

    #[test]
    fn openai_schemas_wraps_each_tool() {
        let mut reg = ToolRegistry::new();
        reg.register(Noop);
        assert_eq!(
            reg.openai_schemas(),
            [json!({
                "type": "function",
                "function": {
                    "name": "noop",
                    "description": "does nothing",
                    "parameters": Noop.parameters_schema(),
                    "strict": true,
                }
            })]
        );
    }
}
