//! Tool-use subsystem: the trait every invokable tool implements, plus
//! the registry the LLM looks up tool calls in.
//!
//! The daemon exposes a single LLM-facing tool (`run`) whose argument is
//! a Unix-shell-style command line. `run` parses the line into a
//! [`chain::Chain`] AST and dispatches each stage through a
//! [`CommandRegistry`] of internal Rust handlers. Two-tier split: `Tool`
//! is what the LLM sees (JSON-in/JSON-out); `Command` is what executes
//! bytes-in/bytes-out with a Unix exit code.

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
#[cfg(any(test, feature = "test-support"))]
pub use policy::{AlwaysAllowGate, DenyAllGate};
pub use policy::{
    CONFIRM_ROUTER, CONFIRM_TIMEOUT, ConfirmRouter, ConfirmationGate, ConfirmationRequest,
    IpcConfirmationGate, NoPendingConfirm, SandboxError, SandboxInfo, SandboxRequest,
    inherit_confirm_router, probe_sandbox,
};
pub use run::RunTool;
pub use vision::VisionGate;

use async_trait::async_trait;
use serde_json::{Value, json};

/// Prefix every MCP-adapted tool's `name()` carries
/// (`mcp__<server>__<tool>`), so a registry can be partitioned into
/// native and MCP tools.
pub const MCP_TOOL_NAME_PREFIX: &str = "mcp__";

/// Why [`Tool::invoke`] produced no result. A failure the model can
/// recover from by running something else belongs in the returned
/// result envelope instead.
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    /// The model's arguments violate the tool's schema or constraints.
    /// The message names the offending argument.
    #[error("{0}")]
    InvalidArgs(String),
    /// A memory, conversation, or semantic store call failed.
    #[error(transparent)]
    Store(#[from] assistd_memory::MemoryError),
}

/// A single tool the LLM can invoke.
#[async_trait]
pub trait Tool: Send + Sync + 'static {
    /// Machine-readable identifier used by the LLM to call this tool.
    fn name(&self) -> &str;

    /// Human-readable description the LLM sees when deciding whether to
    /// call the tool.
    fn description(&self) -> &str;

    /// JSON Schema describing the `arguments` object the LLM must pass to
    /// [`Tool::invoke`]. Used to build the OpenAI-compatible `tools` array.
    fn parameters_schema(&self) -> Value;

    /// Execute the tool with JSON-shaped arguments and return a
    /// JSON-shaped result.
    async fn invoke(&self, args: Value) -> Result<Value, ToolError>;
}

/// Lookup table for tools registered with the daemon.
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

    /// Render the registry as an OpenAI chat-completions `tools` array.
    /// Each entry is a `{"type": "function", "function": {...}}` object
    /// with `strict: true`, guaranteeing the model's arguments conform
    /// to `parameters_schema()`.
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

/// Crate version.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
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
