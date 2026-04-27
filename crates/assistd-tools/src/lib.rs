#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::print_stdout,
        clippy::print_stderr
    )
)]

//! Tool-use subsystem: the trait every invokable tool implements, plus
//! the registry the LLM looks up tool calls in.
//!
//! The daemon exposes a single LLM-facing tool — `run` — whose argument is
//! a Unix-shell-style command line. `run` parses the line into a
//! [`chain::Chain`] AST and dispatches each stage through a
//! [`CommandRegistry`] of internal Rust handlers. Two-tier split: `Tool`
//! is what the LLM sees (JSON-in/JSON-out); `Command` is what executes
//! bytes-in/bytes-out with a Unix exit code.

pub mod attachment;
pub mod chain;
pub mod command;
pub mod commands;
pub mod policy;
pub mod presentation;
pub mod run;
pub mod vision;

pub use attachment::{LoadImageError, load_image_attachment};
pub use chain::{Chain, ParseError, execute, parse_chain};
pub use command::{Attachment, Command, CommandInput, CommandOutput, CommandRegistry};
pub use policy::{
    AlwaysAllowGate, ConfirmationGate, ConfirmationRequest, DenyAllGate, ResolvedSandboxMode,
    SandboxInfo, SandboxRequest, matches_denylist, matches_destructive, probe_sandbox,
};
pub use presentation::{PresentResult, present};
pub use run::RunTool;
pub use vision::VisionGate;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::{Value, json};

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
    async fn invoke(&self, args: Value) -> Result<Value>;
}

/// Lookup table for tools registered with the daemon.
#[derive(Default)]
pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<T: Tool>(&mut self, tool: T) {
        self.tools.push(Box::new(tool));
    }

    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools
            .iter()
            .find(|t| t.name() == name)
            .map(|t| t.as_ref())
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tools.iter().map(|t| t.name())
    }

    /// Render the registry as an OpenAI chat-completions `tools` array.
    /// Each entry is a `{"type": "function", "function": {...}}` object
    /// with `strict: true` — guaranteeing the model's arguments conform
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
        async fn invoke(&self, _args: Value) -> Result<Value> {
            Ok(Value::Null)
        }
    }

    #[test]
    fn registry_starts_empty() {
        let reg = ToolRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
        assert!(reg.get("noop").is_none());
    }

    #[tokio::test]
    async fn registered_tool_is_findable_and_invokable() {
        let mut reg = ToolRegistry::new();
        reg.register(Noop);
        assert_eq!(reg.len(), 1);
        let tool = reg.get("noop").expect("tool registered");
        let result = tool.invoke(Value::Null).await.unwrap();
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn openai_schemas_wraps_each_tool() {
        let mut reg = ToolRegistry::new();
        reg.register(Noop);
        let schemas = reg.openai_schemas();
        assert_eq!(schemas.len(), 1);
        let entry = &schemas[0];
        assert_eq!(entry["type"], "function");
        assert_eq!(entry["function"]["name"], "noop");
        assert_eq!(entry["function"]["strict"], true);
        assert_eq!(entry["function"]["parameters"]["type"], "object");
    }

    #[test]
    fn version_is_not_empty() {
        assert!(!version().is_empty());
    }
}
