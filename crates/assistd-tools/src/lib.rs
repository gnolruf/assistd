//! Tool-use subsystem: [`Tool`]s the model calls with JSON, chiefly
//! [`RunTool`], which runs shell-style chains of byte-oriented [`Command`]s.

use std::borrow::Cow;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

pub mod attachment;
mod catalog;
pub mod chain;
pub mod command;
pub mod commands;
mod exec;
pub mod memory;
pub mod memory_tools;
pub mod policy;
pub mod presentation;
pub mod run;
mod tool_output;
pub mod vision;

pub use assistd_ipc::ImageMime;
pub use attachment::{LoadImageError, load_image_attachment};
pub use catalog::ToolCatalog;
pub use command::{Attachment, Command, CommandInput, CommandOutput, CommandRegistry};
pub use memory::{DEFAULT_SEARCH_LIMIT, MemoryOps};
pub use memory_tools::{RememberTool, ReminisceTool};
pub use policy::{
    APPROVALS_FILE, APPROVED_APPS_FILE, APPROVED_HOSTS_FILE, APPROVED_MCP_TOOLS_FILE, Allowlist,
    AllowlistError, Approval, ApprovalGate, Approvals, CONFIRM_ROUTER, CONFIRM_TIMEOUT,
    ConfirmRouter, ConfirmationGate, ConfirmationRequest, DestructivePattern, IpcConfirmationGate,
    NoPendingConfirm, Protected, Risk, SandboxError, SandboxInfo, SandboxRequest, SearchPath,
    SharedDirs, ToolSandbox, ToolsDisabled, inherit_confirm_router, probe_sandbox,
};
#[cfg(any(test, feature = "test-support"))]
pub use policy::{AlwaysAllowGate, DenyAllGate};
pub use presentation::{OutputCap, PresentSpec};
pub use run::RunTool;
pub use tool_output::{RawStreams, ToolOutput};
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
    /// A memory, transcript, or semantic store call failed.
    #[error(transparent)]
    Store(#[from] assistd_memory::MemoryError),
}

/// A single tool the LLM can invoke.
#[async_trait]
pub trait Tool: fmt::Debug + Send + Sync + 'static {
    /// Identifier the model calls this tool by.
    fn name(&self) -> &str;

    /// Description the model sees when deciding whether to call the tool,
    /// read each time the tool list is sent.
    fn description(&self) -> Cow<'_, str>;

    /// JSON Schema for the `arguments` object [`Tool::invoke`] accepts.
    fn parameters_schema(&self) -> Value;

    /// Execute the tool with JSON arguments.
    async fn invoke(&self, args: Value) -> Result<ToolOutput, ToolError>;
}

/// Lookup table of registered tools; cloning shares the tools. Every
/// tool's [`ToolOutput::output`] is held to the registry's [`OutputCap`].
#[derive(Debug, Clone)]
pub struct ToolRegistry {
    tools: Vec<Arc<dyn Tool>>,
    output_cap: Arc<OutputCap>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::with_output_cap(Arc::new(OutputCap::new(PresentSpec::default())))
    }
}

impl ToolRegistry {
    /// An empty registry capped to the default `tools.output` limits.
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty registry whose tools' results are cut to `output_cap`.
    pub fn with_output_cap(output_cap: Arc<OutputCap>) -> Self {
        Self {
            tools: Vec::new(),
            output_cap,
        }
    }

    /// Register a tool by value.
    pub fn register<T: Tool>(&mut self, tool: T) {
        self.register_boxed(Box::new(tool));
    }

    /// Register an already-boxed tool.
    pub fn register_boxed(&mut self, tool: Box<dyn Tool>) {
        self.tools.push(Arc::new(CappedTool {
            inner: tool,
            output_cap: Arc::clone(&self.output_cap),
        }));
    }

    /// Look up a registered tool by its `name()`.
    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools
            .iter()
            .find(|t| t.name() == name)
            .map(AsRef::as_ref)
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

/// A registered tool whose `output` is cut to `output_cap`, spilling under
/// the tool's name and flagging the result `truncated`.
#[derive(Debug)]
struct CappedTool {
    inner: Box<dyn Tool>,
    output_cap: Arc<OutputCap>,
}

#[async_trait]
impl Tool for CappedTool {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn description(&self) -> Cow<'_, str> {
        self.inner.description()
    }

    fn parameters_schema(&self) -> Value {
        self.inner.parameters_schema()
    }

    async fn invoke(&self, args: Value) -> Result<ToolOutput, ToolError> {
        let mut result = self.inner.invoke(args).await?;
        let cut = self
            .output_cap
            .cap(self.inner.name(), std::mem::take(&mut result.output));
        result.output = cut.text;
        result.truncated |= cut.truncated;
        if cut.overflow_file.is_some() {
            result.overflow_file = cut.overflow_file;
        }
        Ok(result)
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
    use std::time::Duration;

    use super::*;

    #[derive(Debug)]
    struct Noop;

    #[async_trait]
    impl Tool for Noop {
        fn name(&self) -> &'static str {
            "noop"
        }
        fn description(&self) -> Cow<'_, str> {
            "does nothing".into()
        }
        fn parameters_schema(&self) -> Value {
            json!({"type": "object", "properties": {}, "additionalProperties": false})
        }
        async fn invoke(&self, _args: Value) -> Result<ToolOutput, ToolError> {
            Ok(ToolOutput::default())
        }
    }

    #[derive(Debug)]
    struct Verbose;

    #[async_trait]
    impl Tool for Verbose {
        fn name(&self) -> &'static str {
            "verbose"
        }
        fn description(&self) -> Cow<'_, str> {
            "prints a lot".into()
        }
        fn parameters_schema(&self) -> Value {
            json!({})
        }
        async fn invoke(&self, _args: Value) -> Result<ToolOutput, ToolError> {
            Ok(ToolOutput::text("x".repeat(10_000), Duration::ZERO))
        }
    }

    #[tokio::test]
    async fn every_registered_tool_is_held_to_the_output_cap() {
        let dir = tempfile::tempdir().unwrap();
        let spec = PresentSpec {
            overflow_dir: dir.path().to_path_buf(),
            ..PresentSpec::default()
        };
        let max_bytes = spec.max_bytes;
        let mut reg = ToolRegistry::with_output_cap(Arc::new(OutputCap::new(spec)));
        reg.register(Verbose);

        let result = reg.get("verbose").unwrap().invoke(json!({})).await.unwrap();

        assert!(result.output.len() <= max_bytes);
        assert!(result.truncated);
        let spilled = dir.path().join("verbose-1.txt");
        assert_eq!(result.overflow_file.as_deref(), Some(spilled.as_path()));
        assert_eq!(std::fs::read(spilled).unwrap().len(), 10_000);
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
