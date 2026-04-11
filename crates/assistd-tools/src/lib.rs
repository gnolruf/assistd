//! Tool-use subsystem: the trait every invokable tool implements, plus
//! the registry the LLM looks up tool calls in.
//!
//! Milestone 3 ("tool use") will register real tools here (shell, files,
//! HTTP, etc.) and run them in a sandbox. Milestone 1 ships no tools —
//! the registry exists so the call-site wiring is in place.

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;

/// A single tool the LLM can invoke.
#[async_trait]
pub trait Tool: Send + Sync + 'static {
    /// Machine-readable identifier used by the LLM to call this tool.
    fn name(&self) -> &str;

    /// Human-readable description the LLM sees when deciding whether to
    /// call the tool.
    fn description(&self) -> &str;

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
    fn version_is_not_empty() {
        assert!(!version().is_empty());
    }
}
