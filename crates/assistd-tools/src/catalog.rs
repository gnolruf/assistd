//! The tools offered to the model, which grow as tools from servers that
//! start in the background are added.

use std::sync::Arc;

use parking_lot::RwLock;

use crate::{Tool, ToolRegistry};

/// The current [`ToolRegistry`]. Each turn takes a snapshot, so tools added
/// mid-turn are first offered on the next turn.
#[derive(Debug)]
pub struct ToolCatalog {
    current: RwLock<Arc<ToolRegistry>>,
}

impl ToolCatalog {
    pub fn new(registry: Arc<ToolRegistry>) -> Self {
        Self {
            current: RwLock::new(registry),
        }
    }

    /// The registry as it stands now.
    pub fn snapshot(&self) -> Arc<ToolRegistry> {
        Arc::clone(&self.current.read())
    }

    /// Offer `tools` from the next snapshot on.
    pub fn extend(&self, tools: Vec<Box<dyn Tool>>) {
        let mut current = self.current.write();
        let mut next = ToolRegistry::clone(&current);
        for tool in tools {
            next.register_boxed(tool);
        }
        *current = Arc::new(next);
    }
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;
    use serde_json::{Value, json};

    use super::*;
    use crate::ToolError;

    #[derive(Debug)]
    struct Named(&'static str);

    #[async_trait]
    impl Tool for Named {
        fn name(&self) -> &str {
            self.0
        }
        fn description(&self) -> &'static str {
            ""
        }
        fn parameters_schema(&self) -> Value {
            json!({})
        }
        async fn invoke(&self, _args: Value) -> Result<Value, ToolError> {
            Ok(Value::Null)
        }
    }

    #[test]
    fn extending_leaves_earlier_snapshots_unchanged() {
        let mut base = ToolRegistry::new();
        base.register(Named("run"));
        let catalog = ToolCatalog::new(Arc::new(base));
        let before = catalog.snapshot();

        catalog.extend(vec![Box::new(Named("mcp__fs__read"))]);

        assert_eq!(before.names().collect::<Vec<_>>(), ["run"]);
        assert_eq!(
            catalog.snapshot().names().collect::<Vec<_>>(),
            ["run", "mcp__fs__read"]
        );
    }
}
