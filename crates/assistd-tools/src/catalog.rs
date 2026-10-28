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
mod tests;
