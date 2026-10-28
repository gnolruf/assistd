use std::borrow::Cow;

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
    fn description(&self) -> Cow<'_, str> {
        "".into()
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
