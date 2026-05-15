//! Render MCP-adapted tools as a Markdown section for the system prompt.
//!
//! Mirrors [`assistd_tools::prompt::format_tool_listing`] but emits the
//! MCP-specific header so the model can distinguish locally-shipped
//! tools from MCP-sourced ones. The daemon appends this block at
//! startup after partitioning the unified [`assistd_tools::ToolRegistry`]
//! by the [`assistd_tools::MCP_TOOL_NAME_PREFIX`] convention.

use assistd_tools::Tool;

/// Render `tools` as a Markdown section suitable for appending to the
/// chat system prompt. Returns an empty string when `tools` is empty so
/// the caller can unconditionally append; the daemon's helper drops the
/// section entirely when no MCP servers exposed any tools.
pub fn format_mcp_listing(tools: &[&dyn Tool]) -> String {
    if tools.is_empty() {
        return String::new();
    }
    let mut s = String::from("## Available MCP tools\n");
    for t in tools {
        push_bullet(&mut s, t.name(), t.description());
    }
    s.trim_end().to_string()
}

fn push_bullet(out: &mut String, name: &str, desc: &str) {
    out.push_str("- `");
    out.push_str(name);
    out.push_str("`: ");
    let mut first = true;
    for line in desc.lines() {
        if first {
            out.push_str(line);
            first = false;
        } else {
            out.push_str("\n  ");
            out.push_str(line);
        }
    }
    out.push('\n');
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{McpClient, ToolResult};
    use crate::{ToolSchema, adapt_client_as_tools};
    use anyhow::Result;
    use async_trait::async_trait;
    use serde_json::{Value, json};
    use std::sync::Arc;

    struct FakeMcpClient {
        schemas: Vec<ToolSchema>,
    }

    #[async_trait]
    impl McpClient for FakeMcpClient {
        async fn list_tools(&self) -> Result<Vec<ToolSchema>> {
            Ok(self.schemas.clone())
        }
        async fn invoke(&self, _name: &str, _arguments: Value) -> Result<ToolResult> {
            Ok(ToolResult::Text(String::new()))
        }
    }

    #[test]
    fn format_mcp_listing_empty_returns_empty_string() {
        assert_eq!(format_mcp_listing(&[]), "");
    }

    #[test]
    fn format_mcp_listing_uses_mcp_header() {
        struct Fake;
        #[async_trait]
        impl Tool for Fake {
            fn name(&self) -> &str {
                "mcp__server__tool"
            }
            fn description(&self) -> &str {
                "irrelevant"
            }
            fn parameters_schema(&self) -> Value {
                json!({"type": "object"})
            }
            async fn invoke(&self, _args: Value) -> Result<Value> {
                Ok(Value::Null)
            }
        }
        let f = Fake;
        let out = format_mcp_listing(&[&f as &dyn Tool]);
        assert!(out.starts_with("## Available MCP tools\n"), "got: {out}");
    }

    #[tokio::test]
    async fn format_mcp_listing_renders_prefixed_names() {
        let client: Arc<dyn McpClient> = Arc::new(FakeMcpClient {
            schemas: vec![ToolSchema {
                name: "search".into(),
                description: "search the web".into(),
                input_schema: json!({"type": "object"}),
            }],
        });
        let tools = adapt_client_as_tools(client, "mcp__web").await.unwrap();
        let refs: Vec<&dyn Tool> = tools.iter().map(|b| b.as_ref()).collect();
        let out = format_mcp_listing(&refs);
        assert!(
            out.contains("- `mcp__web__search`: search the web"),
            "{out}"
        );
    }

    #[test]
    fn format_mcp_listing_indents_multiline_descriptions() {
        struct Fake;
        #[async_trait]
        impl Tool for Fake {
            fn name(&self) -> &str {
                "mcp__foo__bar"
            }
            fn description(&self) -> &str {
                "Line one.\nLine two."
            }
            fn parameters_schema(&self) -> Value {
                json!({"type": "object"})
            }
            async fn invoke(&self, _args: Value) -> Result<Value> {
                Ok(Value::Null)
            }
        }
        let f = Fake;
        let out = format_mcp_listing(&[&f as &dyn Tool]);
        assert!(
            out.contains("- `mcp__foo__bar`: Line one.\n  Line two."),
            "got: {out}"
        );
    }

    #[test]
    fn format_mcp_listing_preserves_order() {
        struct N(&'static str);
        #[async_trait]
        impl Tool for N {
            fn name(&self) -> &str {
                self.0
            }
            fn description(&self) -> &str {
                "x"
            }
            fn parameters_schema(&self) -> Value {
                json!({"type": "object"})
            }
            async fn invoke(&self, _args: Value) -> Result<Value> {
                Ok(Value::Null)
            }
        }
        let a = N("mcp__a__one");
        let b = N("mcp__b__two");
        let c = N("mcp__c__three");
        let tools: Vec<&dyn Tool> = vec![&a, &b, &c];
        let out = format_mcp_listing(&tools);
        let i_a = out.find("mcp__a__one").unwrap();
        let i_b = out.find("mcp__b__two").unwrap();
        let i_c = out.find("mcp__c__three").unwrap();
        assert!(i_a < i_b && i_b < i_c);
    }
}
