//! Render the daemon's tool catalog as a Markdown section for the
//! system prompt. The output is concatenated onto the base prompt at
//! daemon startup so a newly registered tool is automatically advertised
//! to the model without anyone editing `DEFAULT_SYSTEM_PROMPT`.

use crate::Tool;

/// Render `tools` as a Markdown section suitable for appending to the
/// chat system prompt. Returns an empty string when `tools` is empty so
/// the caller can unconditionally append without inserting blank
/// sections.
///
/// Each bullet uses the tool's full `description()`; embedded newlines
/// are indented two spaces so the Markdown bullet continuation renders
/// cleanly when the model is shown the prompt.
pub fn format_tool_listing(tools: &[&dyn Tool]) -> String {
    if tools.is_empty() {
        return String::new();
    }
    let mut s = String::from("## Available native tools\n");
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
    use anyhow::Result;
    use async_trait::async_trait;
    use serde_json::{Value, json};

    struct Fake {
        name: &'static str,
        desc: &'static str,
    }

    #[async_trait]
    impl Tool for Fake {
        fn name(&self) -> &str {
            self.name
        }
        fn description(&self) -> &str {
            self.desc
        }
        fn parameters_schema(&self) -> Value {
            json!({"type": "object"})
        }
        async fn invoke(&self, _args: Value) -> Result<Value> {
            Ok(Value::Null)
        }
    }

    #[test]
    fn format_tool_listing_empty_returns_empty_string() {
        assert_eq!(format_tool_listing(&[]), "");
    }

    #[test]
    fn format_tool_listing_renders_full_description() {
        let f = Fake {
            name: "foo",
            desc: "Do foo. Use it for X.",
        };
        let tools: Vec<&dyn Tool> = vec![&f];
        let out = format_tool_listing(&tools);
        assert_eq!(
            out,
            "## Available native tools\n- `foo`: Do foo. Use it for X."
        );
    }

    #[test]
    fn format_tool_listing_indents_multiline_descriptions() {
        // A description containing newlines must wrap continuation lines
        // under the bullet so Markdown rendering and the model both
        // attach them to the right tool.
        let f = Fake {
            name: "multi",
            desc: "Line one.\nLine two.",
        };
        let tools: Vec<&dyn Tool> = vec![&f];
        let out = format_tool_listing(&tools);
        assert_eq!(
            out,
            "## Available native tools\n- `multi`: Line one.\n  Line two."
        );
    }

    #[test]
    fn format_tool_listing_preserves_order() {
        let a = Fake {
            name: "a",
            desc: "first.",
        };
        let b = Fake {
            name: "b",
            desc: "second.",
        };
        let c = Fake {
            name: "c",
            desc: "third.",
        };
        let tools: Vec<&dyn Tool> = vec![&a, &b, &c];
        let out = format_tool_listing(&tools);
        // Bullets appear in registration order.
        let a_idx = out.find("`a`").expect("a bullet");
        let b_idx = out.find("`b`").expect("b bullet");
        let c_idx = out.find("`c`").expect("c bullet");
        assert!(a_idx < b_idx);
        assert!(b_idx < c_idx);
    }

    #[test]
    fn format_tool_listing_includes_h2_header() {
        let f = Fake {
            name: "anything",
            desc: "anything",
        };
        let tools: Vec<&dyn Tool> = vec![&f];
        let out = format_tool_listing(&tools);
        assert!(out.starts_with("## Available native tools\n"), "got: {out}");
    }
}
