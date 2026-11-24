//! The hash of what a server declares a tool to be, which an approval of the tool is pinned to.

use std::fmt::Write;

use assistd_ipc::DefinitionHash;
use rmcp::model::Tool;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

/// SHA-256 of `tool`'s name, description, input schema, output schema and annotations as
/// JSON with every object's keys sorted.
pub(crate) fn definition_hash(tool: &Tool) -> DefinitionHash {
    let declared = json!({
        "name": tool.name,
        "description": tool.description,
        "inputSchema": tool.input_schema,
        "outputSchema": tool.output_schema,
        "annotations": tool.annotations,
    });
    let mut canonical = String::new();
    write_canonical(&declared, &mut canonical);
    let hex = Sha256::digest(canonical.as_bytes())
        .iter()
        .fold(String::new(), |mut hex, byte| {
            let _ = write!(hex, "{byte:02x}");
            hex
        });
    DefinitionHash(hex)
}

/// Append `value` as compact JSON, each object's keys in byte order.
fn write_canonical(value: &Value, canonical: &mut String) {
    match value {
        Value::Object(fields) => {
            let mut fields: Vec<(&String, &Value)> = fields.iter().collect();
            fields.sort_by_key(|(key, _)| *key);
            canonical.push('{');
            for (position, (key, field)) in fields.into_iter().enumerate() {
                if position > 0 {
                    canonical.push(',');
                }
                canonical.push_str(&Value::from(key.as_str()).to_string());
                canonical.push(':');
                write_canonical(field, canonical);
            }
            canonical.push('}');
        }
        Value::Array(items) => {
            canonical.push('[');
            for (position, item) in items.iter().enumerate() {
                if position > 0 {
                    canonical.push(',');
                }
                write_canonical(item, canonical);
            }
            canonical.push(']');
        }
        scalar => canonical.push_str(&scalar.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool(definition: Value) -> Tool {
        serde_json::from_value(definition).expect("tool definition")
    }

    #[test]
    fn hash_follows_what_the_tool_declares_and_not_how_it_is_written() {
        let listed = tool(json!({
            "name": "delete",
            "description": "deletes a file",
            "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}},
            "annotations": {"readOnlyHint": false}
        }));
        let reordered = tool(json!({
            "annotations": {"readOnlyHint": false},
            "inputSchema": {"properties": {"path": {"type": "string"}}, "type": "object"},
            "title": "Delete",
            "description": "deletes a file",
            "name": "delete"
        }));
        let hash = definition_hash(&listed);
        assert_eq!(hash, definition_hash(&reordered));
        assert_eq!(hash.0.len(), 64);

        let redefinitions = [
            json!({"description": "deletes a directory tree"}),
            json!({"inputSchema": {"type": "object", "properties": {"glob": {"type": "string"}}}}),
            json!({"outputSchema": {"type": "object"}}),
            json!({"annotations": {"readOnlyHint": true}}),
        ];
        for redefinition in redefinitions {
            let mut redefined = serde_json::to_value(&listed).expect("tool as JSON");
            redefined
                .as_object_mut()
                .expect("tool is an object")
                .extend(redefinition.as_object().expect("object").clone());
            assert_ne!(
                hash,
                definition_hash(&tool(redefined.clone())),
                "{redefined}"
            );
        }
    }
}
