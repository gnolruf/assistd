//! Which of a server's advertised tools may enter the registry, and under
//! what name.

use std::collections::HashSet;

use assistd_tools::MCP_TOOL_NAME_PREFIX;
use rmcp::model::Tool;
use thiserror::Error;
use tracing::warn;

/// Longest registry name a tool may have, prefix included.
const MAX_TOOL_NAME_LEN: usize = 64;

/// Largest description, in bytes, sent to the model with every request.
const MAX_DESCRIPTION_BYTES: usize = 4 * 1024;

/// Largest input schema, in bytes of compact JSON.
const MAX_SCHEMA_BYTES: usize = 16 * 1024;

/// Most tools admitted from one server.
const MAX_TOOLS_PER_SERVER: usize = 128;

/// Largest combined description-plus-schema size, in bytes, admitted from one server.
const MAX_SERVER_CATALOG_BYTES: usize = 128 * 1024;

/// Why an advertised tool was left out of the registry.
#[derive(Debug, Error)]
enum Rejection {
    #[error("name `{0}` is empty or not up to {MAX_TOOL_NAME_LEN} characters of [A-Za-z0-9_-]")]
    InvalidName(String),
    #[error("description is {0} bytes, over the {MAX_DESCRIPTION_BYTES}-byte cap")]
    DescriptionTooLong(usize),
    #[error("input schema is {0} bytes, over the {MAX_SCHEMA_BYTES}-byte cap")]
    SchemaTooLarge(usize),
    #[error("another tool from this server already has the name `{0}`")]
    Duplicate(String),
}

/// Pair each acceptable tool from `server_name` with its registry name
/// `mcp__<server_name>__<tool>`, skipping with a warning any whose name is
/// invalid or taken, or whose description or schema is over its cap. Tools are
/// admitted in listed order until the server's count or byte budget runs out.
pub(crate) fn admit_tools(server_name: &str, tools: Vec<Tool>) -> Vec<(String, Tool)> {
    let mut taken = HashSet::new();
    let mut admitted = Vec::new();
    let mut catalog_bytes = 0;
    let listed = tools.len();
    for (position, tool) in tools.into_iter().enumerate() {
        let registry_name = format!("{MCP_TOOL_NAME_PREFIX}{server_name}__{}", tool.name);
        let size = match check_tool(&registry_name, &tool, &mut taken) {
            Ok(size) => size,
            Err(rejection) => {
                warn_skipped(server_name, &tool, &rejection);
                continue;
            }
        };
        if admitted.len() == MAX_TOOLS_PER_SERVER || catalog_bytes + size > MAX_SERVER_CATALOG_BYTES
        {
            warn!(
                target: "assistd::mcp",
                server = %server_name,
                skipped = listed - position,
                "MCP server is over its {MAX_TOOLS_PER_SERVER}-tool or \
                 {MAX_SERVER_CATALOG_BYTES}-byte catalog cap; skipping its remaining tools",
            );
            break;
        }
        catalog_bytes += size;
        admitted.push((registry_name, tool));
    }
    admitted
}

fn warn_skipped(server_name: &str, tool: &Tool, rejection: &Rejection) {
    warn!(
        target: "assistd::mcp",
        server = %server_name,
        tool = %tool.name.escape_debug(),
        reason = %rejection,
        "skipping MCP tool",
    );
}

/// Accept `tool` under `registry_name`, recording the name in `taken`, and
/// return its description-plus-schema size in bytes.
fn check_tool(
    registry_name: &str,
    tool: &Tool,
    taken: &mut HashSet<String>,
) -> Result<usize, Rejection> {
    if tool.name.is_empty() || !is_valid_tool_name(registry_name) {
        return Err(Rejection::InvalidName(
            registry_name.escape_debug().to_string(),
        ));
    }
    let description_len = tool.description.as_deref().map_or(0, str::len);
    if description_len > MAX_DESCRIPTION_BYTES {
        return Err(Rejection::DescriptionTooLong(description_len));
    }
    let schema_len = tool.schema_as_json_value().to_string().len();
    if schema_len > MAX_SCHEMA_BYTES {
        return Err(Rejection::SchemaTooLarge(schema_len));
    }
    if !taken.insert(registry_name.to_string()) {
        return Err(Rejection::Duplicate(registry_name.to_string()));
    }
    Ok(description_len + schema_len)
}

fn is_valid_tool_name(name: &str) -> bool {
    (1..=MAX_TOOL_NAME_LEN).contains(&name.len())
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
}
