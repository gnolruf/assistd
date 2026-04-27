#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::print_stdout,
        clippy::print_stderr
    )
)]

//! MCP (Model Context Protocol) **client** trait sketch.
//!
//! Milestone 5 will land a concrete `stdio` JSON-RPC implementation
//! plus per-server lifecycle (spawn, ready-probe, restart-on-crash).
//! This crate exists ahead of that work to settle the trait shape and
//! the bridge into the existing [`assistd_tools::Tool`] registry, so
//! we're not retrofitting `AppState` and the agent loop mid-feature.
//!
//! # Architecture
//!
//! From the daemon's perspective, an MCP server is a *tool source*:
//! - On startup, the daemon connects to each configured MCP server.
//! - `list_tools` returns each server's tool catalog.
//! - For each entry, the daemon wraps it in an [`McpToolAdapter`] and
//!   registers it with the existing `ToolRegistry` — same registry
//!   the LLM already iterates for OpenAI-style schemas.
//! - When the LLM calls a wrapped tool, the adapter forwards the
//!   call to [`McpClient::invoke`] on the originating server.
//!
//! # Why a separate trait, not a `Tool` impl?
//!
//! A single MCP server typically exposes many tools, with distinct
//! schemas and dynamic discovery. Modeling each tool as its own
//! `Tool` impl would either require generating a struct per tool at
//! compile time (impossible — schemas come from the wire) or a giant
//! match. Instead we keep `McpClient` as a connection-level handle,
//! and use the [`McpToolAdapter`] type to turn each
//! `(client, tool_name, schema)` tuple into a `Tool` registry entry.
//!
//! # Image-bearing tool results
//!
//! [`ToolResult::Image`] feeds straight into the existing
//! `assistd_tools::Attachment::Image` so a remote screenshot tool
//! lands in the same vision-input pipeline as the local
//! `screenshot` command. This is the reason the IPC schema work in
//! `assistd-ipc` carries `ImageAttachment` over the wire — keeping
//! local and remote attachments shape-compatible end-to-end.

use std::sync::Arc;

use anyhow::Result;
use assistd_tools::{Attachment, Tool};
use async_trait::async_trait;
use serde_json::{Value, json};

/// One tool exposed by an MCP server.
#[derive(Debug, Clone)]
pub struct ToolSchema {
    /// Tool name as it will appear in the LLM's `tools` list. The
    /// daemon may decorate this (e.g. `mcp:server-name:tool-name`)
    /// before registration to avoid collisions across servers.
    pub name: String,
    /// Human-readable description the LLM sees.
    pub description: String,
    /// JSON Schema for the `arguments` object the LLM will pass.
    /// Forwarded verbatim to the underlying server.
    pub input_schema: Value,
}

/// Result of an MCP tool invocation. Mirrors the three shapes
/// llama.cpp / OpenAI tool calls already flow through:
/// strings, structured JSON, and images.
#[derive(Debug, Clone)]
pub enum ToolResult {
    Text(String),
    Image { mime: String, bytes: Vec<u8> },
    Json(Value),
}

/// A connection to a single MCP server.
#[async_trait]
pub trait McpClient: Send + Sync + 'static {
    /// List all tools the server currently exposes. Called at
    /// startup; concrete implementations may refresh on demand
    /// (Milestone 5) but must be safe to call concurrently.
    async fn list_tools(&self) -> Result<Vec<ToolSchema>>;

    /// Invoke `name` with `arguments`. The daemon strips any
    /// registry-side prefix (e.g. `mcp:server-name:`) before
    /// forwarding, so `name` is the server-native identifier.
    async fn invoke(&self, name: &str, arguments: Value) -> Result<ToolResult>;
}

/// Wraps a single (client, tool) pair as a [`Tool`] registry entry.
/// The daemon's tool-registry code is unchanged: it iterates
/// `ToolRegistry` and the adapter takes care of the dispatch hop.
pub struct McpToolAdapter {
    client: Arc<dyn McpClient>,
    schema: ToolSchema,
    /// The name as exposed to the LLM. May be a prefixed form
    /// (`mcp:foo:bar`) when the daemon namespaces multiple servers.
    /// We carry the *server-native* name in `schema.name` and the
    /// *registry* name here so `invoke` can strip the prefix.
    registry_name: String,
}

impl McpToolAdapter {
    pub fn new(client: Arc<dyn McpClient>, schema: ToolSchema, registry_name: String) -> Self {
        Self {
            client,
            schema,
            registry_name,
        }
    }
}

#[async_trait]
impl Tool for McpToolAdapter {
    fn name(&self) -> &str {
        &self.registry_name
    }

    fn description(&self) -> &str {
        &self.schema.description
    }

    fn parameters_schema(&self) -> Value {
        self.schema.input_schema.clone()
    }

    async fn invoke(&self, args: Value) -> Result<Value> {
        let result = self.client.invoke(&self.schema.name, args).await?;
        Ok(tool_result_to_json(result))
    }
}

/// Map a [`ToolResult`] into the JSON shape `RunTool` already emits
/// to the LLM. `Image` results carry their bytes alongside the JSON
/// so the tool dispatch site can lift them into [`Attachment::Image`]
/// for the next turn — same as `see` / `screenshot`.
fn tool_result_to_json(r: ToolResult) -> Value {
    match r {
        ToolResult::Text(s) => json!({ "type": "text", "text": s }),
        ToolResult::Json(v) => json!({ "type": "json", "value": v }),
        ToolResult::Image { mime, bytes } => json!({
            "type": "image",
            "mime": mime,
            // Raw bytes inlined as base64 keeps the JSON-only contract
            // of `Tool::invoke`. The dispatch site (Milestone 5) decodes
            // and lifts to Attachment::Image before the next LLM turn.
            "bytes_base64": base64_encode(&bytes),
        }),
    }
}

/// Lift a JSON-encoded image (as produced by [`tool_result_to_json`])
/// back into an [`Attachment::Image`] for the LLM's next turn. Returns
/// `None` if the value isn't an `image`-typed result. Will be called
/// from the agent loop when Milestone 5 wires MCP into the dispatch.
pub fn image_attachment_from_tool_result(v: &Value) -> Option<Attachment> {
    if v.get("type").and_then(Value::as_str) != Some("image") {
        return None;
    }
    let mime = v.get("mime").and_then(Value::as_str)?;
    let b64 = v.get("bytes_base64").and_then(Value::as_str)?;
    let bytes = base64_decode(b64).ok()?;
    Some(Attachment::Image {
        mime: mime.to_string(),
        bytes,
    })
}

/// Build [`Tool`] registry entries for every tool an MCP client
/// exposes. The daemon will call this once per configured server at
/// startup, prefix the names with the server label to avoid
/// collisions, and register each entry with the existing
/// `ToolRegistry`. No concrete client implementation lives here yet —
/// Milestone 5 lands one and the wiring just becomes a `for server in
/// config.mcp.servers { ... }` loop in `build_tools`.
pub async fn adapt_client_as_tools(
    client: Arc<dyn McpClient>,
    name_prefix: &str,
) -> Result<Vec<Box<dyn Tool>>> {
    let schemas = client.list_tools().await?;
    let mut out: Vec<Box<dyn Tool>> = Vec::with_capacity(schemas.len());
    for schema in schemas {
        let registry_name = if name_prefix.is_empty() {
            schema.name.clone()
        } else {
            format!("{name_prefix}:{}", schema.name)
        };
        out.push(Box::new(McpToolAdapter::new(
            client.clone(),
            schema,
            registry_name,
        )));
    }
    Ok(out)
}

fn base64_encode(bytes: &[u8]) -> String {
    use base64_dep::Engine;
    base64_dep::engine::general_purpose::STANDARD.encode(bytes)
}

fn base64_decode(s: &str) -> Result<Vec<u8>, base64_dep::DecodeError> {
    use base64_dep::Engine;
    base64_dep::engine::general_purpose::STANDARD.decode(s)
}

// Internal alias to keep base64 a single import path. assistd-tools
// already depends on base64 transitively via assistd-ipc; declaring
// our own dep would be redundant. We re-export under a private name
// so the public surface of this crate stays minimal.
mod base64_dep {
    pub use base64::*;
}

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trait-only fake server: returns a static tool list and echoes
    /// arguments back as a Text result. Lets us exercise the adapter
    /// without standing up real I/O.
    struct FakeMcpClient {
        schemas: Vec<ToolSchema>,
    }

    #[async_trait]
    impl McpClient for FakeMcpClient {
        async fn list_tools(&self) -> Result<Vec<ToolSchema>> {
            Ok(self.schemas.clone())
        }

        async fn invoke(&self, name: &str, arguments: Value) -> Result<ToolResult> {
            Ok(ToolResult::Text(format!("called {name} with {arguments}")))
        }
    }

    fn one_tool_client() -> Arc<dyn McpClient> {
        Arc::new(FakeMcpClient {
            schemas: vec![ToolSchema {
                name: "search".into(),
                description: "search the web".into(),
                input_schema: json!({"type": "object", "properties": {}}),
            }],
        })
    }

    #[tokio::test]
    async fn adapter_forwards_tool_metadata() {
        let client = one_tool_client();
        let tools = adapt_client_as_tools(client, "mcp:web").await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name(), "mcp:web:search");
        assert_eq!(tools[0].description(), "search the web");
    }

    #[tokio::test]
    async fn adapter_strips_prefix_before_invoking() {
        // Registry sees `mcp:web:search` but the upstream server
        // expects bare `search`. The adapter must forward the
        // server-native name.
        let client = one_tool_client();
        let tools = adapt_client_as_tools(client, "mcp:web").await.unwrap();
        let tool = tools.into_iter().next().unwrap();
        let out = tool.invoke(json!({"q": "rust"})).await.unwrap();
        assert_eq!(out["type"], "text");
        let text = out["text"].as_str().unwrap();
        assert!(text.starts_with("called search "), "{text}");
    }

    #[tokio::test]
    async fn empty_prefix_leaves_name_unchanged() {
        let client = one_tool_client();
        let tools = adapt_client_as_tools(client, "").await.unwrap();
        assert_eq!(tools[0].name(), "search");
    }

    #[test]
    fn image_tool_result_serialises_to_base64_json() {
        let v = tool_result_to_json(ToolResult::Image {
            mime: "image/png".into(),
            bytes: vec![0xDE, 0xAD, 0xBE, 0xEF],
        });
        assert_eq!(v["type"], "image");
        assert_eq!(v["mime"], "image/png");
        assert_eq!(v["bytes_base64"], "3q2+7w==");
    }

    #[test]
    fn image_round_trips_through_json_to_attachment() {
        let bytes = vec![1u8, 2, 3, 4, 5, 6];
        let json = tool_result_to_json(ToolResult::Image {
            mime: "image/jpeg".into(),
            bytes: bytes.clone(),
        });
        let lifted = image_attachment_from_tool_result(&json).expect("must lift");
        match lifted {
            Attachment::Image {
                mime,
                bytes: lifted_bytes,
            } => {
                assert_eq!(mime, "image/jpeg");
                assert_eq!(lifted_bytes, bytes);
            }
        }
    }

    #[test]
    fn image_lift_returns_none_for_non_image_result() {
        let v = tool_result_to_json(ToolResult::Text("hi".into()));
        assert!(image_attachment_from_tool_result(&v).is_none());
    }

    #[test]
    fn version_is_not_empty() {
        assert!(!version().is_empty());
    }
}
