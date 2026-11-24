//! The LLM-callable memory tool: `remember` saves key/value facts.

use std::borrow::Cow;
use std::sync::{Arc, LazyLock};
use std::time::Instant;

use async_trait::async_trait;
use regex::Regex;
use serde_json::{Value, json};

use assistd_ipc::Access;

use crate::TurnContext;
use crate::memory::MemoryOps;
use crate::{Tool, ToolError, ToolOutput};

/// Keys reject whitespace and uppercase so one concept keeps one spelling;
/// hyphens allow ISO dates (`standup.2026-09-11`).
const KEY_PATTERN: &str = r"^[a-z0-9._-]+$";
static KEY_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(KEY_PATTERN).expect("KEY_PATTERN compiles"));

/// Saves `(key, value)` facts about the user. A save's result repeats the key and
/// value, so every client that shows results shows what was kept.
#[derive(Debug)]
pub struct RememberTool {
    ops: Arc<MemoryOps>,
}

impl RememberTool {
    /// A tool saving through `ops`.
    pub fn new(ops: Arc<MemoryOps>) -> Self {
        Self { ops }
    }
}

#[async_trait]
impl Tool for RememberTool {
    fn name(&self) -> &'static str {
        "remember"
    }

    fn description(&self) -> Cow<'_, str> {
        "Save a durable memory about the user. Call whenever the user states a \
         stable preference, something about themselves, or anything they want \
         remembered; `key` names the memory and `text` is its value. Saving an \
         existing key overwrites it. Examples: \"I prefer vim over emacs\" -> \
         remember(key=\"editor_preference\", text=\"vim\"); \"my name is Ben\" \
         -> remember(key=\"user.name\", text=\"Ben\"). Do NOT save ephemeral \
         context from the current conversation. To look saved memories up, use \
         the `run` command `search --in memories <words>`."
            .into()
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "key": {
                    "type": "string",
                    "description": "snake_case identifier with optional dot \
                                    namespacing (e.g. user.name, \
                                    editor_preference, standup.2026-09-11), \
                                    matching ^[a-z0-9._-]+$."
                },
                "text": {
                    "type": "string",
                    "description": "The memory or preference to store."
                }
            },
            "required": ["key", "text"]
        })
    }

    /// The result repeats what the model sent.
    fn reads_host(&self, _args: &Value) -> bool {
        false
    }

    #[tracing::instrument(skip(self, args))]
    async fn invoke(&self, turn: &TurnContext, args: Value) -> Result<ToolOutput, ToolError> {
        let start = Instant::now();
        let key = valid_key(&args)?;
        let value = required_str(&args, "text")?.to_string();
        let report = format!("remembered {key}: {value}");
        self.ops.save(&key, value).await?;
        tracing::info!(target: "assistd::memory", key = %key, "remember saved");
        Ok(ToolOutput {
            accesses: vec![Access::Saved(key)],
            ..ToolOutput::text(report, start.elapsed())
        })
    }
}

fn valid_key(args: &Value) -> Result<String, ToolError> {
    let key = required_str(args, "key")?;
    if !KEY_RE.is_match(key) {
        return Err(ToolError::InvalidArgs(format!(
            "`key` must match {KEY_PATTERN} (snake_case + dot \
             namespacing, e.g. editor_preference or user.name)"
        )));
    }
    Ok(key.to_string())
}

fn required_str<'a>(args: &'a Value, name: &str) -> Result<&'a str, ToolError> {
    args.get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| ToolError::InvalidArgs(format!("`{name}` (string) is required")))
}

#[cfg(test)]
mod tests {
    use assistd_memory::{MemoryStore, SqliteHandle, SqliteTranscriptStore, TranscriptStore};
    use tempfile::TempDir;

    use super::*;

    /// A SQLite-backed `MemoryOps`; hold the tempdir for the test.
    async fn fresh_ops() -> (Arc<MemoryOps>, TempDir) {
        let temp = tempfile::tempdir().unwrap();
        let handle = SqliteHandle::open(&temp.path().join("memory.db"))
            .await
            .unwrap();
        let mem = Arc::new(MemoryStore::new(handle.clone()));
        let conv: Arc<dyn TranscriptStore> = Arc::new(SqliteTranscriptStore::new(handle));
        (Arc::new(MemoryOps::new(mem, conv)), temp)
    }

    fn invalid_args(err: ToolError) -> String {
        match err {
            ToolError::InvalidArgs(msg) => msg,
            other @ ToolError::Store(_) => panic!("expected InvalidArgs, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn remember_dedups_by_key() {
        let (ops, _dir) = fresh_ops().await;
        let tool = RememberTool::new(ops.clone());
        tool.invoke(
            &TurnContext::default(),
            json!({"key": "editor_preference", "text": "vim"}),
        )
        .await
        .unwrap();
        let saved = tool
            .invoke(
                &TurnContext::default(),
                json!({"key": "editor_preference", "text": "neovim"}),
            )
            .await
            .unwrap();
        assert_eq!(saved.output, "remembered editor_preference: neovim");
        let rows = ops.list_full("").await.unwrap();
        let keys: Vec<&str> = rows.iter().map(|row| row.key.as_str()).collect();
        assert_eq!(keys, ["editor_preference"]);
        assert_eq!(
            ops.load("editor_preference").await.unwrap().as_deref(),
            Some("neovim")
        );
    }

    #[tokio::test]
    async fn remember_rejects_invalid_key() {
        let (ops, _dir) = fresh_ops().await;
        let tool = RememberTool::new(ops);
        for bad in ["has spaces", "Editor_Pref", "trailing!", "slash/ed", ""] {
            let err = tool
                .invoke(&TurnContext::default(), json!({"key": bad, "text": "x"}))
                .await
                .unwrap_err();
            let msg = invalid_args(err);
            assert!(
                msg.starts_with(&format!("`key` must match {KEY_PATTERN} ")),
                "{bad:?}: {msg}"
            );
        }
    }

    /// Strict mode requires every declared property to be listed as
    /// required and no others to be accepted.
    #[tokio::test]
    async fn schema_satisfies_strict_mode() {
        let (ops, _dir) = fresh_ops().await;
        let schema = RememberTool::new(ops).parameters_schema();
        assert_eq!(schema["additionalProperties"], false);
        let properties: Vec<&String> = schema["properties"].as_object().unwrap().keys().collect();
        let required: Vec<&str> = schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(Value::as_str)
            .collect();
        assert_eq!(required, properties);
    }
}
