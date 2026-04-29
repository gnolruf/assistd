//! LLM-callable memory tools: `remember` and `recall`.
//!
//! Both tools sit alongside [`crate::RunTool`] in the daemon's
//! [`crate::ToolRegistry`]. The model decides when to invoke them; both
//! delegate straight to [`MemoryOps`], which writes through the same
//! single SQLite writer the chat-turn persistence path uses (so saves
//! never block the agent thread for disk).
//!
//! Dedup is by key: the `memories` table has `key TEXT NOT NULL UNIQUE`
//! and `save_memory` does `ON CONFLICT(key) DO UPDATE`, so re-saving the
//! same key overwrites the value rather than producing a second row.
//! That is the AC #2 contract for "duplicate memories are not
//! re-inserted" — the LLM is nudged toward stable snake_case keys via
//! the [`RememberTool::description`] examples, and the
//! `^[a-z0-9._]+$` validator below rejects whitespace/uppercase so two
//! turns about the same concept don't drift onto different keys.

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use regex::Regex;
use serde_json::{Value, json};

use crate::Tool;
use crate::memory::MemoryOps;

/// Cap on `recall` result size. The agent loop sees these pairs as text
/// in a tool result; keeping the cap small keeps the followup turn's
/// context small. 50 is plenty for a single user's preferences.
const RECALL_LIMIT: usize = 50;

/// Validation regex for memory keys. `lazy_static`-free: built once per
/// invoke, which is fine here (one tool call per turn) — and avoids a
/// new `OnceLock` for a single-line regex.
const KEY_PATTERN: &str = r"^[a-z0-9._]+$";

/// LLM-callable tool that saves a `(key, value)` pair into persistent
/// memory. Holds a clone of [`MemoryOps`] so it can write through the
/// shared SQLite writer task.
pub struct RememberTool {
    ops: Arc<MemoryOps>,
}

impl RememberTool {
    pub fn new(ops: Arc<MemoryOps>) -> Self {
        Self { ops }
    }
}

#[async_trait]
impl Tool for RememberTool {
    fn name(&self) -> &str {
        "remember"
    }

    fn description(&self) -> &str {
        "Save a fact or preference about the user across conversations. \
         Call this whenever the user states a stable preference, fact about \
         themselves, or anything they say they want remembered. Examples: \
         \"I prefer vim over emacs\" -> remember(key=\"editor_preference\", \
         value=\"vim\"); \"my name is Ben\" -> remember(key=\"user.name\", \
         value=\"Ben\"); \"I work in PST\" -> remember(key=\"user.timezone\", \
         value=\"PST\"). Use snake_case keys with optional dots for \
         namespacing (e.g. user.name, editor_preference, project.assistd.dir). \
         Calling remember with an existing key overwrites the previous \
         value. Do NOT call this for ephemeral context within a single \
         conversation — only durable user-facts."
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
                                    editor_preference). Must match \
                                    ^[a-z0-9._]+$."
                },
                "value": {
                    "type": "string",
                    "description": "the fact or preference text to store"
                }
            },
            "required": ["key", "value"]
        })
    }

    #[tracing::instrument(skip(self, args), fields(key = tracing::field::Empty))]
    async fn invoke(&self, args: Value) -> Result<Value> {
        let start = Instant::now();
        let key = args
            .get("key")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("`key` (string) is required"))?
            .to_string();
        let value = args
            .get("value")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("`value` (string) is required"))?
            .to_string();
        tracing::Span::current().record("key", key.as_str());

        let re = Regex::new(KEY_PATTERN).expect("KEY_PATTERN compiles");
        if !re.is_match(&key) {
            return Err(anyhow!(
                "`key` must match {KEY_PATTERN} (snake_case + dot \
                 namespacing, e.g. editor_preference or user.name)"
            ));
        }

        self.ops.save(&key, value).await?;
        let duration_ms = start.elapsed().as_millis();
        tracing::info!(
            target: "assistd::memory",
            key = %key,
            duration_ms = duration_ms,
            "remember saved"
        );
        Ok(json!({
            "output":      format!("remembered {key}"),
            "exit_code":   0,
            "duration_ms": duration_ms,
            "truncated":   false,
        }))
    }
}

/// LLM-callable tool that returns previously-saved memories whose keys
/// match a prefix. `prefix == ""` lists every memory (capped at
/// [`RECALL_LIMIT`]).
pub struct RecallTool {
    ops: Arc<MemoryOps>,
}

impl RecallTool {
    pub fn new(ops: Arc<MemoryOps>) -> Self {
        Self { ops }
    }
}

#[async_trait]
impl Tool for RecallTool {
    fn name(&self) -> &str {
        "recall"
    }

    fn description(&self) -> &str {
        "Retrieve previously remembered facts and preferences about the user. \
         Call this at the start of a turn when prior context might help \
         (e.g. before answering \"what editor should I use?\", before \
         personalizing a response, or whenever the user references something \
         they may have told you in a past conversation). The `prefix` argument \
         filters by key prefix; pass an empty string \"\" to list every \
         memory, or e.g. \"user.\" to list only keys under user.*. Returns \
         up to 50 `<key>: <value>` lines, sorted alphabetically. If no \
         memories match, the output is `(no memories)`."
    }

    fn parameters_schema(&self) -> Value {
        // `prefix` is required (rather than optional) because
        // `ToolRegistry::openai_schemas` hardcodes `strict: true`, which
        // requires every property in `properties` to also appear in
        // `required`. The LLM passes `""` to list everything — the
        // description spells this out.
        json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "prefix": {
                    "type": "string",
                    "description": "Key prefix filter. Pass \"\" to list \
                                    every memory; otherwise e.g. \"user.\" \
                                    or \"editor_\" to filter."
                }
            },
            "required": ["prefix"]
        })
    }

    #[tracing::instrument(skip(self, args), fields(prefix = tracing::field::Empty))]
    async fn invoke(&self, args: Value) -> Result<Value> {
        let start = Instant::now();
        let prefix = args
            .get("prefix")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("`prefix` (string) is required (use \"\" for all)"))?
            .to_string();
        tracing::Span::current().record("prefix", prefix.as_str());

        // Probe `list` first so we can detect overflow vs. exactly-cap.
        // The N+1 inside `list_pairs` is fine for ≤ RECALL_LIMIT keys;
        // the extra `list` here is one cheap read.
        let total_keys = self.ops.list(&prefix).await?.len();
        let mut pairs = self.ops.list_pairs(&prefix, RECALL_LIMIT).await?;
        // SqliteMemoryStore returns keys in lexicographic order today,
        // but the trait makes no guarantee — sort here so the test and
        // the LLM see a stable order regardless of backend.
        pairs.sort_by(|a, b| a.0.cmp(&b.0));

        let truncated = total_keys > RECALL_LIMIT;
        let output = if pairs.is_empty() {
            "(no memories)".to_string()
        } else {
            let mut s =
                String::with_capacity(pairs.iter().map(|(k, v)| k.len() + v.len() + 2).sum());
            for (k, v) in &pairs {
                s.push_str(k);
                s.push_str(": ");
                s.push_str(v);
                s.push('\n');
            }
            // Drop the trailing newline so the LLM sees a clean block.
            if s.ends_with('\n') {
                s.pop();
            }
            s
        };
        let duration_ms = start.elapsed().as_millis();
        tracing::info!(
            target: "assistd::memory",
            prefix = %prefix,
            returned = pairs.len(),
            truncated = truncated,
            duration_ms = duration_ms,
            "recall returned pairs"
        );
        Ok(json!({
            "output":      output,
            "exit_code":   0,
            "duration_ms": duration_ms,
            "truncated":   truncated,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ToolRegistry;
    use assistd_memory::{
        NoConversationStore, NoMemoryStore, SqliteConversationStore, SqliteHandle,
        SqliteMemoryStore,
    };
    use tokio::sync::watch;

    /// Stand up a SQLite-backed `MemoryOps` in a tempdir. The writer
    /// task is leaked at end of test (the `_w` JoinHandle is dropped);
    /// that mirrors the existing memory-crate test pattern.
    async fn fresh_ops() -> (Arc<MemoryOps>, tokio::task::JoinHandle<()>) {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("memory.db");
        // Leak the tempdir for the duration of the test — `path` must
        // outlive the handle. Cleanup happens on test process exit.
        std::mem::forget(temp);
        let (_tx, rx) = watch::channel(false);
        let (handle, writer) = SqliteHandle::open(&path, rx).await.unwrap();
        let handle = Arc::new(handle);
        let mem: Arc<dyn assistd_memory::MemoryStore> =
            Arc::new(SqliteMemoryStore::new(handle.clone()));
        let conv: Arc<dyn assistd_memory::ConversationStore> =
            Arc::new(SqliteConversationStore::new(handle));
        (Arc::new(MemoryOps::new(mem, conv)), writer)
    }

    fn no_ops() -> Arc<MemoryOps> {
        Arc::new(MemoryOps::new(
            Arc::new(NoMemoryStore),
            Arc::new(NoConversationStore),
        ))
    }

    // --- Remember --------------------------------------------------------

    #[tokio::test]
    async fn remember_saves_key_value() {
        // AC #1 wire-level: a `remember` invocation lands a row in the
        // store at the requested key.
        let (ops, _w) = fresh_ops().await;
        let tool = RememberTool::new(ops.clone());
        let result = tool
            .invoke(json!({"key": "editor_preference", "value": "vim"}))
            .await
            .unwrap();
        assert_eq!(result["exit_code"], 0);
        assert_eq!(result["output"], "remembered editor_preference");
        assert_eq!(
            ops.load("editor_preference").await.unwrap().as_deref(),
            Some("vim")
        );
    }

    #[tokio::test]
    async fn remember_dedups_by_key() {
        // AC #2: re-saving the same key overwrites the value and
        // produces no second row. The schema's UNIQUE + ON CONFLICT
        // guarantees this — the test pins the contract.
        let (ops, _w) = fresh_ops().await;
        let tool = RememberTool::new(ops.clone());
        tool.invoke(json!({"key": "editor_preference", "value": "vim"}))
            .await
            .unwrap();
        tool.invoke(json!({"key": "editor_preference", "value": "neovim"}))
            .await
            .unwrap();
        let keys = ops.list("").await.unwrap();
        assert_eq!(keys, vec!["editor_preference"]);
        assert_eq!(
            ops.load("editor_preference").await.unwrap().as_deref(),
            Some("neovim")
        );
    }

    #[tokio::test]
    async fn remember_rejects_invalid_key() {
        let (ops, _w) = fresh_ops().await;
        let tool = RememberTool::new(ops);
        for bad in ["has spaces", "Editor_Pref", "trailing!", ""] {
            let err = tool
                .invoke(json!({"key": bad, "value": "x"}))
                .await
                .unwrap_err();
            let msg = err.to_string();
            assert!(
                msg.contains("key") || msg.contains("required"),
                "expected a key/required error for {bad:?}, got: {msg}"
            );
        }
    }

    #[tokio::test]
    async fn remember_rejects_missing_args() {
        let (ops, _w) = fresh_ops().await;
        let tool = RememberTool::new(ops);
        let err = tool.invoke(json!({})).await.unwrap_err();
        assert!(
            err.to_string().contains("key"),
            "missing-key error should mention `key`: {err}"
        );
        let err = tool.invoke(json!({"key": "user.name"})).await.unwrap_err();
        assert!(
            err.to_string().contains("value"),
            "missing-value error should mention `value`: {err}"
        );
    }

    // --- Recall ----------------------------------------------------------

    #[tokio::test]
    async fn recall_returns_pairs_sorted() {
        let (ops, _w) = fresh_ops().await;
        ops.save("user.name", "Ben".into()).await.unwrap();
        ops.save("editor_preference", "vim".into()).await.unwrap();
        ops.save("user.timezone", "PST".into()).await.unwrap();
        let tool = RecallTool::new(ops);
        let result = tool.invoke(json!({"prefix": ""})).await.unwrap();
        assert_eq!(result["exit_code"], 0);
        assert_eq!(result["truncated"], false);
        let output = result["output"].as_str().unwrap();
        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(
            lines,
            vec![
                "editor_preference: vim",
                "user.name: Ben",
                "user.timezone: PST",
            ]
        );
    }

    #[tokio::test]
    async fn recall_filters_by_prefix() {
        let (ops, _w) = fresh_ops().await;
        ops.save("user.name", "Ben".into()).await.unwrap();
        ops.save("user.timezone", "PST".into()).await.unwrap();
        ops.save("editor_preference", "vim".into()).await.unwrap();
        let tool = RecallTool::new(ops);
        let result = tool.invoke(json!({"prefix": "user."})).await.unwrap();
        let output = result["output"].as_str().unwrap();
        assert!(output.contains("user.name: Ben"), "{output}");
        assert!(output.contains("user.timezone: PST"), "{output}");
        assert!(
            !output.contains("editor_preference"),
            "prefix filter should exclude editor_preference: {output}"
        );
    }

    #[tokio::test]
    async fn recall_empty_returns_no_memories_marker() {
        let (ops, _w) = fresh_ops().await;
        let tool = RecallTool::new(ops);
        let result = tool.invoke(json!({"prefix": ""})).await.unwrap();
        assert_eq!(result["output"], "(no memories)");
        assert_eq!(result["truncated"], false);
    }

    #[tokio::test]
    async fn recall_caps_at_limit() {
        // Save more than RECALL_LIMIT entries; recall returns exactly
        // RECALL_LIMIT and reports truncated.
        let (ops, _w) = fresh_ops().await;
        for i in 0..(RECALL_LIMIT + 5) {
            // Zero-pad the index so lexicographic sort matches numeric
            // — otherwise key.10 sorts before key.2, and the bookkeeping
            // about which 50 we kept becomes confusing.
            ops.save(&format!("k.{i:03}"), format!("v{i}"))
                .await
                .unwrap();
        }
        let tool = RecallTool::new(ops);
        let result = tool.invoke(json!({"prefix": "k."})).await.unwrap();
        assert_eq!(result["truncated"], true);
        let output = result["output"].as_str().unwrap();
        assert_eq!(output.lines().count(), RECALL_LIMIT);
    }

    #[tokio::test]
    async fn recall_with_no_memory_store_no_ops() {
        // TUI graceful path: NoMemoryStore returns empty for `list`, so
        // recall reports the no-memories marker rather than erroring.
        let tool = RecallTool::new(no_ops());
        let result = tool.invoke(json!({"prefix": ""})).await.unwrap();
        assert_eq!(result["output"], "(no memories)");
    }

    #[tokio::test]
    async fn recall_rejects_missing_prefix() {
        // strict-mode schema requires `prefix` — a missing arg is a
        // protocol error from the model and surfaces as Err, which the
        // agent loop turns into a synthetic tool result with a recovery
        // hint.
        let tool = RecallTool::new(no_ops());
        let err = tool.invoke(json!({})).await.unwrap_err();
        assert!(
            err.to_string().contains("prefix"),
            "expected error to mention `prefix`: {err}"
        );
    }

    // --- Schema sanity ---------------------------------------------------

    #[test]
    fn registered_in_openai_schemas_correctly() {
        // Build a registry holding both tools and inspect the wire
        // shape `step` will send to llama-server. Pins the strict-mode
        // contract that `prefix` is required for recall.
        let mut reg = ToolRegistry::new();
        reg.register(RememberTool::new(no_ops()));
        reg.register(RecallTool::new(no_ops()));
        let schemas = reg.openai_schemas();
        assert_eq!(schemas.len(), 2);

        let remember = &schemas[0]["function"];
        assert_eq!(remember["name"], "remember");
        assert_eq!(remember["strict"], true);
        let req = remember["parameters"]["required"].as_array().unwrap();
        let names: Vec<&str> = req.iter().filter_map(|v| v.as_str()).collect();
        assert!(names.contains(&"key"));
        assert!(names.contains(&"value"));

        let recall = &schemas[1]["function"];
        assert_eq!(recall["name"], "recall");
        assert_eq!(recall["strict"], true);
        let req = recall["parameters"]["required"].as_array().unwrap();
        let names: Vec<&str> = req.iter().filter_map(|v| v.as_str()).collect();
        assert_eq!(names, vec!["prefix"]);
    }

    #[tokio::test]
    async fn remember_then_recall_round_trip() {
        // Belt-and-suspenders: drive both tools end-to-end against the
        // same MemoryOps and verify the saved pair shows up in recall.
        let (ops, _w) = fresh_ops().await;
        let remember = RememberTool::new(ops.clone());
        let recall = RecallTool::new(ops);
        remember
            .invoke(json!({"key": "editor_preference", "value": "vim"}))
            .await
            .unwrap();
        let result = recall.invoke(json!({"prefix": ""})).await.unwrap();
        assert!(
            result["output"]
                .as_str()
                .unwrap()
                .contains("editor_preference: vim"),
            "recall should surface the saved pair: {result:#}"
        );
    }
}
