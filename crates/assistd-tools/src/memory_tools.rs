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
use assistd_embed::{EmbedJob, Embedder};
use assistd_memory::SemanticStore;
use async_trait::async_trait;
use regex::Regex;
use serde_json::{Value, json};
use tokio::sync::mpsc;

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
/// shared SQLite writer task. Also enqueues an embed job for the saved
/// value so future `recall(mode="semantic")` queries can find the
/// memory by paraphrase, not just by exact key prefix.
pub struct RememberTool {
    ops: Arc<MemoryOps>,
    /// Channel into the background embedder task. Closed-and-dropped
    /// when the embedding subsystem is disabled — `try_send` then no-ops
    /// silently and the memory still saves; only the index entry is
    /// missed (a future backfill can recover it).
    embed_tx: mpsc::Sender<EmbedJob>,
}

impl RememberTool {
    pub fn new(ops: Arc<MemoryOps>, embed_tx: mpsc::Sender<EmbedJob>) -> Self {
        Self { ops, embed_tx }
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

        let memory_id = self.ops.save(&key, value.clone()).await?;
        // Enqueue an embed job so semantic recall can find this memory
        // by paraphrase. `try_send` (not `send`) so a wedged embedder
        // never backpressures the tool. Skip when the id is 0 — that's
        // the `NoMemoryStore` sentinel, no row to FK.
        if memory_id != 0
            && self
                .embed_tx
                .try_send(EmbedJob::Memory {
                    memory_id,
                    text: value,
                })
                .is_err()
        {
            tracing::debug!(
                target: "assistd::embed",
                memory_id,
                "embed queue full or closed; remembered without semantic index entry"
            );
        }
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

/// LLM-callable tool that returns previously-saved memories.
///
/// Supports two query modes:
/// - `prefix`: list keys starting with the query string (`""` lists all).
///   Cheap, deterministic, ideal for hierarchical browsing like `user.*`.
/// - `semantic`: embed the query and rank memories by cosine similarity
///   against the saved values' embeddings. Robust to paraphrase — works
///   even when the user describes the fact in different words than the
///   stored key/value text. Falls back to the no-memories sentinel when
///   the embedding subsystem is disabled (or no memories have been
///   embedded yet).
///
/// Both modes return `<key>: <value>` lines so the model's downstream
/// formatting stays uniform.
pub struct RecallTool {
    ops: Arc<MemoryOps>,
    embedder: Arc<dyn Embedder>,
    semantic: Arc<dyn SemanticStore>,
    /// Embedding model name used to filter `memory_embeddings` rows by
    /// `model` so a query against today's embedder never collides with
    /// vectors produced by a previous model.
    embedding_model: String,
}

impl RecallTool {
    pub fn new(
        ops: Arc<MemoryOps>,
        embedder: Arc<dyn Embedder>,
        semantic: Arc<dyn SemanticStore>,
        embedding_model: String,
    ) -> Self {
        Self {
            ops,
            embedder,
            semantic,
            embedding_model,
        }
    }

    async fn recall_prefix(&self, prefix: &str) -> Result<(String, bool, usize)> {
        // Probe `list` first so we can detect overflow vs. exactly-cap.
        let total_keys = self.ops.list(prefix).await?.len();
        let mut pairs = self.ops.list_pairs(prefix, RECALL_LIMIT).await?;
        pairs.sort_by(|a, b| a.0.cmp(&b.0));
        let truncated = total_keys > RECALL_LIMIT;
        Ok((format_pairs(&pairs), truncated, pairs.len()))
    }

    async fn recall_semantic(&self, query: &str) -> Result<(String, bool, usize)> {
        if self.embedding_model.is_empty() {
            // Embedding subsystem disabled — degrade gracefully so the
            // model can fall back to prefix mode without erroring.
            return Ok(("(no memories)".to_string(), false, 0));
        }
        let vec = match self.embedder.embed(query.to_string()).await {
            Ok(v) => v,
            Err(e) => {
                // Best-effort: log and return empty rather than
                // surfacing as a tool error. The LLM can retry with
                // prefix mode if it cares.
                tracing::debug!(
                    target: "assistd::embed",
                    error = %e,
                    "recall(semantic) embed failed; returning empty"
                );
                return Ok(("(no memories)".to_string(), false, 0));
            }
        };
        let hits = self
            .semantic
            .nearest_memories(vec, RECALL_LIMIT, &self.embedding_model)
            .await?;
        if hits.is_empty() {
            return Ok(("(no memories)".to_string(), false, 0));
        }
        // Render the same `<key>: <value>` shape as the prefix path so
        // downstream formatting in the model stays uniform.
        let pairs: Vec<(String, String)> = hits
            .iter()
            .map(|h| (h.key.clone(), h.value.clone()))
            .collect();
        let n = pairs.len();
        Ok((format_pairs(&pairs), false, n))
    }
}

#[async_trait]
impl Tool for RecallTool {
    fn name(&self) -> &str {
        "recall"
    }

    fn description(&self) -> &str {
        "Retrieve previously remembered facts and preferences about the user. \
         Call this when prior context might help (e.g. answering \"what editor \
         should I use?\", personalizing a response, or whenever the user \
         references something they told you before). \
         Two query modes: \
         (1) `mode=\"prefix\"` lists keys starting with `query` — pass `query=\"\"` \
         to list every memory, or `query=\"user.\"` to filter by namespace. \
         (2) `mode=\"semantic\"` embeds `query` and ranks memories by meaning. \
         Use semantic for natural-language questions like \"what editor do I \
         prefer?\"; use prefix for hierarchical browsing. \
         Returns up to 50 `<key>: <value>` lines. If no memories match, the \
         output is `(no memories)`."
    }

    fn parameters_schema(&self) -> Value {
        // `query` and `mode` are both required because
        // `ToolRegistry::openai_schemas` hardcodes `strict: true`, which
        // requires every declared property to appear in `required`.
        json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Either a key prefix (when mode=\"prefix\") or \
                                    a natural-language question (when mode=\"semantic\"). \
                                    Pass \"\" with mode=\"prefix\" to list every memory."
                },
                "mode": {
                    "type": "string",
                    "enum": ["prefix", "semantic"],
                    "description": "How to match. \"prefix\" lists keys starting with \
                                    `query`. \"semantic\" embeds `query` and ranks memories \
                                    by meaning (use for paraphrase-tolerant lookup)."
                }
            },
            "required": ["query", "mode"]
        })
    }

    #[tracing::instrument(skip(self, args), fields(mode = tracing::field::Empty))]
    async fn invoke(&self, args: Value) -> Result<Value> {
        let start = Instant::now();
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("`query` (string) is required"))?
            .to_string();
        let mode = args
            .get("mode")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("`mode` (string, \"prefix\" or \"semantic\") is required"))?
            .to_string();
        tracing::Span::current().record("mode", mode.as_str());

        let (output, truncated, returned) = match mode.as_str() {
            "prefix" => self.recall_prefix(&query).await?,
            "semantic" => self.recall_semantic(&query).await?,
            other => {
                return Err(anyhow!(
                    "`mode` must be \"prefix\" or \"semantic\" (got {other:?})"
                ));
            }
        };

        let duration_ms = start.elapsed().as_millis();
        tracing::info!(
            target: "assistd::memory",
            mode = %mode,
            returned = returned,
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

fn format_pairs(pairs: &[(String, String)]) -> String {
    if pairs.is_empty() {
        return "(no memories)".to_string();
    }
    let mut s = String::with_capacity(pairs.iter().map(|(k, v)| k.len() + v.len() + 2).sum());
    for (k, v) in pairs {
        s.push_str(k);
        s.push_str(": ");
        s.push_str(v);
        s.push('\n');
    }
    if s.ends_with('\n') {
        s.pop();
    }
    s
}

/// LLM-callable tool that searches *past conversation history* for
/// messages similar in meaning to the query. Different data source from
/// `recall`, which works over saved key/value facts. Use this when the
/// user references something they "discussed before" or "worked on
/// last month" — paraphrase-tolerant semantic search over the chunked
/// conversation log.
pub struct SearchMemoryTool {
    embedder: Arc<dyn Embedder>,
    semantic: Arc<dyn SemanticStore>,
    embedding_model: String,
}

impl SearchMemoryTool {
    pub fn new(
        embedder: Arc<dyn Embedder>,
        semantic: Arc<dyn SemanticStore>,
        embedding_model: String,
    ) -> Self {
        Self {
            embedder,
            semantic,
            embedding_model,
        }
    }
}

#[async_trait]
impl Tool for SearchMemoryTool {
    fn name(&self) -> &str {
        "search_memory"
    }

    fn description(&self) -> &str {
        "Search past conversation history (across sessions) for messages \
         similar in meaning to a query. Differs from `recall`: `recall` \
         looks up *saved facts* (key/value), while `search_memory` searches \
         *past dialogue text* — use it when the user references something \
         they 'discussed before' or 'worked on last month'. Robust to \
         paraphrase: a query like \"that rust project we discussed\" will \
         match a past message about \"the assistd embedding daemon in Rust\". \
         Returns up to `limit` ranked snippets with timestamps and roles."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language query. The model embeds \
                                    this and finds the top-`limit` past messages \
                                    by cosine similarity."
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Maximum number of hits to return (1-20). \
                                    Smaller is sharper; larger casts a wider net."
                }
            },
            "required": ["query", "limit"]
        })
    }

    #[tracing::instrument(skip(self, args), fields(limit = tracing::field::Empty))]
    async fn invoke(&self, args: Value) -> Result<Value> {
        let start = Instant::now();
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("`query` (string) is required"))?
            .to_string();
        let limit = args
            .get("limit")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| anyhow!("`limit` (integer) is required"))?;
        if !(1..=20).contains(&limit) {
            return Err(anyhow!("`limit` must be in 1..=20 (got {limit})"));
        }
        tracing::Span::current().record("limit", limit);

        if self.embedding_model.is_empty() {
            return Ok(json!({
                "output":      "(no past conversations indexed)",
                "exit_code":   0,
                "duration_ms": start.elapsed().as_millis(),
                "truncated":   false,
            }));
        }
        let vec = match self.embedder.embed(query).await {
            Ok(v) => v,
            Err(e) => {
                tracing::debug!(
                    target: "assistd::embed",
                    error = %e,
                    "search_memory embed failed; returning empty"
                );
                return Ok(json!({
                    "output":      "(embedding unavailable)",
                    "exit_code":   0,
                    "duration_ms": start.elapsed().as_millis(),
                    "truncated":   false,
                }));
            }
        };
        let hits = self
            .semantic
            .nearest_chunks(vec, limit as usize, &self.embedding_model)
            .await?;

        let output = if hits.is_empty() {
            "(no matches)".to_string()
        } else {
            let mut s = String::new();
            for h in &hits {
                // Compact one-line-per-hit format. Timestamps + role +
                // similarity give the model enough metadata to weigh
                // the relevance of each line.
                s.push_str(&format!(
                    "[{} {} sim={:.0}%] {}\n",
                    h.timestamp,
                    h.role.as_wire(),
                    h.similarity * 100.0,
                    h.content.replace('\n', " ")
                ));
            }
            if s.ends_with('\n') {
                s.pop();
            }
            s
        };
        let duration_ms = start.elapsed().as_millis();
        tracing::info!(
            target: "assistd::embed",
            returned = hits.len(),
            duration_ms = duration_ms,
            "search_memory returned chunks"
        );
        Ok(json!({
            "output":      output,
            "exit_code":   0,
            "duration_ms": duration_ms,
            "truncated":   false,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ToolRegistry;
    use assistd_embed::NoEmbedder;
    use assistd_memory::{
        NoConversationStore, NoMemoryStore, NoSemanticStore, SqliteConversationStore, SqliteHandle,
        SqliteMemoryStore,
    };
    use tokio::sync::watch;

    /// Tests that don't exercise the embed channel use a 1-capacity
    /// channel with the receiver dropped — `try_send` always fails, so
    /// the embed path is exercised but doesn't actually do anything.
    fn closed_embed_tx() -> mpsc::Sender<EmbedJob> {
        let (tx, rx) = mpsc::channel::<EmbedJob>(1);
        drop(rx);
        tx
    }

    /// Open embed channel for tests that want to assert RememberTool
    /// actually queues an embed job. Returns the sender to give to the
    /// tool plus the receiver to drain in the test body.
    fn live_embed_tx() -> (mpsc::Sender<EmbedJob>, mpsc::Receiver<EmbedJob>) {
        mpsc::channel::<EmbedJob>(8)
    }

    fn no_embedder() -> Arc<dyn Embedder> {
        Arc::new(NoEmbedder)
    }

    fn no_semantic() -> Arc<dyn SemanticStore> {
        Arc::new(NoSemanticStore)
    }

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
        let tool = RememberTool::new(ops.clone(), closed_embed_tx());
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
        let tool = RememberTool::new(ops.clone(), closed_embed_tx());
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
        let tool = RememberTool::new(ops, closed_embed_tx());
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
        let tool = RememberTool::new(ops, closed_embed_tx());
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
        let tool = RecallTool::new(ops, no_embedder(), no_semantic(), String::new());
        let result = tool
            .invoke(json!({"query": "", "mode": "prefix"}))
            .await
            .unwrap();
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
        let tool = RecallTool::new(ops, no_embedder(), no_semantic(), String::new());
        let result = tool
            .invoke(json!({"query": "user.", "mode": "prefix"}))
            .await
            .unwrap();
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
        let tool = RecallTool::new(ops, no_embedder(), no_semantic(), String::new());
        let result = tool
            .invoke(json!({"query": "", "mode": "prefix"}))
            .await
            .unwrap();
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
        let tool = RecallTool::new(ops, no_embedder(), no_semantic(), String::new());
        let result = tool
            .invoke(json!({"query": "k.", "mode": "prefix"}))
            .await
            .unwrap();
        assert_eq!(result["truncated"], true);
        let output = result["output"].as_str().unwrap();
        assert_eq!(output.lines().count(), RECALL_LIMIT);
    }

    #[tokio::test]
    async fn recall_with_no_memory_store_no_ops() {
        // TUI graceful path: NoMemoryStore returns empty for `list`, so
        // recall reports the no-memories marker rather than erroring.
        let tool = RecallTool::new(no_ops(), no_embedder(), no_semantic(), String::new());
        let result = tool
            .invoke(json!({"query": "", "mode": "prefix"}))
            .await
            .unwrap();
        assert_eq!(result["output"], "(no memories)");
    }

    #[tokio::test]
    async fn recall_rejects_missing_args() {
        // strict-mode schema requires `query` and `mode` — a missing
        // arg is a protocol error from the model and surfaces as Err.
        let tool = RecallTool::new(no_ops(), no_embedder(), no_semantic(), String::new());
        let err = tool.invoke(json!({})).await.unwrap_err();
        assert!(
            err.to_string().contains("query"),
            "expected error to mention `query`: {err}"
        );
        let err = tool.invoke(json!({"query": "anything"})).await.unwrap_err();
        assert!(
            err.to_string().contains("mode"),
            "expected error to mention `mode`: {err}"
        );
    }

    #[tokio::test]
    async fn recall_rejects_invalid_mode() {
        let tool = RecallTool::new(no_ops(), no_embedder(), no_semantic(), String::new());
        let err = tool
            .invoke(json!({"query": "x", "mode": "bogus"}))
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("prefix") || err.to_string().contains("semantic"),
            "expected error to list valid modes: {err}"
        );
    }

    #[tokio::test]
    async fn recall_semantic_with_disabled_embedder_returns_no_memories() {
        // NoEmbedder + empty model name → recall_semantic short-circuits
        // to the no-memories sentinel (matches the behavior the LLM
        // sees when embedding is turned off in config).
        let tool = RecallTool::new(no_ops(), no_embedder(), no_semantic(), String::new());
        let result = tool
            .invoke(json!({"query": "what editor do I prefer", "mode": "semantic"}))
            .await
            .unwrap();
        assert_eq!(result["output"], "(no memories)");
    }

    // --- Schema sanity ---------------------------------------------------

    #[test]
    fn registered_in_openai_schemas_correctly() {
        // Build a registry holding all three tools and inspect the wire
        // shape `step` will send to llama-server. Pins the strict-mode
        // contract that every property is required.
        let mut reg = ToolRegistry::new();
        reg.register(RememberTool::new(no_ops(), closed_embed_tx()));
        reg.register(RecallTool::new(
            no_ops(),
            no_embedder(),
            no_semantic(),
            String::new(),
        ));
        reg.register(SearchMemoryTool::new(
            no_embedder(),
            no_semantic(),
            String::new(),
        ));
        let schemas = reg.openai_schemas();
        assert_eq!(schemas.len(), 3);

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
        assert!(names.contains(&"query"));
        assert!(names.contains(&"mode"));

        let search = &schemas[2]["function"];
        assert_eq!(search["name"], "search_memory");
        assert_eq!(search["strict"], true);
        let req = search["parameters"]["required"].as_array().unwrap();
        let names: Vec<&str> = req.iter().filter_map(|v| v.as_str()).collect();
        assert!(names.contains(&"query"));
        assert!(names.contains(&"limit"));
    }

    #[tokio::test]
    async fn remember_enqueues_embed_job_with_value_text() {
        // RememberTool should fire an EmbedJob::Memory through the
        // channel after a successful save, so the saved value can be
        // semantically searched later.
        let (ops, _w) = fresh_ops().await;
        let (etx, mut erx) = live_embed_tx();
        let tool = RememberTool::new(ops.clone(), etx);
        tool.invoke(json!({"key": "editor_preference", "value": "vim is the way"}))
            .await
            .unwrap();
        // Drain the queue. The job's memory_id should be non-zero (the
        // SQLite-backed save returns the actual rowid).
        let job = tokio::time::timeout(std::time::Duration::from_secs(1), erx.recv())
            .await
            .expect("embed job arrived")
            .expect("channel still open");
        match job {
            EmbedJob::Memory { memory_id, text } => {
                assert!(memory_id > 0, "expected real rowid, got {memory_id}");
                assert_eq!(text, "vim is the way");
            }
            EmbedJob::Chunk { .. } => panic!("expected Memory job, got Chunk"),
        }
    }

    #[tokio::test]
    async fn remember_then_recall_round_trip() {
        // Belt-and-suspenders: drive both tools end-to-end against the
        // same MemoryOps and verify the saved pair shows up in recall.
        let (ops, _w) = fresh_ops().await;
        let remember = RememberTool::new(ops.clone(), closed_embed_tx());
        let recall = RecallTool::new(ops, no_embedder(), no_semantic(), String::new());
        remember
            .invoke(json!({"key": "editor_preference", "value": "vim"}))
            .await
            .unwrap();
        let result = recall
            .invoke(json!({"query": "", "mode": "prefix"}))
            .await
            .unwrap();
        assert!(
            result["output"]
                .as_str()
                .unwrap()
                .contains("editor_preference: vim"),
            "recall should surface the saved pair: {result:#}"
        );
    }
}
