//! LLM-callable memory tools. `remember` and `recall` work on saved
//! key/value facts; `reminisce` searches past dialogue.

use std::sync::{Arc, LazyLock};
use std::time::Instant;

use assistd_embed::{EmbedJob, Embedder};
use assistd_memory::{EmbeddingHit, MemoryHit, SemanticStore, SessionId};
use async_trait::async_trait;
use regex::Regex;
use serde_json::{Value, json};
use tokio::sync::{mpsc, watch};

use crate::memory::MemoryOps;
use crate::{Tool, ToolError};

const RECALL_LIMIT: usize = 50;

/// Keys reject whitespace and uppercase so one concept keeps one spelling;
/// hyphens allow ISO dates (`standup.2026-09-11`).
const KEY_PATTERN: &str = r"^[a-z0-9._-]+$";
static KEY_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(KEY_PATTERN).expect("KEY_PATTERN compiles"));

/// Saves a `(key, value)` pair and queues its value for embedding so
/// `recall` can find it by paraphrase.
pub struct RememberTool {
    ops: Arc<MemoryOps>,
    /// Closed when embedding is disabled; the memory still saves unindexed.
    embed_tx: mpsc::Sender<EmbedJob>,
}

impl RememberTool {
    /// A tool saving through `ops` and queueing each value on `embed_tx`.
    pub fn new(ops: Arc<MemoryOps>, embed_tx: mpsc::Sender<EmbedJob>) -> Self {
        Self { ops, embed_tx }
    }

    fn queue_embedding(&self, memory_id: i64, text: String) {
        if self
            .embed_tx
            .try_send(EmbedJob::Memory { memory_id, text })
            .is_err()
        {
            tracing::debug!(
                target: "assistd::embed",
                memory_id,
                "embed queue full or closed; remembered without semantic index entry"
            );
        }
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
         namespacing and hyphens where they read naturally (e.g. \
         user.name, editor_preference, standup.2026-09-11). \
         Calling remember with an existing key overwrites the previous \
         value. Do NOT call this for ephemeral context within a single \
         conversation; only durable user-facts."
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
                                    editor_preference, standup.2026-09-11). \
                                    Must match ^[a-z0-9._-]+$."
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
    async fn invoke(&self, args: Value) -> Result<Value, ToolError> {
        let start = Instant::now();
        let key = required_str(&args, "key")?.to_string();
        let value = required_str(&args, "value")?.to_string();
        tracing::Span::current().record("key", key.as_str());

        if !KEY_RE.is_match(&key) {
            return Err(ToolError::InvalidArgs(format!(
                "`key` must match {KEY_PATTERN} (snake_case + dot \
                 namespacing, e.g. editor_preference or user.name)"
            )));
        }

        let memory_id = self.ops.save(&key, value.clone()).await?;
        if memory_id != 0 {
            self.queue_embedding(memory_id, value);
        }
        let duration_ms = start.elapsed().as_millis();
        tracing::info!(
            target: "assistd::memory",
            key = %key,
            duration_ms = duration_ms,
            "remember saved"
        );
        Ok(tool_result(&format!("remembered {key}"), duration_ms))
    }
}

/// Returns saved memories ranked by semantic similarity to a query, as
/// `<key>: <value>` lines.
pub struct RecallTool {
    embedder: Arc<dyn Embedder>,
    semantic: Arc<dyn SemanticStore>,
    /// Only vectors from this model are matched.
    embedding_model: String,
}

impl RecallTool {
    /// A tool ranking memories in `semantic` by `embedder` vectors; an empty
    /// `embedding_model` means embedding is disabled.
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

    /// Memories nearest `query`; empty when embedding is disabled or fails.
    async fn nearest_memories(&self, query: String) -> Result<Vec<MemoryHit>, ToolError> {
        if self.embedding_model.is_empty() {
            return Ok(Vec::new());
        }
        match self.embedder.embed(query).await {
            Ok(query_vec) => Ok(self
                .semantic
                .nearest_memories(query_vec, RECALL_LIMIT, &self.embedding_model)
                .await?),
            Err(e) => {
                tracing::debug!(
                    target: "assistd::embed",
                    error = %e,
                    "recall embed failed; returning empty"
                );
                Ok(Vec::new())
            }
        }
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
         references something they told you before). Embeds `query` and ranks \
         memories by semantic similarity, so paraphrased questions still match \
         the stored fact (e.g. \"what editor do I prefer?\" finds an \
         `editor_preference` memory). Returns up to 50 `<key>: <value>` lines. \
         If no memories match, the output is `(no memories)`."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language question. The model embeds \
                                    this and ranks memories by cosine similarity \
                                    against their stored values."
                }
            },
            "required": ["query"]
        })
    }

    #[tracing::instrument(skip(self, args))]
    async fn invoke(&self, args: Value) -> Result<Value, ToolError> {
        let start = Instant::now();
        let query = required_str(&args, "query")?.to_string();

        let hits = self.nearest_memories(query).await?;
        let output = format_memories(&hits);

        let duration_ms = start.elapsed().as_millis();
        tracing::info!(
            target: "assistd::memory",
            returned = hits.len(),
            duration_ms = duration_ms,
            "recall returned"
        );
        Ok(tool_result(&output, duration_ms))
    }
}

/// Semantic search over past conversations, excluding the session in
/// progress because its dialogue is already in the model's context.
pub struct ReminisceTool {
    embedder: Arc<dyn Embedder>,
    semantic: Arc<dyn SemanticStore>,
    embedding_model: String,
    current_session: watch::Receiver<Arc<SessionId>>,
}

impl ReminisceTool {
    /// A tool ranking past messages in `semantic` by `embedder` vectors,
    /// skipping the session `current_session` names at call time. An empty
    /// `embedding_model` means embedding is disabled.
    pub fn new(
        embedder: Arc<dyn Embedder>,
        semantic: Arc<dyn SemanticStore>,
        embedding_model: String,
        current_session: watch::Receiver<Arc<SessionId>>,
    ) -> Self {
        Self {
            embedder,
            semantic,
            embedding_model,
            current_session,
        }
    }
}

#[async_trait]
impl Tool for ReminisceTool {
    fn name(&self) -> &str {
        "reminisce"
    }

    fn description(&self) -> &str {
        "Search *earlier* conversations — every session except the one \
         in progress, which is already in context — for messages similar \
         in meaning to a query. Complement to `recall`: `recall` looks \
         up *saved facts* (key/value), `reminisce` searches *past \
         dialogue text*; use it when the user references something they \
         'discussed before' or 'worked on last month'. Robust to \
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
    async fn invoke(&self, args: Value) -> Result<Value, ToolError> {
        let start = Instant::now();
        let query = required_str(&args, "query")?.to_string();
        let limit = reminisce_limit(&args)?;
        tracing::Span::current().record("limit", limit);

        if self.embedding_model.is_empty() {
            return Ok(tool_result(
                "(no past conversations indexed)",
                start.elapsed().as_millis(),
            ));
        }
        let query_vec = match self.embedder.embed(query).await {
            Ok(query_vec) => query_vec,
            Err(e) => {
                tracing::debug!(
                    target: "assistd::embed",
                    error = %e,
                    "reminisce embed failed; returning empty"
                );
                return Ok(tool_result(
                    "(embedding unavailable)",
                    start.elapsed().as_millis(),
                ));
            }
        };
        let current = self.current_session.borrow().clone();
        let hits = self
            .semantic
            .nearest_chunks(
                query_vec,
                limit as usize,
                &self.embedding_model,
                Some(&current),
            )
            .await?;

        let output = format_chunks(&hits);
        let duration_ms = start.elapsed().as_millis();
        tracing::info!(
            target: "assistd::embed",
            returned = hits.len(),
            duration_ms = duration_ms,
            "reminisce returned chunks"
        );
        Ok(tool_result(&output, duration_ms))
    }
}

fn format_memories(hits: &[MemoryHit]) -> String {
    if hits.is_empty() {
        return "(no memories)".to_string();
    }
    hits.iter()
        .map(|hit| format!("{}: {}", hit.key, hit.value))
        .collect::<Vec<_>>()
        .join("\n")
}

fn reminisce_limit(args: &Value) -> Result<i64, ToolError> {
    let limit = args
        .get("limit")
        .and_then(Value::as_i64)
        .ok_or_else(|| ToolError::InvalidArgs("`limit` (integer) is required".into()))?;
    if !(1..=20).contains(&limit) {
        return Err(ToolError::InvalidArgs(format!(
            "`limit` must be in 1..=20 (got {limit})"
        )));
    }
    Ok(limit)
}

fn format_chunks(hits: &[EmbeddingHit]) -> String {
    if hits.is_empty() {
        return "(no matches)".to_string();
    }
    hits.iter()
        .map(|hit| {
            format!(
                "[{} {} sim={:.0}%] {}",
                hit.timestamp,
                hit.role.as_wire(),
                hit.similarity * 100.0,
                hit.content.replace('\n', " ")
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn required_str<'a>(args: &'a Value, name: &str) -> Result<&'a str, ToolError> {
    args.get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| ToolError::InvalidArgs(format!("`{name}` (string) is required")))
}

fn tool_result(output: &str, duration_ms: u128) -> Value {
    json!({
        "output":      output,
        "exit_code":   0,
        "duration_ms": duration_ms,
        "truncated":   false,
    })
}

#[cfg(test)]
mod tests {
    use assistd_embed::{EmbedError, NoEmbedder};
    use assistd_memory::{
        ConversationStore, MemoryError, MemoryStore, NoConversationStore, NoMemoryStore,
        NoSemanticStore, SqliteConversationStore, SqliteHandle, SqliteMemoryStore,
    };
    use tempfile::TempDir;
    use tokio::task::JoinHandle;

    use super::*;

    /// A sender whose receiver is dropped, so every `try_send` fails.
    fn closed_embed_tx() -> mpsc::Sender<EmbedJob> {
        let (tx, rx) = mpsc::channel::<EmbedJob>(1);
        drop(rx);
        tx
    }

    fn live_embed_tx() -> (mpsc::Sender<EmbedJob>, mpsc::Receiver<EmbedJob>) {
        mpsc::channel::<EmbedJob>(8)
    }

    fn no_embedder() -> Arc<dyn Embedder> {
        Arc::new(NoEmbedder)
    }

    fn no_semantic() -> Arc<dyn SemanticStore> {
        Arc::new(NoSemanticStore)
    }

    struct FixedEmbedder;

    #[async_trait]
    impl Embedder for FixedEmbedder {
        async fn embed(&self, _text: String) -> Result<Vec<f32>, EmbedError> {
            Ok(vec![1.0])
        }
        fn model(&self) -> &str {
            "m"
        }
        fn dim(&self) -> usize {
            1
        }
    }

    /// Records the session `reminisce` asked to leave out.
    #[derive(Default)]
    struct ExclusionSpy {
        excluded: parking_lot::Mutex<Option<String>>,
    }

    #[async_trait]
    impl SemanticStore for ExclusionSpy {
        async fn nearest_chunks(
            &self,
            _q: Vec<f32>,
            _k: usize,
            _model: &str,
            exclude_session: Option<&SessionId>,
        ) -> Result<Vec<EmbeddingHit>, MemoryError> {
            *self.excluded.lock() = exclude_session.map(|s| s.0.clone());
            Ok(Vec::new())
        }
        async fn nearest_memories(
            &self,
            _q: Vec<f32>,
            _k: usize,
            _model: &str,
        ) -> Result<Vec<MemoryHit>, MemoryError> {
            Ok(Vec::new())
        }
        async fn count_for_model(&self, _model: &str) -> Result<(i64, i64), MemoryError> {
            Ok((0, 0))
        }
        async fn count_stale(&self, _current: &str) -> Result<(i64, Vec<String>), MemoryError> {
            Ok((0, Vec::new()))
        }
        async fn memories_missing_embedding(
            &self,
            _c: &str,
        ) -> Result<Vec<(i64, String)>, MemoryError> {
            Ok(Vec::new())
        }
        async fn chunks_missing_embedding(
            &self,
            _c: &str,
        ) -> Result<Vec<(i64, String)>, MemoryError> {
            Ok(Vec::new())
        }
        async fn store_chunk_embedding(
            &self,
            _chunk_id: i64,
            _model: String,
            _dim: i64,
            _vector: Vec<u8>,
        ) -> Result<(), MemoryError> {
            Ok(())
        }
        async fn store_memory_embedding(
            &self,
            _memory_id: i64,
            _model: String,
            _dim: i64,
            _vector: Vec<u8>,
        ) -> Result<(), MemoryError> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn reminisce_excludes_the_session_in_progress() {
        let spy = Arc::new(ExclusionSpy::default());
        let first = Arc::new(SessionId::new());
        let (session_tx, session_rx) = watch::channel(first.clone());
        let tool = ReminisceTool::new(
            Arc::new(FixedEmbedder),
            spy.clone(),
            "m".to_string(),
            session_rx,
        );

        tool.invoke(json!({"query": "the rust daemon", "limit": 3}))
            .await
            .unwrap();
        assert_eq!(spy.excluded.lock().as_deref(), Some(first.0.as_str()));

        let second = Arc::new(SessionId::new());
        session_tx.send_replace(second.clone());
        tool.invoke(json!({"query": "the rust daemon", "limit": 3}))
            .await
            .unwrap();
        assert_eq!(
            spy.excluded.lock().as_deref(),
            Some(second.0.as_str()),
            "the tool must follow a session switch, not pin its first session"
        );
    }

    /// A SQLite-backed `MemoryOps`; hold the writer and tempdir for the test.
    async fn fresh_ops() -> (Arc<MemoryOps>, JoinHandle<()>, TempDir) {
        let temp = tempfile::tempdir().unwrap();
        let (_tx, rx) = watch::channel(false);
        let (handle, writer) = SqliteHandle::open(&temp.path().join("memory.db"), rx)
            .await
            .unwrap();
        let handle = Arc::new(handle);
        let mem: Arc<dyn MemoryStore> = Arc::new(SqliteMemoryStore::new(handle.clone()));
        let conv: Arc<dyn ConversationStore> = Arc::new(SqliteConversationStore::new(handle));
        (Arc::new(MemoryOps::new(mem, conv)), writer, temp)
    }

    fn no_ops() -> Arc<MemoryOps> {
        Arc::new(MemoryOps::new(
            Arc::new(NoMemoryStore),
            Arc::new(NoConversationStore),
        ))
    }

    fn invalid_args(err: ToolError) -> String {
        match err {
            ToolError::InvalidArgs(msg) => msg,
            other => panic!("expected InvalidArgs, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn remember_saves_key_value() {
        let (ops, _w, _dir) = fresh_ops().await;
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
        let (ops, _w, _dir) = fresh_ops().await;
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
        let tool = RememberTool::new(no_ops(), closed_embed_tx());
        for bad in ["has spaces", "Editor_Pref", "trailing!", "slash/ed", ""] {
            let err = tool
                .invoke(json!({"key": bad, "value": "x"}))
                .await
                .unwrap_err();
            let msg = invalid_args(err);
            assert!(
                msg.starts_with(&format!("`key` must match {KEY_PATTERN} ")),
                "{bad:?}: {msg}"
            );
        }
    }

    #[tokio::test]
    async fn remember_accepts_hyphenated_and_dotted_keys() {
        let (ops, _w, _dir) = fresh_ops().await;
        let tool = RememberTool::new(ops.clone(), closed_embed_tx());
        for key in ["standup.2026-09-11", "project.assistd-tools.dir"] {
            tool.invoke(json!({"key": key, "value": "noted"}))
                .await
                .unwrap_or_else(|e| panic!("{key} should be a valid key: {e}"));
            assert_eq!(ops.load(key).await.unwrap().as_deref(), Some("noted"));
        }
    }

    #[tokio::test]
    async fn remember_rejects_missing_args() {
        let tool = RememberTool::new(no_ops(), closed_embed_tx());
        let err = tool.invoke(json!({})).await.unwrap_err();
        assert_eq!(invalid_args(err), "`key` (string) is required");
        let err = tool.invoke(json!({"key": "user.name"})).await.unwrap_err();
        assert_eq!(invalid_args(err), "`value` (string) is required");
    }

    #[tokio::test]
    async fn remember_enqueues_embed_job_with_value_text() {
        let (ops, _w, _dir) = fresh_ops().await;
        let (etx, mut erx) = live_embed_tx();
        let tool = RememberTool::new(ops.clone(), etx);
        tool.invoke(json!({"key": "editor_preference", "value": "vim is the way"}))
            .await
            .unwrap();
        match erx
            .try_recv()
            .expect("embed job queued before invoke returns")
        {
            EmbedJob::Memory { memory_id, text } => {
                assert!(memory_id > 0, "expected real rowid, got {memory_id}");
                assert_eq!(text, "vim is the way");
            }
            EmbedJob::Chunk { .. } => panic!("expected Memory job, got Chunk"),
        }
    }

    #[tokio::test]
    async fn recall_with_disabled_embedder_returns_no_memories() {
        let tool = RecallTool::new(no_embedder(), no_semantic(), String::new());
        let result = tool
            .invoke(json!({"query": "what editor do I prefer"}))
            .await
            .unwrap();
        assert_eq!(result["output"], "(no memories)");
        assert_eq!(result["exit_code"], 0);
        assert_eq!(result["truncated"], false);
    }

    /// With embedding configured, a failed embed and an empty result
    /// both read as no memories rather than an error.
    #[tokio::test]
    async fn recall_without_hits_returns_no_memories() {
        for (case, embedder) in [
            ("embed fails", no_embedder()),
            ("no hits", Arc::new(FixedEmbedder) as Arc<dyn Embedder>),
        ] {
            let tool = RecallTool::new(embedder, no_semantic(), "m".into());
            let result = tool.invoke(json!({"query": "anything"})).await.unwrap();
            assert_eq!(result["output"], "(no memories)", "{case}");
        }
    }

    #[tokio::test]
    async fn recall_rejects_missing_query() {
        let tool = RecallTool::new(no_embedder(), no_semantic(), String::new());
        let err = tool.invoke(json!({})).await.unwrap_err();
        assert_eq!(invalid_args(err), "`query` (string) is required");
    }

    /// Strict mode requires every declared property to be listed as
    /// required and no others to be accepted.
    #[test]
    fn schemas_satisfy_strict_mode() {
        let tools: [Box<dyn Tool>; 3] = [
            Box::new(RememberTool::new(no_ops(), closed_embed_tx())),
            Box::new(RecallTool::new(no_embedder(), no_semantic(), String::new())),
            Box::new(ReminisceTool::new(
                no_embedder(),
                no_semantic(),
                String::new(),
                watch::channel(Arc::new(SessionId::new())).1,
            )),
        ];
        for tool in tools {
            let schema = tool.parameters_schema();
            let name = tool.name();
            assert_eq!(schema["additionalProperties"], false, "{name}");
            let mut properties: Vec<&str> = schema["properties"]
                .as_object()
                .unwrap_or_else(|| panic!("{name}: properties object"))
                .keys()
                .map(String::as_str)
                .collect();
            let mut required: Vec<&str> = schema["required"]
                .as_array()
                .unwrap_or_else(|| panic!("{name}: required array"))
                .iter()
                .filter_map(Value::as_str)
                .collect();
            properties.sort_unstable();
            required.sort_unstable();
            assert_eq!(required, properties, "{name}");
        }
    }
}
