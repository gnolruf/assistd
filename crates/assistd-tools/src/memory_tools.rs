//! LLM-callable memory tools. `remember` saves and searches key/value
//! facts; `reminisce` searches past dialogue.

use std::borrow::Cow;
use std::sync::{Arc, LazyLock};
use std::time::Instant;

use assistd_embed::{EmbedJob, Embedder, EmbedderHandle, enqueue_embed_job};
use assistd_memory::{EmbeddingHit, EmbeddingTarget, MemoryHit, SemanticStore, SessionId};
use async_trait::async_trait;
use regex::Regex;
use serde_json::{Value, json};
use tokio::sync::{mpsc, watch};

use crate::memory::MemoryOps;
use crate::{Tool, ToolError, ToolOutput};

const SEARCH_LIMIT: usize = 50;

/// Keys reject whitespace and uppercase so one concept keeps one spelling;
/// hyphens allow ISO dates (`standup.2026-09-11`).
const KEY_PATTERN: &str = r"^[a-z0-9._-]+$";
static KEY_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(KEY_PATTERN).expect("KEY_PATTERN compiles"));

/// Saves `(key, value)` facts about the user, queueing each new or changed
/// value for embedding, and searches them by semantic similarity to a query.
#[derive(Debug)]
pub struct RememberTool {
    ops: Arc<MemoryOps>,
    /// Closed when embedding is disabled; the memory still saves unindexed.
    embed_tx: mpsc::Sender<EmbedJob>,
    embedder: Arc<EmbedderHandle>,
    semantic: Arc<dyn SemanticStore>,
}

/// One validated `remember` call.
#[derive(Debug)]
enum RememberRequest {
    Save { key: String, value: String },
    Search { query: String },
}

impl RememberRequest {
    fn parse(args: &Value) -> Result<Self, ToolError> {
        let text = required_str(args, "text")?.to_string();
        match required_str(args, "action")? {
            "save" => Ok(Self::Save {
                key: valid_key(args)?,
                value: text,
            }),
            "search" => Ok(Self::Search { query: text }),
            other => Err(ToolError::InvalidArgs(format!(
                "`action` must be \"save\" or \"search\" (got {other:?})"
            ))),
        }
    }
}

impl RememberTool {
    /// A tool saving through `ops`, queueing each value on `embed_tx`, and
    /// searching memories in `semantic` by vectors from `embedder`.
    pub fn new(
        ops: Arc<MemoryOps>,
        embed_tx: mpsc::Sender<EmbedJob>,
        embedder: Arc<EmbedderHandle>,
        semantic: Arc<dyn SemanticStore>,
    ) -> Self {
        Self {
            ops,
            embed_tx,
            embedder,
            semantic,
        }
    }

    async fn save(&self, key: &str, value: String) -> Result<String, ToolError> {
        let saved = self.ops.save(key, value.clone()).await?;
        if saved.value_changed {
            enqueue_embed_job(
                &self.embed_tx,
                EmbedJob {
                    target: EmbeddingTarget::Memory(saved.id),
                    text: value,
                },
            );
        }
        tracing::info!(target: "assistd::memory", key = %key, "remember saved");
        Ok(format!("remembered {key}"))
    }

    async fn search(&self, query: String) -> Result<String, ToolError> {
        let embedder = match self.embedder.get() {
            Ok(embedder) => embedder,
            Err(e) => return Ok(format!("(saved memories cannot be searched: {e})")),
        };
        let hits = self.nearest_memories(embedder.as_ref(), query).await?;
        tracing::info!(
            target: "assistd::memory",
            returned = hits.len(),
            "remember search returned"
        );
        Ok(format_memories(&hits))
    }

    /// Memories nearest `query`; empty when the embed fails.
    async fn nearest_memories(
        &self,
        embedder: &dyn Embedder,
        query: String,
    ) -> Result<Vec<MemoryHit>, ToolError> {
        match embedder.embed(query).await {
            Ok(query_vec) => Ok(self
                .semantic
                .nearest_memories(query_vec, SEARCH_LIMIT)
                .await?),
            Err(e) => {
                tracing::debug!(
                    target: "assistd::embed",
                    error = %e,
                    "remember search embed failed; returning empty"
                );
                Ok(Vec::new())
            }
        }
    }
}

#[async_trait]
impl Tool for RememberTool {
    fn name(&self) -> &'static str {
        "remember"
    }

    fn description(&self) -> Cow<'_, str> {
        "Save or look up durable memories about the user. \
         action=\"save\": call whenever the user states a stable preference, \
         something about themselves, or anything they want remembered; `key` \
         names the memory and `text` is its value. Saving an existing key \
         overwrites it. Examples: \"I prefer vim over emacs\" -> \
         remember(action=\"save\", key=\"editor_preference\", text=\"vim\"); \
         \"my name is Ben\" -> remember(action=\"save\", key=\"user.name\", \
         text=\"Ben\"). Do NOT save ephemeral context from the current \
         conversation. action=\"search\": call when prior context might help \
         (personalizing an answer, or the user refers to something they told \
         you before); `text` is a natural-language question and `key` is \
         null. Matching is by meaning, so \"what editor do I prefer?\" finds \
         `editor_preference`. Search returns up to 50 `<key>: <value>` \
         lines, or `(no memories)`."
            .into()
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["save", "search"],
                    "description": "\"save\" stores `text` under `key`; \
                                    \"search\" finds saved memories similar \
                                    in meaning to `text`."
                },
                "key": {
                    "type": ["string", "null"],
                    "description": "For save: snake_case identifier with \
                                    optional dot namespacing (e.g. user.name, \
                                    editor_preference, standup.2026-09-11), \
                                    matching ^[a-z0-9._-]+$. For search: null."
                },
                "text": {
                    "type": "string",
                    "description": "For save: the memory or preference to store. \
                                    For search: a natural-language question."
                }
            },
            "required": ["action", "key", "text"]
        })
    }

    #[tracing::instrument(skip(self, args))]
    async fn invoke(&self, args: Value) -> Result<ToolOutput, ToolError> {
        let start = Instant::now();
        let output = match RememberRequest::parse(&args)? {
            RememberRequest::Save { key, value } => self.save(&key, value).await?,
            RememberRequest::Search { query } => self.search(query).await?,
        };
        Ok(ToolOutput::text(output, start.elapsed()))
    }
}

/// Semantic search over past session transcripts, excluding the session in
/// progress because its dialogue is already in the model's context.
#[derive(Debug)]
pub struct ReminisceTool {
    embedder: Arc<EmbedderHandle>,
    semantic: Arc<dyn SemanticStore>,
    current_session: watch::Receiver<SessionId>,
}

impl ReminisceTool {
    /// A tool ranking past messages in `semantic` by vectors from
    /// `embedder`, skipping the session `current_session` names at call time.
    pub fn new(
        embedder: Arc<EmbedderHandle>,
        semantic: Arc<dyn SemanticStore>,
        current_session: watch::Receiver<SessionId>,
    ) -> Self {
        Self {
            embedder,
            semantic,
            current_session,
        }
    }
}

#[async_trait]
impl Tool for ReminisceTool {
    fn name(&self) -> &'static str {
        "reminisce"
    }

    fn description(&self) -> Cow<'_, str> {
        "Search *earlier* conversations — every session except the one \
         in progress, which is already in context — for messages similar \
         in meaning to a query. Complement to `remember`: its search looks \
         up *saved memories* (key/value), `reminisce` searches *past \
         dialogue text*; use it when the user references something they \
         'discussed before' or 'worked on last month'. Robust to \
         paraphrase: a query like \"that rust project we discussed\" will \
         match a past message about \"the assistd embedding daemon in Rust\". \
         Returns up to `limit` ranked snippets with timestamps and roles."
            .into()
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
    async fn invoke(&self, args: Value) -> Result<ToolOutput, ToolError> {
        let start = Instant::now();
        let query = required_str(&args, "query")?.to_string();
        let limit = reminisce_limit(&args)?;
        tracing::Span::current().record("limit", limit);

        let embedder = match self.embedder.get() {
            Ok(embedder) => embedder,
            Err(e) => {
                let note = format!("(past conversations cannot be searched: {e})");
                return Ok(ToolOutput::text(note, start.elapsed()));
            }
        };
        let query_vec = match embedder.embed(query).await {
            Ok(query_vec) => query_vec,
            Err(e) => {
                tracing::debug!(
                    target: "assistd::embed",
                    error = %e,
                    "reminisce embed failed; returning empty"
                );
                return Ok(ToolOutput::text("(embedding unavailable)", start.elapsed()));
            }
        };
        let current = *self.current_session.borrow();
        let hits = self
            .semantic
            .nearest_chunks(query_vec, limit, Some(current))
            .await?;

        let result = ToolOutput::text(format_chunks(&hits), start.elapsed());
        tracing::info!(
            target: "assistd::embed",
            returned = hits.len(),
            duration_ms = result.duration_ms(),
            "reminisce returned chunks"
        );
        Ok(result)
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

fn reminisce_limit(args: &Value) -> Result<usize, ToolError> {
    let limit = args
        .get("limit")
        .and_then(Value::as_i64)
        .ok_or_else(|| ToolError::InvalidArgs("`limit` (integer) is required".into()))?;
    usize::try_from(limit)
        .ok()
        .filter(|limit| (1..=20).contains(limit))
        .ok_or_else(|| ToolError::InvalidArgs(format!("`limit` must be in 1..=20 (got {limit})")))
}

fn format_chunks(hits: &[EmbeddingHit]) -> String {
    if hits.is_empty() {
        return "(no matches)".to_string();
    }
    hits.iter()
        .map(|hit| {
            format!(
                "[{} {} sim={:.0}%] {}",
                hit.timestamp.format("%Y-%m-%dT%H:%M:%SZ"),
                hit.role.as_wire(),
                hit.similarity * 100.0,
                hit.content.replace('\n', " ")
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn valid_key(args: &Value) -> Result<String, ToolError> {
    let key = args.get("key").and_then(Value::as_str).ok_or_else(|| {
        ToolError::InvalidArgs("`key` (string) is required when `action` is \"save\"".into())
    })?;
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
    use assistd_embed::EmbedError;
    use assistd_memory::{
        ChunkId, Embedding, MemoryError, MemoryId, MemoryStore, NoMemoryStore, NoSemanticStore,
        NoTranscriptStore, SqliteHandle, SqliteMemoryStore, SqliteTranscriptStore, TranscriptStore,
        UnindexedRows,
    };
    use assistd_utils::readiness::Readiness;
    use tempfile::TempDir;

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

    fn ready(embedder: Arc<dyn Embedder>) -> Arc<EmbedderHandle> {
        Arc::new(EmbedderHandle::new(Readiness::Ready(embedder)))
    }

    fn no_embedder() -> Arc<EmbedderHandle> {
        Arc::new(EmbedderHandle::new(Readiness::Unavailable(
            "disabled in config".into(),
        )))
    }

    fn no_semantic() -> Arc<dyn SemanticStore> {
        Arc::new(NoSemanticStore)
    }

    #[derive(Debug)]
    struct FixedEmbedder;

    #[async_trait]
    impl Embedder for FixedEmbedder {
        async fn embed(&self, _text: String) -> Result<Vec<f32>, EmbedError> {
            Ok(vec![1.0])
        }
        fn model(&self) -> &'static str {
            "m"
        }
        fn dim(&self) -> usize {
            1
        }
    }

    /// Records the session `reminisce` asked to leave out.
    #[derive(Debug, Default)]
    struct ExclusionSpy {
        excluded: parking_lot::Mutex<Option<SessionId>>,
    }

    #[async_trait]
    impl SemanticStore for ExclusionSpy {
        async fn nearest_chunks(
            &self,
            _q: Vec<f32>,
            _k: usize,
            exclude_session: Option<SessionId>,
        ) -> Result<Vec<EmbeddingHit>, MemoryError> {
            *self.excluded.lock() = exclude_session;
            Ok(Vec::new())
        }
        async fn nearest_memories(
            &self,
            _q: Vec<f32>,
            _k: usize,
        ) -> Result<Vec<MemoryHit>, MemoryError> {
            Ok(Vec::new())
        }
        async fn drop_other_models(&self, _model: &str) -> Result<u64, MemoryError> {
            Ok(0)
        }
        async fn count_missing(&self) -> Result<UnindexedRows, MemoryError> {
            Ok(UnindexedRows::default())
        }
        async fn memories_missing_embedding(
            &self,
            _after: Option<MemoryId>,
            _limit: usize,
        ) -> Result<Vec<(MemoryId, String)>, MemoryError> {
            Ok(Vec::new())
        }
        async fn chunks_missing_embedding(
            &self,
            _after: Option<ChunkId>,
            _limit: usize,
        ) -> Result<Vec<(ChunkId, String)>, MemoryError> {
            Ok(Vec::new())
        }
        async fn store_embeddings(
            &self,
            _model: &str,
            _embeddings: Vec<Embedding>,
        ) -> Result<(), MemoryError> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn reminisce_excludes_the_session_in_progress() {
        let spy = Arc::new(ExclusionSpy::default());
        let first = SessionId::new();
        let (session_tx, session_rx) = watch::channel(first);
        let tool = ReminisceTool::new(ready(Arc::new(FixedEmbedder)), spy.clone(), session_rx);

        tool.invoke(json!({"query": "the rust daemon", "limit": 3}))
            .await
            .unwrap();
        assert_eq!(*spy.excluded.lock(), Some(first));

        let second = SessionId::new();
        session_tx.send_replace(second);
        tool.invoke(json!({"query": "the rust daemon", "limit": 3}))
            .await
            .unwrap();
        assert_eq!(
            *spy.excluded.lock(),
            Some(second),
            "the tool must follow a session switch, not pin its first session"
        );
    }

    /// A SQLite-backed `MemoryOps`; hold the tempdir for the test.
    async fn fresh_ops() -> (Arc<MemoryOps>, TempDir) {
        let temp = tempfile::tempdir().unwrap();
        let handle = SqliteHandle::open(&temp.path().join("memory.db"))
            .await
            .unwrap();
        let mem: Arc<dyn MemoryStore> = Arc::new(SqliteMemoryStore::new(handle.clone()));
        let conv: Arc<dyn TranscriptStore> = Arc::new(SqliteTranscriptStore::new(handle));
        (Arc::new(MemoryOps::new(mem, conv)), temp)
    }

    fn no_ops() -> Arc<MemoryOps> {
        Arc::new(MemoryOps::new(
            Arc::new(NoMemoryStore),
            Arc::new(NoTranscriptStore),
        ))
    }

    fn remember_tool(ops: Arc<MemoryOps>, embed_tx: mpsc::Sender<EmbedJob>) -> RememberTool {
        RememberTool::new(ops, embed_tx, no_embedder(), no_semantic())
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
        let tool = remember_tool(ops.clone(), closed_embed_tx());
        tool.invoke(json!({"action": "save", "key": "editor_preference", "text": "vim"}))
            .await
            .unwrap();
        tool.invoke(json!({"action": "save", "key": "editor_preference", "text": "neovim"}))
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
        let tool = remember_tool(no_ops(), closed_embed_tx());
        for bad in ["has spaces", "Editor_Pref", "trailing!", "slash/ed", ""] {
            let err = tool
                .invoke(json!({"action": "save", "key": bad, "text": "x"}))
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
    async fn remember_save_needs_a_key_but_search_does_not() {
        let tool = remember_tool(no_ops(), closed_embed_tx());
        let err = tool
            .invoke(json!({"action": "save", "key": null, "text": "vim"}))
            .await
            .unwrap_err();
        assert!(invalid_args(err).starts_with("`key` (string) is required"));
        let found = tool
            .invoke(json!({"action": "search", "key": null, "text": "my editor?"}))
            .await
            .unwrap();
        assert!(
            found
                .output
                .starts_with("(saved memories cannot be searched"),
            "{found:?}"
        );
    }

    #[tokio::test]
    async fn remember_enqueues_embed_job_with_value_text() {
        let (ops, _dir) = fresh_ops().await;
        let (etx, mut erx) = live_embed_tx();
        let tool = remember_tool(ops.clone(), etx);
        tool.invoke(
            json!({"action": "save", "key": "editor_preference", "text": "vim is the way"}),
        )
        .await
        .unwrap();
        let job = erx
            .try_recv()
            .expect("embed job queued before invoke returns");
        let EmbeddingTarget::Memory(memory_id) = job.target else {
            panic!("expected a memory job, got {job:?}");
        };
        assert!(
            memory_id > MemoryId(0),
            "expected real rowid, got {memory_id}"
        );
        assert_eq!(job.text, "vim is the way");
    }

    /// Strict mode requires every declared property to be listed as
    /// required and no others to be accepted.
    #[test]
    fn schemas_satisfy_strict_mode() {
        let tools: [Box<dyn Tool>; 2] = [
            Box::new(remember_tool(no_ops(), closed_embed_tx())),
            Box::new(ReminisceTool::new(
                no_embedder(),
                no_semantic(),
                watch::channel(SessionId::new()).1,
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
