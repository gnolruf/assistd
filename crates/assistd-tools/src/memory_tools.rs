//! LLM-callable memory tools. `remember` saves and searches key/value
//! facts; `reminisce` searches past dialogue.

use std::borrow::Cow;
use std::sync::{Arc, LazyLock};
use std::time::Instant;

use assistd_embed::{EmbedJob, Embedder, EmbedderHandle, enqueue_embed_job};
use assistd_memory::{
    ChunkLocator, EmbeddingHit, EmbeddingTarget, MemoryHit, SemanticStore, SessionId, SessionScope,
    TranscriptStore,
};
use async_trait::async_trait;
use regex::Regex;
use serde_json::{Value, json};
use tokio::sync::{mpsc, watch};

use crate::memory::MemoryOps;
use crate::{Tool, ToolError, ToolOutput};

mod exchanges;

use exchanges::Exchanges;

const SEARCH_LIMIT: usize = 50;

/// Byte budget for all of `reminisce`'s hit lines together; lower-ranked hits that
/// would overflow it are dropped.
const REMINISCE_OUTPUT_BYTES: usize = 8 * 1024;

const TIMESTAMP_FORMAT: &str = "%Y-%m-%dT%H:%M:%SZ";

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
    transcripts: Arc<dyn TranscriptStore>,
    current_session: watch::Receiver<SessionId>,
}

/// One validated `reminisce` call.
#[derive(Debug)]
struct ReminisceRequest {
    query: String,
    limit: usize,
    session_prefix: Option<String>,
}

impl ReminisceRequest {
    fn parse(args: &Value) -> Result<Self, ToolError> {
        let session_prefix = match args.get("session") {
            None | Some(Value::Null) => None,
            Some(Value::String(prefix)) if !prefix.is_empty() => Some(prefix.clone()),
            Some(other) => {
                return Err(ToolError::InvalidArgs(format!(
                    "`session` must be a session id from a hit, or null (got {other})"
                )));
            }
        };
        Ok(Self {
            query: required_str(args, "query")?.to_string(),
            limit: reminisce_limit(args)?,
            session_prefix,
        })
    }
}

impl ReminisceTool {
    /// A tool ranking past messages in `semantic` by vectors from `embedder`,
    /// resolving session ids through `transcripts`, and skipping the session
    /// `current_session` names at call time.
    pub fn new(
        embedder: Arc<EmbedderHandle>,
        semantic: Arc<dyn SemanticStore>,
        transcripts: Arc<dyn TranscriptStore>,
        current_session: watch::Receiver<SessionId>,
    ) -> Self {
        Self {
            embedder,
            semantic,
            transcripts,
            current_session,
        }
    }

    /// The sessions to search: every past one, or the one `session_prefix` names.
    /// Errors when the prefix names no session or the session in progress.
    async fn scope(&self, session_prefix: Option<&str>) -> Result<SessionScope, ToolError> {
        let current = *self.current_session.borrow();
        let Some(prefix) = session_prefix else {
            return Ok(SessionScope::AllExcept(current));
        };
        match self.transcripts.resolve_session(prefix).await? {
            None => Err(ToolError::InvalidArgs(format!(
                "no past session has id {prefix:?}"
            ))),
            Some(session) if session == current => Err(ToolError::InvalidArgs(format!(
                "session {prefix:?} is the conversation in progress, already in context"
            ))),
            Some(session) => Ok(SessionScope::Only(session)),
        }
    }

    async fn search(&self, request: ReminisceRequest) -> Result<String, ToolError> {
        let scope = self.scope(request.session_prefix.as_deref()).await?;
        let embedder = match self.embedder.get() {
            Ok(embedder) => embedder,
            Err(e) => return Ok(format!("(past conversations cannot be searched: {e})")),
        };
        let query_vec = match embedder.embed(request.query).await {
            Ok(query_vec) => query_vec,
            Err(e) => {
                tracing::debug!(
                    target: "assistd::embed",
                    error = %e,
                    "reminisce embed failed; returning empty"
                );
                return Ok("(embedding unavailable)".to_string());
            }
        };
        let hits = self
            .semantic
            .nearest_chunks(query_vec, request.limit, scope)
            .await?;
        tracing::info!(
            target: "assistd::embed",
            returned = hits.len(),
            "reminisce returned chunks"
        );
        match scope {
            SessionScope::Only(_) => self.format_exchanges(&hits).await,
            SessionScope::All | SessionScope::AllExcept(_) => Ok(format_chunks(&hits)),
        }
    }

    /// `hits` from one session, each between its neighbouring chunks and merged
    /// with hits it overlaps, under a header naming the session; hits whose
    /// addition would overflow the output budget are dropped with the rest.
    async fn format_exchanges(&self, hits: &[EmbeddingHit]) -> Result<String, ToolError> {
        let Some(first) = hits.first() else {
            return Ok("(no matches in that session)".to_string());
        };
        let header = session_header(&first.locator);
        let mut exchanges = Exchanges::default();
        let mut kept = 0;
        for hit in hits {
            let adjacent = self.semantic.adjacent_chunks(&hit.locator).await?;
            let mut grown = exchanges.clone();
            grown.add(hit, adjacent);
            let grown_len: usize = grown.lines().iter().map(|line| line.len() + 1).sum();
            if header.len() + grown_len > REMINISCE_OUTPUT_BYTES {
                break;
            }
            exchanges = grown;
            kept += 1;
        }
        let mut lines = vec![header];
        lines.extend(exchanges.lines());
        push_omitted_note(&mut lines, hits.len() - kept);
        Ok(lines.join("\n"))
    }
}

#[async_trait]
impl Tool for ReminisceTool {
    fn name(&self) -> &'static str {
        "reminisce"
    }

    fn description(&self) -> Cow<'_, str> {
        "Search *earlier* conversations — every session except the one \
         in progress, which is already in context — for passages similar \
         in meaning to a query. Complement to `remember`: its search looks \
         up *saved memories* (key/value), `reminisce` searches *past \
         dialogue text*; use it when the user references something they \
         'discussed before' or 'worked on last month'. Robust to \
         paraphrase: a query like \"that rust project we discussed\" will \
         match a past passage about \"the assistd embedding daemon in Rust\". \
         Use it in two steps. First search broadly with `session` null: \
         each of up to `limit` ranked snippets names its session's short id \
         and title, then its timestamp, role and the matching excerpt. Then, \
         to read a chat you found, search again with `session` set to that \
         id: hits come only from that chat, each shown with the message \
         before and after it so you see the exchange."
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
                                    this and finds the top-`limit` past passages \
                                    by cosine similarity."
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Maximum number of hits to return (1-20). \
                                    Smaller is sharper; larger casts a wider net."
                },
                "session": {
                    "type": ["string", "null"],
                    "description": "null searches every past chat. A session \
                                    id copied from an earlier hit (e.g. \
                                    3f9a1c2e) searches only that chat and shows \
                                    the messages around each hit."
                }
            },
            "required": ["query", "limit", "session"]
        })
    }

    #[tracing::instrument(skip(self, args), fields(limit = tracing::field::Empty))]
    async fn invoke(&self, args: Value) -> Result<ToolOutput, ToolError> {
        let start = Instant::now();
        let request = ReminisceRequest::parse(&args)?;
        tracing::Span::current().record("limit", request.limit);
        let output = self.search(request).await?;
        Ok(ToolOutput::text(output, start.elapsed()))
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
    let mut lines = Vec::new();
    let mut used = 0;
    for line in hits.iter().map(format_chunk) {
        if used + line.len() > REMINISCE_OUTPUT_BYTES {
            break;
        }
        used += line.len() + 1;
        lines.push(line);
    }
    let omitted = hits.len() - lines.len();
    push_omitted_note(&mut lines, omitted);
    lines.join("\n")
}

fn push_omitted_note(lines: &mut Vec<String>, omitted: usize) {
    if omitted > 0 {
        lines.push(format!("({omitted} lower-ranked matches omitted)"));
    }
}

fn format_chunk(hit: &EmbeddingHit) -> String {
    let ChunkLocator::Message {
        session_id,
        timestamp,
        role,
        session_label,
        ..
    } = &hit.locator;
    format!(
        "[session {} \"{session_label}\" {} {} sim={:.0}%] {}",
        session_id.short(),
        timestamp.format(TIMESTAMP_FORMAT),
        role.as_wire(),
        hit.similarity * 100.0,
        one_line(&hit.text)
    )
}

fn session_header(locator: &ChunkLocator) -> String {
    let ChunkLocator::Message {
        session_id,
        session_label,
        ..
    } = locator;
    format!("session {} \"{session_label}\":", session_id.short())
}

/// The unclosed `[timestamp role` tag that opens a line in a session-scoped result.
fn locator_tag(locator: &ChunkLocator) -> String {
    let ChunkLocator::Message {
        timestamp, role, ..
    } = locator;
    format!("[{} {}", timestamp.format(TIMESTAMP_FORMAT), role.as_wire())
}

fn one_line(text: &str) -> String {
    text.replace('\n', " ")
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
        AdjacentChunks, BranchId, ChunkId, Embedding, MemoryError, MemoryId, MemoryStore,
        NoMemoryStore, NoSemanticStore, NoTranscriptStore, PersistedMessage, SqliteHandle,
        SqliteMemoryStore, SqliteSemanticStore, SqliteTranscriptStore, UnindexedRows,
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

    /// Records the scope `reminisce` searched.
    #[derive(Debug, Default)]
    struct ScopeSpy {
        scope: parking_lot::Mutex<Option<SessionScope>>,
    }

    #[async_trait]
    impl SemanticStore for ScopeSpy {
        async fn nearest_chunks(
            &self,
            _q: Vec<f32>,
            _k: usize,
            scope: SessionScope,
        ) -> Result<Vec<EmbeddingHit>, MemoryError> {
            *self.scope.lock() = Some(scope);
            Ok(Vec::new())
        }
        async fn adjacent_chunks(
            &self,
            _locator: &ChunkLocator,
        ) -> Result<AdjacentChunks, MemoryError> {
            Ok(AdjacentChunks::default())
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
        let spy = Arc::new(ScopeSpy::default());
        let first = SessionId::new();
        let (session_tx, session_rx) = watch::channel(first);
        let tool = ReminisceTool::new(
            ready(Arc::new(FixedEmbedder)),
            spy.clone(),
            Arc::new(NoTranscriptStore),
            session_rx,
        );
        let query = json!({"query": "the rust daemon", "limit": 3, "session": null});

        tool.invoke(query.clone()).await.unwrap();
        assert_eq!(*spy.scope.lock(), Some(SessionScope::AllExcept(first)));

        let second = SessionId::new();
        session_tx.send_replace(second);
        tool.invoke(query).await.unwrap();
        assert_eq!(
            *spy.scope.lock(),
            Some(SessionScope::AllExcept(second)),
            "the tool must follow a session switch, not pin its first session"
        );
    }

    /// A saved session's main branch, for appending dialogue.
    struct PastSession {
        transcripts: Arc<SqliteTranscriptStore>,
        semantic: Arc<SqliteSemanticStore>,
        session: SessionId,
        branch: BranchId,
    }

    impl PastSession {
        async fn begin(handle: &SqliteHandle) -> Self {
            let transcripts = Arc::new(SqliteTranscriptStore::new(handle.clone()));
            let session = SessionId::new();
            let branch = transcripts
                .begin_session_with_main_branch(session, 0)
                .await
                .unwrap();
            Self {
                transcripts,
                semantic: Arc::new(SqliteSemanticStore::new(handle.clone())),
                session,
                branch,
            }
        }

        /// Store one user/assistant exchange, embedding the chunks of whichever
        /// side `embed_reply` picks.
        async fn exchange(&self, prompt: &str, reply: &str, embed_reply: bool) {
            let opened = self
                .transcripts
                .begin_turn(self.branch, PersistedMessage::user(prompt))
                .await
                .unwrap();
            let answered = self
                .transcripts
                .append_message_to_branch(
                    self.branch,
                    opened.turn,
                    PersistedMessage::assistant_text(reply, None),
                )
                .await
                .unwrap();
            let embedded = if embed_reply {
                answered.chunks
            } else {
                opened.message.chunks
            };
            let embeddings = embedded
                .into_iter()
                .map(|(chunk, text)| Embedding {
                    target: EmbeddingTarget::Chunk(chunk),
                    text,
                    vector: vec![1.0],
                })
                .collect();
            self.semantic
                .store_embeddings("m", embeddings)
                .await
                .unwrap();
        }

        fn short_id(&self) -> String {
            self.session.short()
        }
    }

    #[tokio::test]
    async fn reminisce_scoped_to_a_session_reads_only_that_chat_and_never_the_live_one() {
        let temp = tempfile::tempdir().unwrap();
        let handle = SqliteHandle::open(&temp.path().join("memory.db"))
            .await
            .unwrap();
        let past = PastSession::begin(&handle).await;
        past.transcripts
            .set_session_title(past.session, "Daemon tuning")
            .await
            .unwrap();
        past.exchange("how do I tune the daemon?", "raise the batch size", true)
            .await;
        past.exchange("and the voice crate?", "leave it alone", false)
            .await;
        let other = PastSession::begin(&handle).await;
        other
            .exchange("a cooking question", "use more salt", true)
            .await;
        let live = PastSession::begin(&handle).await;
        live.exchange("the live question", "the live answer", true)
            .await;
        let tool = ReminisceTool::new(
            ready(Arc::new(FixedEmbedder)),
            past.semantic.clone(),
            past.transcripts.clone(),
            watch::channel(live.session).1,
        );

        let scoped = tool
            .invoke(json!({"query": "tuning", "limit": 5, "session": past.short_id()}))
            .await
            .unwrap()
            .output;
        let header = format!("session {} \"Daemon tuning\":", past.short_id());
        assert!(scoped.starts_with(&header), "{scoped}");
        let exchange = [
            "how do I tune",
            "raise the batch size",
            "and the voice crate?",
            "leave it alone",
        ];
        for text in exchange {
            assert_eq!(scoped.matches(text).count(), 1, "{text:?} in {scoped}");
        }
        assert_eq!(
            scoped.matches("\n\n").count(),
            1,
            "overlapping hits must merge into one run: {scoped}"
        );
        let order: Vec<usize> = exchange
            .iter()
            .map(|text| {
                scoped
                    .find(text)
                    .unwrap_or_else(|| panic!("{text:?} in {scoped}"))
            })
            .collect();
        assert!(
            order.is_sorted(),
            "the exchange must read in order: {scoped}"
        );
        for absent in ["salt", "live"] {
            assert!(!scoped.contains(absent), "{absent:?} leaked into {scoped}");
        }

        let err = tool
            .invoke(json!({"query": "live", "limit": 5, "session": live.short_id()}))
            .await
            .unwrap_err();
        assert!(invalid_args(err).contains("in progress"));
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
                Arc::new(NoTranscriptStore),
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
