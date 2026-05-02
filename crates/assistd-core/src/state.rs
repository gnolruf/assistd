use crate::{Config, PresenceManager, run_agent_turn};
use anyhow::Result;
use assistd_config::EmbeddingConfig;
use assistd_embed::{EmbedJob, Embedder, NoEmbedder};
use assistd_ipc::{Event, PresenceState, Request, VoiceCaptureState};
use assistd_llm::{LlmBackend, LlmEvent};
use assistd_memory::{
    ChunkingConfig, ConversationStore, MemoryStore, NoConversationStore, NoMemoryStore,
    NoSemanticStore, PersistedMessage, PersistedRole, SemanticStore, SessionId, SqliteHandle,
    TurnId, chunk_message,
};
use assistd_tools::{Attachment, MemoryOps, ToolRegistry};
use assistd_voice::{
    ContinuousListener, SentenceBuffer, SpeakDecision, VoiceInput, VoiceOutputController,
};
use assistd_wm::{NoWindowManager, WindowManager};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, mpsc};
use tracing::Instrument;

/// Truncate `s` to at most `max_chars` *characters* (not bytes), appending
/// an ellipsis when truncated. Walks `char_indices` so we never split a
/// multi-byte character. Used to keep injected context lines short.
fn truncate_for_context(s: &str, max_chars: usize) -> String {
    let total = s.chars().count();
    if total <= max_chars {
        return s.replace('\n', " ");
    }
    let cutoff = s
        .char_indices()
        .nth(max_chars)
        .map(|(b, _)| b)
        .unwrap_or(s.len());
    let mut head = s[..cutoff].replace('\n', " ");
    head.push('…');
    head
}

/// Convert wire-level [`assistd_ipc::ImageAttachment`] entries into the
/// internal [`Attachment`] type the agent loop expects. Returns the first
/// decode error so the caller can surface it cleanly to the client.
fn decode_wire_attachments(
    wire: &[assistd_ipc::ImageAttachment],
) -> std::result::Result<Vec<Attachment>, String> {
    wire.iter()
        .map(|w| {
            let bytes = w
                .decode_bytes()
                .map_err(|e| format!("base64 decode failed for {}: {e}", w.mime))?;
            Ok(Attachment::Image {
                mime: w.mime.clone(),
                bytes,
            })
        })
        .collect()
}

/// Shared, long-lived daemon state handed to every request handler.
///
/// Subsystem handles (tools, voice, window manager, memory) are expected
/// to be `Arc`-held and cheaply cloneable (actor façades over background
/// tasks) so that multiple concurrent requests can all hold
/// `Arc<AppState>` without contention.
pub struct AppState {
    pub config: Config,
    pub llm: Arc<dyn LlmBackend>,
    pub presence: Arc<PresenceManager>,
    pub tools: Arc<ToolRegistry>,
    pub voice: Arc<dyn VoiceInput>,
    pub listener: Arc<dyn ContinuousListener>,
    /// Speaks LLM responses aloud, sentence by sentence, through a
    /// runtime controller that owns the toggle/skip/interrupt policy.
    /// When TTS is disabled in config or the build, the inner backend
    /// is `NoVoiceOutput` which silently accepts every speak() call.
    pub voice_output: Arc<VoiceOutputController>,
    /// Re-probes `/props` at the top of each query so a model swap on
    /// the running llama-server flips the vision gate. `None` in tests
    /// where there is no real llama-server to probe — `handle_query`
    /// then skips the revalidation step entirely.
    pub vision_revalidator: Option<Arc<crate::VisionRevalidator>>,
    /// Persistent key/value memory shared across sessions. Wired to
    /// [`assistd_memory::NoMemoryStore`] in builds without a backend
    /// configured; Milestone 4 swaps in the SQLite-backed concrete
    /// implementation. Held as `Arc<dyn MemoryStore>` so the trait
    /// boundary stays the same regardless of backend.
    pub memory: Arc<dyn MemoryStore>,
    /// Persistent conversation history (sessions, turns, messages, FTS).
    /// Sibling to [`Self::memory`]; both bind to a `NoConversationStore`
    /// placeholder when memory is disabled in config.
    pub conversations: Arc<dyn ConversationStore>,
    /// Combined CRUD façade exposing both the KV [`Self::memory`] and
    /// the richer [`Self::conversations`] under one handle. Used by the
    /// IPC dispatch arms for `Request::Memory*`.
    pub memory_ops: Arc<MemoryOps>,
    /// Identifier for this daemon process's session, set at startup by
    /// [`assistd_memory::ConversationStore::begin_session`]. Every row
    /// the persistence hook writes gets tagged with it so a future
    /// `assistd memory reminisce` knows which daemon run produced it.
    pub session_id: Arc<SessionId>,
    /// Embedder used for query-side embedding (`inject_semantic_context`)
    /// and for the LLM-callable `recall` / `reminisce` tools. The
    /// background embedder task that processes [`Self::embed_tx`] holds
    /// its own clone of this `Arc`.
    pub embedder: Arc<dyn Embedder>,
    /// Top-K vector retrieval over `embeddings` (chunks) and
    /// `memory_embeddings` (memories). Wired to `NoSemanticStore` when
    /// the embedding subsystem is disabled or fails to start.
    pub semantic: Arc<dyn SemanticStore>,
    /// Bounded queue feeding the embedder task. `try_send` from the
    /// persistence hook so a wedged embedder never backpressures
    /// streaming. When the subsystem is disabled, the receiver is
    /// dropped and `try_send` no-ops with a warn (acceptable — an
    /// unindexed chunk can be picked up by a future backfill).
    pub embed_tx: mpsc::Sender<EmbedJob>,
    /// Direct handle to the SQLite store, used solely for
    /// [`SqliteHandle::store_chunk`] from the persistence hook.
    /// `None` when the memory subsystem is disabled (`NoConversationStore`
    /// short-circuits before chunking would matter).
    pub chunks: Option<Arc<SqliteHandle>>,
    /// Snapshot of `[embedding]` config so the persistence hook can
    /// build a `ChunkingConfig` without re-reading the top-level
    /// `Config` struct on every message.
    pub embedding_cfg: EmbeddingConfig,
    /// Compositor / window-manager backend. Wired to
    /// [`assistd_wm::NoWindowManager`] when no compositor backend is
    /// configured or the configured backend failed to connect at
    /// startup; otherwise a concrete backend (e.g. `I3Backend`) speaks
    /// the compositor's IPC protocol.
    pub window_manager: Arc<dyn WindowManager>,
    /// Serializes entire agent turns. Concurrent queries each grab this
    /// lock before running `run_agent_turn`, so one query's tool-call /
    /// tool-result cycle never interleaves with another's — which would
    /// otherwise leave the backend's conversation state with dangling
    /// `tool_calls` messages.
    agent_turn_lock: Arc<Mutex<()>>,
}

impl AppState {
    pub fn new(
        config: Config,
        llm: Arc<dyn LlmBackend>,
        presence: Arc<PresenceManager>,
        tools: Arc<ToolRegistry>,
        voice: Arc<dyn VoiceInput>,
        listener: Arc<dyn ContinuousListener>,
        voice_output: Arc<VoiceOutputController>,
    ) -> Self {
        let memory: Arc<dyn MemoryStore> = Arc::new(NoMemoryStore);
        let conversations: Arc<dyn ConversationStore> = Arc::new(NoConversationStore);
        let memory_ops = Arc::new(MemoryOps::new(memory.clone(), conversations.clone()));
        let embedding_cfg = config.embedding.clone();
        // Bounded with capacity 1 + dropped receiver = closed channel.
        // Every `try_send` will fail; the persistence hook degrades to
        // chunk-only behavior, which is what we want when the embedding
        // subsystem isn't wired in.
        let (embed_tx, embed_rx) = mpsc::channel::<EmbedJob>(1);
        drop(embed_rx);
        Self {
            config,
            llm,
            presence,
            tools,
            voice,
            listener,
            voice_output,
            vision_revalidator: None,
            memory,
            conversations,
            memory_ops,
            session_id: Arc::new(SessionId::new()),
            embedder: Arc::new(NoEmbedder),
            semantic: Arc::new(NoSemanticStore),
            embed_tx,
            chunks: None,
            embedding_cfg,
            window_manager: Arc::new(NoWindowManager),
            agent_turn_lock: Arc::new(Mutex::new(())),
        }
    }

    /// Builder-style setter for the optional vision revalidator. Kept
    /// out of [`Self::new`] so existing test constructors don't need to
    /// supply it.
    pub fn with_vision_revalidator(mut self, r: Arc<crate::VisionRevalidator>) -> Self {
        self.vision_revalidator = Some(r);
        self
    }

    /// Builder-style setter for the memory backend. Kept out of
    /// [`Self::new`] so existing test constructors keep working with
    /// the default `NoMemoryStore`. Milestone 4's SQLite store will
    /// be threaded through this method.
    pub fn with_memory(mut self, m: Arc<dyn MemoryStore>) -> Self {
        self.memory = m.clone();
        // Keep memory_ops in sync so downstream dispatch arms see the
        // SQLite-backed store too. `MemoryOps` is just two `Arc`s — a
        // cheap rebuild beats forcing every caller to remember to set
        // both fields.
        self.memory_ops = Arc::new(MemoryOps::new(m, self.conversations.clone()));
        self
    }

    /// Builder-style setter for the conversation store. Mirrors
    /// [`Self::with_memory`] — required when wiring the SQLite backend
    /// at daemon startup.
    pub fn with_conversations(mut self, c: Arc<dyn ConversationStore>) -> Self {
        self.conversations = c.clone();
        self.memory_ops = Arc::new(MemoryOps::new(self.memory.clone(), c));
        self
    }

    /// Override the daemon's session id. Set at startup once
    /// [`assistd_memory::ConversationStore::begin_session`] has
    /// returned the persisted uuid; tests skip this and inherit the
    /// auto-generated default.
    pub fn with_session(mut self, s: Arc<SessionId>) -> Self {
        self.session_id = s;
        self
    }

    /// Builder-style setter for the embedder. Defaults to `NoEmbedder`
    /// so tests and fallback paths still satisfy the trait bound.
    pub fn with_embedder(mut self, e: Arc<dyn Embedder>) -> Self {
        self.embedder = e;
        self
    }

    /// Builder-style setter for the semantic-search store.
    pub fn with_semantic(mut self, s: Arc<dyn SemanticStore>) -> Self {
        self.semantic = s;
        self
    }

    /// Builder-style setter for the embed-job sender. The daemon
    /// supplies a live channel wired to the embedder task; tests can
    /// inherit the closed sentinel from `new()`.
    pub fn with_embed_tx(mut self, tx: mpsc::Sender<EmbedJob>) -> Self {
        self.embed_tx = tx;
        self
    }

    /// Builder-style setter for the SQLite handle, needed by the
    /// persistence hook to send `WriteOp::StoreChunk`. Daemon startup
    /// passes the same `Arc` it builds the conversation/memory stores
    /// from; tests leave this as `None`.
    pub fn with_chunks(mut self, h: Arc<SqliteHandle>) -> Self {
        self.chunks = Some(h);
        self
    }

    /// Builder-style setter for the embedding subsystem config. Default
    /// is `Config::default().embedding` (matches the daemon's startup
    /// path); tests can override to drive the persistence hook with a
    /// disabled or differently-tuned config.
    pub fn with_embedding_cfg(mut self, cfg: EmbeddingConfig) -> Self {
        self.embedding_cfg = cfg;
        self
    }

    /// Builder-style setter for the window-manager backend. Defaults to
    /// `NoWindowManager` so existing test constructors keep working;
    /// the daemon supplies a concrete backend (e.g. `I3Backend`) when
    /// the configured compositor is reachable at startup.
    pub fn with_window_manager(mut self, wm: Arc<dyn WindowManager>) -> Self {
        self.window_manager = wm;
        self
    }

    /// Route a single incoming request to the appropriate subsystem.
    ///
    /// Events are streamed back through `tx`. When this function returns
    /// (either `Ok` or `Err`), no further events will be sent. The caller
    /// is responsible for surfacing `Err` to the client as an
    /// [`Event::Error`] if one hasn't been emitted already.
    pub async fn dispatch(self: Arc<Self>, req: Request, tx: mpsc::Sender<Event>) -> Result<()> {
        match req {
            Request::Query {
                id,
                text,
                attachments,
                version,
            } => {
                if let Some(v) = version {
                    if v > assistd_ipc::PROTOCOL_VERSION {
                        tracing::warn!(
                            client = v,
                            server = assistd_ipc::PROTOCOL_VERSION,
                            "client sent newer protocol version than daemon understands; \
                             extra fields will be ignored"
                        );
                    }
                } else {
                    tracing::debug!("client did not send protocol version; treating as legacy v1");
                }
                self.handle_query(id, text, attachments, tx).await
            }
            Request::SetPresence { id, target } => self.handle_set_presence(id, target, tx).await,
            Request::GetPresence { id } => self.handle_get_presence(id, tx).await,
            Request::Cycle { id } => self.handle_cycle(id, tx).await,
            Request::PttStart { id } => self.handle_ptt_start(id, tx).await,
            Request::PttStop { id } => self.handle_ptt_stop(id, tx).await,
            Request::ListenStart { id } => self.handle_listen_start(id, tx).await,
            Request::ListenStop { id } => self.handle_listen_stop(id, tx).await,
            Request::ListenToggle { id } => self.handle_listen_toggle(id, tx).await,
            Request::GetListenState { id } => self.handle_get_listen_state(id, tx).await,
            Request::VoiceToggle { id } => self.handle_voice_toggle(id, tx).await,
            Request::VoiceSkip { id } => self.handle_voice_skip(id, tx).await,
            Request::GetVoiceState { id } => self.handle_get_voice_state(id, tx).await,
            Request::MemorySave { id, key, value } => {
                self.handle_memory_save(id, key, value, tx).await
            }
            Request::MemoryLoad { id, key } => self.handle_memory_load(id, key, tx).await,
            Request::MemoryList { id, prefix } => self.handle_memory_list(id, prefix, tx).await,
            Request::MemoryListAll { id, prefix, limit } => {
                self.handle_memory_list_all(id, prefix, limit, tx).await
            }
            Request::MemoryDelete { id, key } => self.handle_memory_delete(id, key, tx).await,
            Request::MemoryForget { id, memory_id } => {
                self.handle_memory_forget(id, memory_id, tx).await
            }
            Request::MemorySemanticSearch { id, query, limit } => {
                self.handle_memory_semantic_search(id, query, limit, tx)
                    .await
            }
            Request::MemoryReindex { id } => self.handle_memory_reindex(id, tx).await,
            Request::GetCapabilities { id } => self.handle_get_capabilities(id, tx).await,
            // ConfirmResponse is intercepted by the connection-level
            // read loop (see `assistd-core/src/socket.rs`) and routed to
            // the active confirmation gate's pending oneshot. If we see
            // it as the *initial* request on a fresh connection there
            // is no in-flight prompt to satisfy — surface as an error
            // so a buggy client gets a clear signal instead of timing out.
            Request::ConfirmResponse { id, confirm_id, .. } => {
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!(
                            "ConfirmResponse(confirm_id={confirm_id}) received with no \
                             matching ConfirmRequest in flight on this connection"
                        ),
                    })
                    .await;
                Ok(())
            }
        }
    }

    /// Fire-and-forget persist of one message. Spawns a task that:
    /// 1. Writes the row via the conversation store (returns row id).
    /// 2. If the row is a User/Assistant text message and embedding is
    ///    enabled, splits the content into chunks, persists each chunk
    ///    (returns chunk id), and `try_send`s an `EmbedJob::Chunk` for
    ///    each so the embedder task can index it.
    ///
    /// The whole pipeline is on a `tokio::spawn`'d task so the dispatch
    /// loop never waits on disk or the embed queue. `try_send` (not
    /// `send`) so a wedged embedder doesn't backpressure persistence —
    /// dropped jobs just leave the chunk row unindexed for the next
    /// backfill pass.
    fn persist_message_fire_and_forget(&self, turn: Option<TurnId>, msg: PersistedMessage) {
        let conv = self.conversations.clone();
        let session = self.session_id.clone();
        let chunks_handle = self.chunks.clone();
        let embed_tx = self.embed_tx.clone();
        let embedding_enabled = self.embedding_cfg.enabled;
        let chunking_cfg = ChunkingConfig {
            chunk_chars: self.embedding_cfg.chunk_chars,
            overlap_chars: self.embedding_cfg.chunk_overlap_chars,
        };
        // Snapshot what we need for chunking *before* moving `msg` into
        // append_message. Tool-call assistant rows and tool-result rows
        // skip chunking: the former have empty content, the latter are
        // JSON-y noise that pollutes semantic search.
        let should_embed = embedding_enabled
            && chunks_handle.is_some()
            && matches!(msg.role, PersistedRole::User | PersistedRole::Assistant)
            && !msg.content.is_empty()
            && msg.tool_calls.is_none();
        let content_for_chunks = if should_embed {
            Some(msg.content.clone())
        } else {
            None
        };
        tokio::spawn(async move {
            let row_id = match conv.append_message(&session, turn, msg).await {
                Ok(id) => id,
                Err(e) => {
                    tracing::warn!(
                        target: "assistd::memory",
                        error = %e,
                        "failed to persist message (continuing)"
                    );
                    return;
                }
            };
            // NoConversationStore returns 0 — nothing to chunk.
            let Some(content) = content_for_chunks else {
                return;
            };
            let Some(chunks_handle) = chunks_handle else {
                return;
            };
            if row_id == 0 {
                return;
            }
            for (idx, chunk) in chunk_message(&content, &chunking_cfg)
                .into_iter()
                .enumerate()
            {
                match chunks_handle
                    .store_chunk(row_id, idx as i64, chunk.clone(), None)
                    .await
                {
                    Ok(chunk_id) => {
                        if embed_tx
                            .try_send(EmbedJob::Chunk {
                                chunk_id,
                                text: chunk,
                            })
                            .is_err()
                        {
                            tracing::debug!(
                                target: "assistd::embed",
                                chunk_id,
                                "embed queue full or closed; dropping job"
                            );
                        }
                    }
                    Err(e) => tracing::warn!(
                        target: "assistd::memory",
                        conversation_id = row_id,
                        chunk_index = idx,
                        error = %e,
                        "failed to persist chunk (continuing)"
                    ),
                }
            }
        });
    }

    /// Embed the user query, find the top-K nearest conversation chunks,
    /// and return a "Relevant past context: …" block to inject as a
    /// transient system message. Returns `Ok(None)` when retrieval is a
    /// no-op (short query, NoEmbedder, no hits).
    ///
    /// Best-effort: errors propagate but the caller treats every failure
    /// (embedder down, dim mismatch, …) as "skip injection".
    async fn build_semantic_context(&self, query: &str) -> Result<Option<String>> {
        // Skip very short queries — no useful semantic signal in 1-2
        // chars, and the cost of a wasted embed call is non-zero.
        if query.trim().chars().count() < 3 {
            return Ok(None);
        }
        let vec = self.embedder.embed(query.to_string()).await?;
        let model = self.embedder.model().to_string();
        if model.is_empty() {
            // NoEmbedder path: model() == "" → never matches stored
            // rows. Bail without a write.
            return Ok(None);
        }
        let top_k = self.embedding_cfg.top_k as usize;
        let hits = self.semantic.nearest_chunks(vec, top_k, &model).await?;
        if hits.is_empty() {
            return Ok(None);
        }
        let mut block = String::from("Relevant past context:\n");
        for h in hits {
            let snippet = truncate_for_context(&h.content, 200);
            // One line per hit; format keeps the model-visible block
            // compact and predictable. `as_wire()` is "user"/"assistant".
            block.push_str(&format!(
                "- [{} {} sim={:.0}%] {}\n",
                h.timestamp,
                h.role.as_wire(),
                h.similarity * 100.0,
                snippet
            ));
        }
        Ok(Some(block))
    }

    /// Format the focused-window snapshot as a `Current desktop context`
    /// block for the LLM's per-turn transient system message. Returns
    /// `None` when the window manager has no opinion (no compositor
    /// connected, nothing focused, all fields empty).
    ///
    /// Errors from the WM backend degrade silently to `None` so a flaky
    /// compositor never blocks the user's turn.
    async fn build_window_context(&self) -> Option<String> {
        let ctx = match self.window_manager.focused_context().await {
            Ok(Some(c)) => c,
            Ok(None) => return None,
            Err(e) => {
                tracing::debug!(
                    target: "assistd::context",
                    error = %e,
                    "WindowManager::focused_context failed; skipping window context",
                );
                return None;
            }
        };
        format_window_context_block(&ctx)
    }
}

/// Render a [`assistd_wm::FocusedWindowContext`] into the prompt
/// fragment injected as a transient system message. Pure (no async,
/// no `&self`) so it's directly unit-testable. Returns `None` when
/// every field is empty.
fn format_window_context_block(ctx: &assistd_wm::FocusedWindowContext) -> Option<String> {
    if ctx.class.is_none() && ctx.title.is_none() && ctx.workspace.is_none() {
        return None;
    }
    let mut block = String::from("Current desktop context:\n");
    let class_for_line = ctx.class.as_deref();
    let title_for_line = ctx.title.as_deref();
    if class_for_line.is_some() || title_for_line.is_some() {
        match (class_for_line, title_for_line) {
            (Some(c), Some(t)) => block.push_str(&format!("- Focused window: {c} — {t}\n")),
            (Some(c), None) => block.push_str(&format!("- Focused window: {c}\n")),
            (None, Some(t)) => block.push_str(&format!("- Focused window: (unknown) — {t}\n")),
            (None, None) => unreachable!(),
        }
    }
    if let Some(ws) = ctx.workspace.as_deref() {
        block.push_str(&format!("- Workspace: {ws}\n"));
    }
    let is_term = ctx
        .class
        .as_deref()
        .map(assistd_wm::is_terminal_class)
        .unwrap_or(false);
    let kind = if is_term { "terminal" } else { "non-terminal" };
    block.push_str(&format!("The user is interacting with a {kind} window."));
    if is_term {
        // Bias the LLM toward in-place execution. Phrasing references
        // the actual `run` tool's sub-command names so the model maps
        // the hint cleanly onto its tool schema.
        block.push_str(
            " If the user asks to run a command, build, or test, prefer calling `run` \
             with `command: \"bash\"` (executing the command in this terminal context) over \
             launching a new terminal via `run` with `command: \"wm\"`.",
        );
    }
    Some(block)
}

/// Concatenate the semantic and window context blocks with a blank
/// line between them. Returns `None` only when both inputs are `None`.
/// Pure / unit-testable.
fn combine_context_blocks(semantic: Option<String>, window: Option<String>) -> Option<String> {
    match (semantic, window) {
        (None, None) => None,
        (Some(s), None) => Some(s),
        (None, Some(w)) => Some(w),
        (Some(s), Some(w)) => Some(format!("{}\n{}", s.trim_end(), w)),
    }
}

impl AppState {
    async fn handle_memory_semantic_search(
        self: Arc<Self>,
        id: String,
        query: String,
        limit: u32,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        let model = self.embedder.model().to_string();
        if model.is_empty() {
            // Embedding subsystem disabled — emit Done with no hits so
            // the CLI sees a clean empty stream, not an error.
            let _ = tx.send(Event::Done { id }).await;
            return Ok(());
        }
        let limit = if limit == 0 {
            assistd_tools::DEFAULT_SEARCH_LIMIT
        } else {
            limit as usize
        };
        let vec = match self.embedder.embed(query).await {
            Ok(v) => v,
            Err(e) => {
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("embed failed: {e:#}"),
                    })
                    .await;
                return Err(e);
            }
        };
        match self.semantic.nearest_chunks(vec, limit, &model).await {
            Ok(hits) => {
                for h in hits {
                    let _ = tx
                        .send(Event::SemanticHit {
                            id: id.clone(),
                            conversation_id: h.conversation_id,
                            chunk_id: h.chunk_id,
                            session_id: h.session_id,
                            timestamp: h.timestamp,
                            role: h.role.as_wire().into(),
                            content: h.content,
                            similarity: h.similarity,
                        })
                        .await;
                }
                let _ = tx.send(Event::Done { id }).await;
                Ok(())
            }
            Err(e) => {
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("semantic search failed: {e:#}"),
                    })
                    .await;
                Err(e)
            }
        }
    }

    async fn handle_memory_save(
        self: Arc<Self>,
        id: String,
        key: String,
        value: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        match self.memory_ops.save(&key, value).await {
            // The IPC `MemorySave` handler doesn't surface the row id —
            // its only callers are the `assistd memory save` CLI which
            // just wants confirmation. Discard the id; the LLM tool
            // (`RememberTool`) is the one that consumes it directly.
            Ok(_id) => {
                let _ = tx.send(Event::Done { id }).await;
                Ok(())
            }
            Err(e) => {
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("memory save failed: {e:#}"),
                    })
                    .await;
                Err(e)
            }
        }
    }

    async fn handle_memory_load(
        self: Arc<Self>,
        id: String,
        key: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        match self.memory_ops.load(&key).await {
            Ok(value) => {
                let _ = tx
                    .send(Event::MemoryValue {
                        id: id.clone(),
                        key,
                        value,
                    })
                    .await;
                let _ = tx.send(Event::Done { id }).await;
                Ok(())
            }
            Err(e) => {
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("memory load failed: {e:#}"),
                    })
                    .await;
                Err(e)
            }
        }
    }

    async fn handle_memory_list(
        self: Arc<Self>,
        id: String,
        prefix: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        match self.memory_ops.list(&prefix).await {
            Ok(keys) => {
                let _ = tx
                    .send(Event::MemoryKeys {
                        id: id.clone(),
                        keys,
                    })
                    .await;
                let _ = tx.send(Event::Done { id }).await;
                Ok(())
            }
            Err(e) => {
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("memory list failed: {e:#}"),
                    })
                    .await;
                Err(e)
            }
        }
    }

    async fn handle_memory_delete(
        self: Arc<Self>,
        id: String,
        key: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        match self.memory_ops.delete(&key).await {
            Ok(()) => {
                let _ = tx.send(Event::Done { id }).await;
                Ok(())
            }
            Err(e) => {
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("memory delete failed: {e:#}"),
                    })
                    .await;
                Err(e)
            }
        }
    }

    async fn handle_memory_list_all(
        self: Arc<Self>,
        id: String,
        prefix: String,
        limit: u32,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        match self.memory_ops.list_full(&prefix).await {
            Ok(rows) => {
                // `limit = 0` means no cap (matches the wire convention
                // used by other search-style memory variants). Otherwise
                // truncate after the SQL has already ordered
                // lexicographically by key.
                let cap = if limit == 0 {
                    rows.len()
                } else {
                    rows.len().min(limit as usize)
                };
                for row in rows.into_iter().take(cap) {
                    let _ = tx
                        .send(Event::MemoryRow {
                            id: id.clone(),
                            memory_id: row.id,
                            key: row.key,
                            value: row.value,
                        })
                        .await;
                }
                let _ = tx.send(Event::Done { id }).await;
                Ok(())
            }
            Err(e) => {
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("memory list_all failed: {e:#}"),
                    })
                    .await;
                Err(e)
            }
        }
    }

    async fn handle_memory_forget(
        self: Arc<Self>,
        id: String,
        memory_id: i64,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        match self.memory_ops.forget(memory_id).await {
            Ok(removed) => {
                let _ = tx
                    .send(Event::MemoryForgetResult {
                        id: id.clone(),
                        deleted: removed.is_some(),
                        key: removed,
                    })
                    .await;
                let _ = tx.send(Event::Done { id }).await;
                Ok(())
            }
            Err(e) => {
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("memory forget failed: {e:#}"),
                    })
                    .await;
                Err(e)
            }
        }
    }

    /// Embed every memory and every conversation chunk that has no
    /// embedding under the daemon's currently-configured model. Streams
    /// `ReindexProgress` events as items complete; finishes with `Done`
    /// (or `Error` if no embedder is configured). Per-item failures
    /// during embed/write are logged and counted as still-done so a
    /// single bad row doesn't wedge the whole run — same log-and-drop
    /// posture as the background embedder task.
    async fn handle_memory_reindex(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        let model = self.embedder.model().to_string();
        if model.is_empty() {
            let _ = tx
                .send(Event::Error {
                    id,
                    message: "embedding subsystem disabled; cannot reindex".to_string(),
                })
                .await;
            return Ok(());
        }
        let dim = self.embedder.dim() as i64;

        let chunks = match self.semantic.chunks_missing_embedding(&model).await {
            Ok(v) => v,
            Err(e) => {
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("reindex: list missing chunks: {e:#}"),
                    })
                    .await;
                return Err(e);
            }
        };
        let memories = match self.semantic.memories_missing_embedding(&model).await {
            Ok(v) => v,
            Err(e) => {
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("reindex: list missing memories: {e:#}"),
                    })
                    .await;
                return Err(e);
            }
        };
        let chunks_total = chunks.len() as u32;
        let memories_total = memories.len() as u32;

        // Always emit a kickoff progress for each kind so a client that
        // only sees `(0, 0)` still learns the totals before receiving
        // `Done` — useful when the run has nothing to do.
        let _ = tx
            .send(Event::ReindexProgress {
                id: id.clone(),
                kind: "chunks".to_string(),
                done: 0,
                total: chunks_total,
            })
            .await;
        let _ = tx
            .send(Event::ReindexProgress {
                id: id.clone(),
                kind: "memories".to_string(),
                done: 0,
                total: memories_total,
            })
            .await;

        let mut done = 0u32;
        for (chunk_id, text) in chunks {
            match self.embedder.embed(text).await {
                Ok(vec) => {
                    let blob = assistd_memory::vector_to_blob(&vec);
                    if let Err(e) = self
                        .semantic
                        .store_chunk_embedding(chunk_id, model.clone(), dim, blob)
                        .await
                    {
                        tracing::warn!(
                            target: "assistd::memory",
                            chunk_id,
                            error = %e,
                            "reindex: store_chunk_embedding failed"
                        );
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        target: "assistd::memory",
                        chunk_id,
                        error = %e,
                        "reindex: embed chunk failed"
                    );
                }
            }
            done = done.saturating_add(1);
            let _ = tx
                .send(Event::ReindexProgress {
                    id: id.clone(),
                    kind: "chunks".to_string(),
                    done,
                    total: chunks_total,
                })
                .await;
        }

        let mut done = 0u32;
        for (memory_id, value) in memories {
            match self.embedder.embed(value).await {
                Ok(vec) => {
                    let blob = assistd_memory::vector_to_blob(&vec);
                    if let Err(e) = self
                        .semantic
                        .store_memory_embedding(memory_id, model.clone(), dim, blob)
                        .await
                    {
                        tracing::warn!(
                            target: "assistd::memory",
                            memory_id,
                            error = %e,
                            "reindex: store_memory_embedding failed"
                        );
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        target: "assistd::memory",
                        memory_id,
                        error = %e,
                        "reindex: embed memory failed"
                    );
                }
            }
            done = done.saturating_add(1);
            let _ = tx
                .send(Event::ReindexProgress {
                    id: id.clone(),
                    kind: "memories".to_string(),
                    done,
                    total: memories_total,
                })
                .await;
        }

        let _ = tx.send(Event::Done { id }).await;
        Ok(())
    }

    pub async fn handle_query(
        self: Arc<Self>,
        id: String,
        text: String,
        wire_attachments: Vec<assistd_ipc::ImageAttachment>,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        // Re-probe llama-server's loaded model so a runtime model swap
        // (e.g. a fresh `/models/load`) flips the vision gate before
        // either this turn's image attachments OR the agent's
        // see/screenshot tool calls run against a stale capability.
        // Always called — the probe is a single local HTTP GET with a
        // 2s timeout, and only mutates the gate on actual change.
        if let Some(rev) = self.vision_revalidator.as_ref() {
            rev.revalidate().await;
        }
        // Decode wire attachments into internal Attachment values up
        // front so a malformed base64 payload fails before we wake the
        // model — saves the user the GPU round-trip on a bad request.
        let attachments: Vec<Attachment> = match decode_wire_attachments(&wire_attachments) {
            Ok(v) => v,
            Err(e) => {
                let _ = tx
                    .send(Event::Error {
                        id: id.clone(),
                        message: format!("invalid attachment: {e}"),
                    })
                    .await;
                return Err(anyhow::anyhow!("invalid attachment: {e}"));
            }
        };
        // Auto-wake and take an in-flight guard: a query in any
        // non-Active state blocks here until the daemon is ready to
        // serve. The returned guard keeps the daemon `Active` for the
        // lifetime of the generation — a concurrent `sleep()` will
        // wait until this guard (and the generator that follows) has
        // finished. Failures surface to the client as an Error event
        // so the client doesn't hang.
        let _request_guard = match self.presence.acquire_request_guard().await {
            Ok(g) => g,
            Err(e) => {
                let _ = tx
                    .send(Event::Error {
                        id: id.clone(),
                        message: format!("wake failed: {e:#}"),
                    })
                    .await;
                return Err(e);
            }
        };
        // Separately mark the LLM as streaming so voice transcription
        // can detect GPU contention and queue/fall back to CPU. This is
        // a no-block counter — sleep/drowse still block on
        // `_request_guard` via the inflight RwLock.
        let _stream_guard = self.presence.acquire_stream_guard();

        // Serialize agent turns. Concurrent queries each wait here so
        // one turn's assistant/tool_calls/tool_result triplet lands in
        // conversation state atomically — otherwise a second query
        // could push its own user message between this turn's steps.
        let _agent_guard = self.agent_turn_lock.clone().lock_owned().await;

        // Persistence: open a turn now that we hold the agent guard.
        // begin_turn lands a row in `turns`; a `None` return means the
        // backend is `NoConversationStore` and we silently skip
        // persistence for the rest of the turn. We await the oneshot
        // here because every following persistence call needs the
        // resulting `TurnId` — but it's a single channel hop, not a
        // disk write of the streaming response.
        let turn_id: Option<TurnId> =
            match self.conversations.begin_turn(&self.session_id, &text).await {
                Ok(t) if t.0 != 0 => Some(t),
                // NoConversationStore returns TurnId(0); treat that as "no
                // persistence configured" without logging.
                Ok(_) => None,
                Err(e) => {
                    tracing::warn!(
                        target: "assistd::memory",
                        error = %e,
                        "begin_turn failed; turn will not be persisted"
                    );
                    None
                }
            };
        // Mirror the user's prompt into the conversations table on the
        // way to the LLM — fire-and-forget so the dispatch loop below
        // never waits on disk.
        self.persist_message_fire_and_forget(turn_id, PersistedMessage::user(text.clone()));

        // Auto-inject the LLM's per-turn transient system message. Two
        // sources, composed at the call site because the conversation's
        // transient slot is a single `Option<String>` (a second
        // `set_transient_context` call would clobber the first):
        //   1. Semantic recall — top-K conversation chunks (gated on
        //      embedding config, like before).
        //   2. Window context — focused class/title/workspace from the
        //      WM event task. Always-on; degrades to None when no
        //      compositor is connected. Carries a terminal-bias hint
        //      when the focused class is a known terminal emulator.
        // Both are best-effort: any failure debug-logs and continues.
        let semantic = if self.embedding_cfg.enabled && self.embedding_cfg.auto_inject {
            match self.build_semantic_context(&text).await {
                Ok(b) => b,
                Err(e) => {
                    tracing::debug!(
                        target: "assistd::embed",
                        error = %e,
                        "semantic context injection failed; continuing without it",
                    );
                    None
                }
            }
        } else {
            None
        };
        let window = self.build_window_context().await;
        if let Some(block) = combine_context_blocks(semantic, window)
            && let Err(e) = self.llm.set_transient_context(block).await
        {
            tracing::debug!(
                target: "assistd::context",
                error = %e,
                "set_transient_context failed; continuing without it",
            );
        }

        let (llm_tx, mut llm_rx) = mpsc::channel::<LlmEvent>(32);
        let llm = self.llm.clone();
        let tools = self.tools.clone();
        let max_iterations = self.config.agent.max_iterations;
        // Cancellation token: wired to the agent loop so a slow LLM
        // step or tool dispatch can be preempted promptly. Today we
        // only fire it when this function returns (drop) — that
        // unblocks the agent task if the dispatch loop below bailed
        // mid-stream. Future MCP work will also fire it on explicit
        // shutdown signals.
        let cancel = tokio_util::sync::CancellationToken::new();
        let cancel_for_agent = cancel.clone();
        // Drop guard: when this function returns for any reason
        // (including early bail from the dispatch loop below), the
        // guard cancels the token, ensuring the agent task wakes up
        // and exits even if it's parked in `backend.step`.
        let _cancel_on_return = cancel.clone().drop_guard();
        let generator = tokio::spawn(
            async move {
                run_agent_turn(
                    llm,
                    tools,
                    max_iterations,
                    text,
                    attachments,
                    llm_tx,
                    cancel_for_agent,
                )
                .await
            }
            .in_current_span(),
        );

        // Sentence buffer + speech worker. Each completed sentence is
        // sent to a per-query worker task that calls voice_output.speak
        // serially. Speak returns post-enqueue (not post-playback), so
        // the worker keeps Piper's stdout pipeline ahead of the audio
        // queue and there's no audible gap between utterances.
        //
        // Buffer = 32 ≈ 2 minutes of queued speech — absorbs Piper
        // hiccups without backpressuring the IPC client wire forward.
        let synthesis = &self.config.voice.synthesis;
        let max_sentence_chars = synthesis.max_sentence_chars as usize;
        let code_block_mode = synthesis.code_block_mode;
        let partial_flush_ms = synthesis.partial_flush_ms;
        let mut sentence_buf = SentenceBuffer::new_with_mode(max_sentence_chars, code_block_mode);

        let (speech_tx, mut speech_rx) = mpsc::channel::<String>(32);
        let ctrl = self.voice_output.clone();
        // Capture the current skip-epoch once. Any later `skip()` (or
        // `interrupt()`) advances the global epoch and the worker drops
        // every remaining sentence for THIS query without affecting
        // future queries spawned with the new epoch.
        let start_epoch = ctrl.current_epoch();
        let speech_handle = tokio::spawn(
            async move {
                while let Some(sentence) = speech_rx.recv().await {
                    match ctrl.should_speak(start_epoch) {
                        SpeakDecision::Speak => {
                            if let Err(e) = ctrl.inner().speak(sentence).await {
                                tracing::debug!(
                                    target: "assistd::voice",
                                    error = %e,
                                    "voice_output.speak failed (non-fatal)"
                                );
                            }
                        }
                        SpeakDecision::DropSilent | SpeakDecision::DropForSkip => {
                            // TTS toggled off, or skipped — drain the channel
                            // without speaking. We still loop so the LLM can
                            // keep streaming through the bounded mpsc.
                        }
                    }
                }
                // Channel closed: drain anything still in the audio queue
                // before the worker returns. Any failure here is logged
                // inside wait_idle and downgraded to Ok.
                if let Err(e) = ctrl.inner().wait_idle().await {
                    tracing::debug!(
                        target: "assistd::voice",
                        error = %e,
                        "voice_output.wait_idle failed (non-fatal)"
                    );
                }
            }
            .in_current_span(),
        );

        // Tool calls inhibit the idle-flush — the LLM is *waiting on a
        // tool*, not stalled mid-prose. Without this, a long-running
        // bash command would trigger spurious mid-sentence flushes.
        let mut awaiting_tool_result = false;
        let partial_flush = if partial_flush_ms > 0 {
            Some(Duration::from_millis(partial_flush_ms as u64))
        } else {
            None
        };

        // Persistence accumulator: tee every `LlmEvent::Delta { text }`
        // into this string so the assistant's full reply gets written
        // as one row at end-of-turn. Cleared on each Tool/Done flush.
        let mut assistant_accum = String::new();

        let mut client_alive = true;

        loop {
            let llm_event = match (partial_flush, awaiting_tool_result) {
                (Some(d), false) => match tokio::time::timeout(d, llm_rx.recv()).await {
                    Ok(Some(ev)) => ev,
                    Ok(None) => break,
                    Err(_) => {
                        // Idle timeout: flush partial sentence (if any)
                        // without disturbing fence state.
                        if let Some(partial) = sentence_buf.flush_idle()
                            && speech_tx.send(partial).await.is_err()
                        {
                            tracing::debug!(
                                target: "assistd::voice",
                                "speech worker channel closed; dropping idle flush"
                            );
                        }
                        continue;
                    }
                },
                _ => match llm_rx.recv().await {
                    Some(ev) => ev,
                    None => break,
                },
            };

            // Build the wire event. Sentence-buffer pushes happen
            // here too — but the wire event is sent BEFORE any
            // speech_tx.send below, so client display latency stays
            // independent of TTS backpressure.
            let (wire_event, sentences_to_speak): (Event, Vec<String>) = match llm_event {
                LlmEvent::Delta { text } => {
                    let sentences = sentence_buf.push(&text);
                    // Persistence tee — accumulate the assistant's
                    // full reply for one row at Done.
                    assistant_accum.push_str(&text);
                    (
                        Event::Delta {
                            id: id.clone(),
                            text,
                        },
                        sentences,
                    )
                }
                LlmEvent::ToolCall {
                    id: call_id,
                    name,
                    arguments,
                } => {
                    awaiting_tool_result = true;
                    // Flush any narration we'd accumulated so the
                    // pre-tool-call assistant text doesn't get glued
                    // onto the tool result downstream. Then write the
                    // assistant_with_tool_calls row mirroring the
                    // in-memory `push_assistant_with_tool_calls` shape.
                    if !assistant_accum.is_empty() {
                        let pre_text = std::mem::take(&mut assistant_accum);
                        self.persist_message_fire_and_forget(
                            turn_id,
                            PersistedMessage::assistant_text(pre_text),
                        );
                    }
                    let calls_json = serde_json::json!([{
                        "id":        call_id,
                        "name":      name,
                        "arguments": arguments,
                    }]);
                    self.persist_message_fire_and_forget(
                        turn_id,
                        PersistedMessage::assistant_tool_calls(calls_json),
                    );
                    (
                        Event::ToolCall {
                            id: id.clone(),
                            name,
                            args: arguments,
                        },
                        Vec::new(),
                    )
                }
                LlmEvent::ToolResult {
                    id: call_id,
                    name,
                    result,
                } => {
                    awaiting_tool_result = false;
                    // Persist the tool-result row. We extract the
                    // human-readable `output` field if present
                    // (RunTool's shape) and fall back to the raw JSON.
                    let body = result
                        .get("output")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| result.to_string());
                    self.persist_message_fire_and_forget(
                        turn_id,
                        PersistedMessage::tool_result(body, call_id, name.clone()),
                    );
                    (
                        Event::ToolResult {
                            id: id.clone(),
                            name,
                            result,
                        },
                        Vec::new(),
                    )
                }
                LlmEvent::Done => {
                    let tail = sentence_buf.finish();
                    let sentences = tail.into_iter().collect();
                    // Final assistant row: whatever text accumulated
                    // since the last tool-call flush.
                    if !assistant_accum.is_empty() {
                        let final_text = std::mem::take(&mut assistant_accum);
                        self.persist_message_fire_and_forget(
                            turn_id,
                            PersistedMessage::assistant_text(final_text),
                        );
                    }
                    (Event::Done { id: id.clone() }, sentences)
                }
            };

            // Forward wire event first.
            if client_alive && tx.send(wire_event).await.is_err() {
                client_alive = false;
            }

            // Then enqueue speech (in arrival order). Channel send
            // failure means the worker died — log and continue; we
            // still want to forward remaining wire events.
            for s in sentences_to_speak {
                if speech_tx.send(s).await.is_err() {
                    tracing::debug!(
                        target: "assistd::voice",
                        "speech worker channel closed; dropping sentence"
                    );
                    break;
                }
            }

            if !client_alive {
                // Client gone: stop pulling from llm_rx so the agent
                // loop's tx.is_closed() check fires at its next
                // iteration boundary. The worker keeps draining
                // queued audio.
                break;
            }
        }

        // Stop the speech pipeline. Dropping speech_tx tells the worker
        // to consume the remainder of its channel and then run
        // wait_idle() — playback finishes naturally.
        drop(speech_tx);

        // Wait for the agent turn to finish so we know whether to
        // surface a backend error. The agent_turn_lock can be released
        // as soon as the turn ends — concurrent queries shouldn't wait
        // for audio to finish playing.
        let gen_result = generator.await;
        drop(_agent_guard);

        // Close out the persisted turn whether we exited cleanly,
        // hit an LLM error, or the client disconnected mid-stream —
        // every path lands an `ended_at` timestamp on the row.
        // Fire-and-forget like the message persistence above.
        if let Some(t) = turn_id {
            let conv = self.conversations.clone();
            tokio::spawn(async move {
                if let Err(e) = conv.end_turn(t).await {
                    tracing::warn!(
                        target: "assistd::memory",
                        error = %e,
                        "end_turn failed (continuing)"
                    );
                }
            });
        }

        // Now await the speech worker. _request_guard is still held so
        // presence stays Active through playback.
        let _ = speech_handle.await;

        match gen_result {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => {
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("llm backend error: {e}"),
                    })
                    .await;
                Err(e)
            }
            Err(join_err) => {
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("llm backend panicked: {join_err}"),
                    })
                    .await;
                Err(anyhow::anyhow!("llm backend panicked: {join_err}"))
            }
        }
    }

    async fn handle_set_presence(
        self: Arc<Self>,
        id: String,
        target: PresenceState,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        match self.presence.set_presence(target).await {
            Ok(()) => {
                let _ = tx
                    .send(Event::Presence {
                        id: id.clone(),
                        state: self.presence.state(),
                    })
                    .await;
                let _ = tx.send(Event::Done { id }).await;
                Ok(())
            }
            Err(e) => {
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("set_presence failed: {e:#}"),
                    })
                    .await;
                Err(e)
            }
        }
    }

    async fn handle_get_presence(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        let state = self.presence.state();
        let _ = tx
            .send(Event::Presence {
                id: id.clone(),
                state,
            })
            .await;
        let _ = tx.send(Event::Done { id }).await;
        Ok(())
    }

    async fn handle_cycle(self: Arc<Self>, id: String, tx: mpsc::Sender<Event>) -> Result<()> {
        match self.presence.cycle().await {
            Ok(new_state) => {
                let _ = tx
                    .send(Event::Presence {
                        id: id.clone(),
                        state: new_state,
                    })
                    .await;
                let _ = tx.send(Event::Done { id }).await;
                Ok(())
            }
            Err(e) => {
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("cycle failed: {e:#}"),
                    })
                    .await;
                Err(e)
            }
        }
    }

    /// Begin a push-to-talk recording. Returns immediately on success
    /// so the CLI client exits cleanly — the daemon holds the capture
    /// session open until a matching `PttStop` arrives on a separate
    /// connection. Rejects when continuous listening currently owns
    /// the mic; the two modes can't share one cpal stream.
    async fn handle_ptt_start(self: Arc<Self>, id: String, tx: mpsc::Sender<Event>) -> Result<()> {
        // Barge-in: stop any in-flight TTS playback before opening the
        // mic. Fire unconditionally — even if recording is rejected
        // below, the user signaled "shut up", which we should honor.
        self.voice_output.interrupt().await;
        if self.listener.is_active() {
            let _ = tx
                .send(Event::Error {
                    id,
                    message: "continuous listening is active; disable it before using PTT".into(),
                })
                .await;
            return Ok(());
        }
        match self.voice.start_recording().await {
            Ok(()) => {
                let _ = tx
                    .send(Event::VoiceState {
                        id: id.clone(),
                        state: VoiceCaptureState::Recording,
                    })
                    .await;
                let _ = tx.send(Event::Done { id }).await;
                Ok(())
            }
            Err(e) => {
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("ptt_start failed: {e:#}"),
                    })
                    .await;
                Err(e)
            }
        }
    }

    /// Enable continuous listening. Rejects if PTT currently holds the
    /// mic — the two modes are mutually exclusive because cpal cannot
    /// share one input stream.
    async fn handle_listen_start(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        if self.voice.state() != VoiceCaptureState::Idle {
            let _ = tx
                .send(Event::Error {
                    id,
                    message: "cannot start continuous listening while PTT is recording".into(),
                })
                .await;
            return Ok(());
        }
        match self.listener.start().await {
            Ok(()) => {
                let _ = tx
                    .send(Event::ListenState {
                        id: id.clone(),
                        active: true,
                    })
                    .await;
                let _ = tx.send(Event::Done { id }).await;
                Ok(())
            }
            Err(e) => {
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("listen_start failed: {e:#}"),
                    })
                    .await;
                Err(e)
            }
        }
    }

    /// Disable continuous listening. Idempotent.
    async fn handle_listen_stop(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        match self.listener.stop().await {
            Ok(()) => {
                let _ = tx
                    .send(Event::ListenState {
                        id: id.clone(),
                        active: false,
                    })
                    .await;
                let _ = tx.send(Event::Done { id }).await;
                Ok(())
            }
            Err(e) => {
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("listen_stop failed: {e:#}"),
                    })
                    .await;
                Err(e)
            }
        }
    }

    /// Flip continuous listening state. Routes to start or stop
    /// depending on the current value of `listener.is_active()`.
    async fn handle_listen_toggle(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        if self.listener.is_active() {
            self.handle_listen_stop(id, tx).await
        } else {
            self.handle_listen_start(id, tx).await
        }
    }

    /// Report whether continuous listening is currently active.
    async fn handle_get_listen_state(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        let active = self.listener.is_active();
        let _ = tx
            .send(Event::ListenState {
                id: id.clone(),
                active,
            })
            .await;
        let _ = tx.send(Event::Done { id }).await;
        Ok(())
    }

    /// Flip TTS on/off at runtime. Off cancels currently-queued audio;
    /// on is a pure flag flip. Subsequent sentences from the in-flight
    /// query speak again as soon as the toggle returns. Emits the
    /// post-toggle `VoiceOutputState` + `Done`.
    async fn handle_voice_toggle(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        let new_state = !self.voice_output.enabled();
        self.voice_output.set_enabled(new_state).await;
        let _ = tx
            .send(Event::VoiceOutputState {
                id: id.clone(),
                enabled: new_state,
            })
            .await;
        let _ = tx.send(Event::Done { id }).await;
        Ok(())
    }

    /// Abort the current TTS response: drops queued audio and any
    /// pending sentences for active speech workers. Does not start
    /// recording. Emits `VoiceOutputState` + `Done`; the enabled flag
    /// is unchanged.
    async fn handle_voice_skip(self: Arc<Self>, id: String, tx: mpsc::Sender<Event>) -> Result<()> {
        self.voice_output.skip().await;
        let _ = tx
            .send(Event::VoiceOutputState {
                id: id.clone(),
                enabled: self.voice_output.enabled(),
            })
            .await;
        let _ = tx.send(Event::Done { id }).await;
        Ok(())
    }

    /// Report whether TTS is currently enabled.
    async fn handle_get_voice_state(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        let _ = tx
            .send(Event::VoiceOutputState {
                id: id.clone(),
                enabled: self.voice_output.enabled(),
            })
            .await;
        let _ = tx.send(Event::Done { id }).await;
        Ok(())
    }

    /// Probe the running llama-server's capabilities and surface the
    /// model name in one shot — lets clients render `vision: on/off`
    /// without reaching into the HTTP API directly. Re-probes per
    /// request because a model swap on a long-lived daemon can flip
    /// the vision flag between calls.
    async fn handle_get_capabilities(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        let probe = assistd_llm::probe_capabilities(
            &self.config.llama_server.host,
            self.config.llama_server.port,
        )
        .await;
        // Match the TUI's old basename rendering: prefer the part after
        // the last `/` (e.g. `Qwen3-14B-GGUF:Q4_K_M` from
        // `bartowski/Qwen3-14B-GGUF:Q4_K_M`), fall back to the full
        // string when there's no `/`.
        let model_name = self
            .config
            .model
            .name
            .rsplit_once('/')
            .map(|(_, rest)| rest.to_string())
            .unwrap_or_else(|| self.config.model.name.clone());
        let _ = tx
            .send(Event::Capabilities {
                id: id.clone(),
                vision: probe.vision_supported,
                model_name,
            })
            .await;
        let _ = tx.send(Event::Done { id }).await;
        Ok(())
    }

    /// End the push-to-talk recording, transcribe, and — if the
    /// transcription has content — dispatch it internally as a Query
    /// so the streaming LLM response flows back on the same
    /// connection before the terminal `Done`.
    async fn handle_ptt_stop(self: Arc<Self>, id: String, tx: mpsc::Sender<Event>) -> Result<()> {
        let _ = tx
            .send(Event::VoiceState {
                id: id.clone(),
                state: VoiceCaptureState::Transcribing,
            })
            .await;

        let text = match self.voice.stop_and_transcribe().await {
            Ok(t) => t,
            Err(e) => {
                let _ = tx
                    .send(Event::VoiceState {
                        id: id.clone(),
                        state: VoiceCaptureState::Idle,
                    })
                    .await;
                let _ = tx
                    .send(Event::Error {
                        id,
                        message: format!("ptt_stop failed: {e:#}"),
                    })
                    .await;
                return Err(e);
            }
        };

        let _ = tx
            .send(Event::VoiceState {
                id: id.clone(),
                state: VoiceCaptureState::Idle,
            })
            .await;
        let _ = tx
            .send(Event::Transcription {
                id: id.clone(),
                text: text.clone(),
            })
            .await;

        if text.trim().is_empty() {
            // VAD trimmed to silence — end the stream here rather
            // than dispatching an empty user message to the LLM.
            let _ = tx.send(Event::Done { id }).await;
            return Ok(());
        }

        // Auto-feed the agent loop, reusing the Query handler so
        // streaming deltas, tool calls, and Done all land on the
        // same connection and correlate to this request's id.
        self.handle_query(id, text, Vec::new(), tx).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Config;
    use assistd_config::ToolsOutputConfig;
    use assistd_llm::{EchoBackend, FailedBackend, StepOutcome, ToolCall, ToolResultPayload};
    use assistd_tools::{CommandRegistry, RunTool, commands::EchoCommand};
    use std::sync::Mutex as StdMutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn test_state(backend: Arc<dyn LlmBackend>, initial_state: PresenceState) -> Arc<AppState> {
        Arc::new(AppState::new(
            Config::default(),
            backend,
            PresenceManager::stub(initial_state),
            Arc::new(ToolRegistry::default()),
            Arc::new(assistd_voice::NoVoiceInput::new()),
            Arc::new(assistd_voice::NoContinuousListener::new()),
            VoiceOutputController::new(Arc::new(assistd_voice::NoVoiceOutput), true),
        ))
    }

    fn state_with_voice(
        backend: Arc<dyn LlmBackend>,
        voice: Arc<dyn assistd_voice::VoiceInput>,
    ) -> Arc<AppState> {
        Arc::new(AppState::new(
            Config::default(),
            backend,
            PresenceManager::stub(PresenceState::Active),
            Arc::new(ToolRegistry::default()),
            voice,
            Arc::new(assistd_voice::NoContinuousListener::new()),
            VoiceOutputController::new(Arc::new(assistd_voice::NoVoiceOutput), true),
        ))
    }

    fn state_with_listener(
        backend: Arc<dyn LlmBackend>,
        listener: Arc<dyn assistd_voice::ContinuousListener>,
    ) -> Arc<AppState> {
        Arc::new(AppState::new(
            Config::default(),
            backend,
            PresenceManager::stub(PresenceState::Active),
            Arc::new(ToolRegistry::default()),
            Arc::new(assistd_voice::NoVoiceInput::new()),
            listener,
            VoiceOutputController::new(Arc::new(assistd_voice::NoVoiceOutput), true),
        ))
    }

    async fn collect_events(mut rx: mpsc::Receiver<Event>) -> Vec<Event> {
        let mut out = Vec::new();
        while let Some(ev) = rx.recv().await {
            out.push(ev);
        }
        out
    }

    #[tokio::test]
    async fn dispatch_query_emits_delta_then_done() {
        let state = test_state(Arc::new(EchoBackend::new()), PresenceState::Active);
        let (tx, rx) = mpsc::channel::<Event>(8);
        let req = Request::Query {
            id: "q1".into(),
            text: "hello".into(),
            attachments: Vec::new(),
            version: None,
        };

        state.dispatch(req, tx).await.unwrap();
        let events = collect_events(rx).await;

        assert_eq!(events.len(), 2, "expected Delta+Done, got {events:?}");
        assert!(matches!(
            &events[0],
            Event::Delta { id, text } if id == "q1" && text == "hello"
        ));
        assert!(matches!(&events[1], Event::Done { id } if id == "q1"));
    }

    #[tokio::test]
    async fn dispatch_set_presence_emits_presence_and_done() {
        // Active → Sleeping avoids hitting the stub's non-existent
        // control-plane endpoint while still exercising the transition
        // path end-to-end.
        let state = test_state(Arc::new(EchoBackend::new()), PresenceState::Active);
        let (tx, rx) = mpsc::channel::<Event>(8);
        let req = Request::SetPresence {
            id: "p1".into(),
            target: PresenceState::Sleeping,
        };

        state.clone().dispatch(req, tx).await.unwrap();
        let events = collect_events(rx).await;

        assert_eq!(events.len(), 2, "expected Presence+Done, got {events:?}");
        assert!(matches!(
            &events[0],
            Event::Presence { id, state: PresenceState::Sleeping } if id == "p1"
        ));
        assert!(matches!(&events[1], Event::Done { id } if id == "p1"));
        assert_eq!(state.presence.state(), PresenceState::Sleeping);
    }

    #[tokio::test]
    async fn dispatch_get_presence_reports_current_state_without_transition() {
        let state = test_state(Arc::new(EchoBackend::new()), PresenceState::Drowsy);
        let (tx, rx) = mpsc::channel::<Event>(8);
        let req = Request::GetPresence { id: "g1".into() };

        state.clone().dispatch(req, tx).await.unwrap();
        let events = collect_events(rx).await;

        assert_eq!(events.len(), 2, "expected Presence+Done, got {events:?}");
        assert!(matches!(
            &events[0],
            Event::Presence { id, state: PresenceState::Drowsy } if id == "g1"
        ));
        assert!(matches!(&events[1], Event::Done { id } if id == "g1"));
        // State must be unchanged — GetPresence is a read-only snapshot.
        assert_eq!(state.presence.state(), PresenceState::Drowsy);
    }

    #[tokio::test]
    async fn dispatch_cycle_advances_to_next_state() {
        // Drowsy → Sleeping: the Drowsy→Sleeping branch of `sleep()` is
        // network-free when `llama` is None (as in the stub), so this
        // exercises the full cycle path without a live server.
        let state = test_state(Arc::new(EchoBackend::new()), PresenceState::Drowsy);
        let (tx, rx) = mpsc::channel::<Event>(8);
        let req = Request::Cycle { id: "c1".into() };

        state.clone().dispatch(req, tx).await.unwrap();
        let events = collect_events(rx).await;

        assert_eq!(events.len(), 2, "expected Presence+Done, got {events:?}");
        assert!(matches!(
            &events[0],
            Event::Presence { id, state: PresenceState::Sleeping } if id == "c1"
        ));
        assert!(matches!(&events[1], Event::Done { id } if id == "c1"));
        assert_eq!(state.presence.state(), PresenceState::Sleeping);
    }

    #[tokio::test]
    async fn dispatch_query_backend_error_emits_error_event() {
        let backend = Arc::new(FailedBackend::new("backend broken".into()));
        let state = test_state(backend, PresenceState::Active);
        let (tx, rx) = mpsc::channel::<Event>(8);
        let req = Request::Query {
            id: "q-err".into(),
            text: "boom".into(),
            attachments: Vec::new(),
            version: None,
        };

        let err = state.dispatch(req, tx).await.unwrap_err();
        assert!(err.to_string().contains("backend broken"));

        let events = collect_events(rx).await;
        let err_event = events
            .iter()
            .find(|e| matches!(e, Event::Error { .. }))
            .expect("expected Error event in stream");
        match err_event {
            Event::Error { id, message } => {
                assert_eq!(id, "q-err");
                assert!(
                    message.contains("backend broken"),
                    "error message should propagate backend reason: {message}"
                );
            }
            _ => unreachable!(),
        }
    }

    /// MockBackend that scripts StepOutcomes and records pushed tool
    /// results. Same shape as the one in `agent::tests` but re-declared
    /// here so we can wire it into AppState and exercise IPC mapping.
    struct ScriptedBackend {
        outcomes: std::sync::Mutex<Vec<StepOutcome>>,
    }

    #[async_trait::async_trait]
    impl LlmBackend for ScriptedBackend {
        async fn generate(
            &self,
            _prompt: String,
            _tx: mpsc::Sender<LlmEvent>,
        ) -> anyhow::Result<()> {
            unimplemented!("uses step path")
        }
        async fn push_user(
            &self,
            _text: String,
            _attachments: Vec<assistd_tools::Attachment>,
        ) -> anyhow::Result<()> {
            Ok(())
        }
        async fn push_tool_results(&self, _results: Vec<ToolResultPayload>) -> anyhow::Result<()> {
            Ok(())
        }
        async fn step(
            &self,
            _tools: Vec<serde_json::Value>,
            _tx: mpsc::Sender<LlmEvent>,
        ) -> anyhow::Result<StepOutcome> {
            let outcome = {
                let mut q = self.outcomes.lock().unwrap();
                if q.is_empty() {
                    StepOutcome::Final
                } else {
                    q.remove(0)
                }
            };
            Ok(outcome)
        }
    }

    fn state_with_echo_tools(backend: Arc<dyn LlmBackend>) -> Arc<AppState> {
        let mut commands = CommandRegistry::new();
        commands.register(EchoCommand);
        let mut tools = ToolRegistry::new();
        tools.register(RunTool::new(
            Arc::new(commands),
            &ToolsOutputConfig::default(),
            std::env::temp_dir().join(format!("assistd-state-test-{}", std::process::id())),
        ));
        Arc::new(AppState::new(
            Config::default(),
            backend,
            PresenceManager::stub(PresenceState::Active),
            Arc::new(tools),
            Arc::new(assistd_voice::NoVoiceInput::new()),
            Arc::new(assistd_voice::NoContinuousListener::new()),
            VoiceOutputController::new(Arc::new(assistd_voice::NoVoiceOutput), true),
        ))
    }

    #[tokio::test]
    async fn dispatch_query_forwards_tool_call_and_result_events() {
        // One tool call, then Final. We expect the forwarder to map
        // LlmEvent::ToolCall → Event::ToolCall (with the request id),
        // LlmEvent::ToolResult → Event::ToolResult, then Done.
        let backend = Arc::new(ScriptedBackend {
            outcomes: std::sync::Mutex::new(vec![
                StepOutcome::ToolCalls(vec![ToolCall {
                    id: "call-opaque".into(),
                    name: "run".into(),
                    arguments: serde_json::json!({"command": "echo hi"}),
                }]),
                StepOutcome::Final,
            ]),
        });
        let state = state_with_echo_tools(backend);
        let (tx, rx) = mpsc::channel::<Event>(16);
        let req = Request::Query {
            id: "req-42".into(),
            text: "go".into(),
            attachments: Vec::new(),
            version: None,
        };
        state.dispatch(req, tx).await.unwrap();

        let events = collect_events(rx).await;

        let tool_call = events
            .iter()
            .find(|e| matches!(e, Event::ToolCall { .. }))
            .expect("expected Event::ToolCall in stream");
        match tool_call {
            Event::ToolCall { id, name, args } => {
                // IPC id is the *request* id, not the LLM's call id.
                assert_eq!(id, "req-42");
                assert_eq!(name, "run");
                assert_eq!(args["command"], "echo hi");
            }
            _ => unreachable!(),
        }

        let tool_result = events
            .iter()
            .find(|e| matches!(e, Event::ToolResult { .. }))
            .expect("expected Event::ToolResult in stream");
        match tool_result {
            Event::ToolResult { id, name, result } => {
                assert_eq!(id, "req-42");
                assert_eq!(name, "run");
                // RunTool's echo produces "hi\n" with a success footer.
                assert!(
                    result["output"]
                        .as_str()
                        .map(|s| s.contains("hi") && s.contains("[exit:0"))
                        .unwrap_or(false),
                    "expected echo output in result: {result}"
                );
            }
            _ => unreachable!(),
        }

        assert!(
            matches!(events.last(), Some(Event::Done { id }) if id == "req-42"),
            "expected terminal Done: {events:?}"
        );
    }

    /// Mock VoiceInput driven by a script of canned start/stop
    /// outcomes, used to exercise the PttStart / PttStop handlers
    /// without touching cpal or whisper.
    struct MockVoice {
        start_result: std::sync::Mutex<Option<anyhow::Result<()>>>,
        stop_result: std::sync::Mutex<Option<anyhow::Result<String>>>,
        state_tx: tokio::sync::watch::Sender<assistd_voice::VoiceCaptureState>,
    }

    impl MockVoice {
        fn new(start: anyhow::Result<()>, stop: anyhow::Result<String>) -> Self {
            let (state_tx, _) = tokio::sync::watch::channel(assistd_voice::VoiceCaptureState::Idle);
            Self {
                start_result: std::sync::Mutex::new(Some(start)),
                stop_result: std::sync::Mutex::new(Some(stop)),
                state_tx,
            }
        }
    }

    #[async_trait::async_trait]
    impl assistd_voice::VoiceInput for MockVoice {
        async fn start_recording(&self) -> anyhow::Result<()> {
            self.start_result
                .lock()
                .unwrap()
                .take()
                .unwrap_or_else(|| Ok(()))
        }
        async fn stop_and_transcribe(&self) -> anyhow::Result<String> {
            self.stop_result
                .lock()
                .unwrap()
                .take()
                .unwrap_or_else(|| Ok(String::new()))
        }
        fn state(&self) -> assistd_voice::VoiceCaptureState {
            *self.state_tx.borrow()
        }
        fn subscribe(&self) -> tokio::sync::watch::Receiver<assistd_voice::VoiceCaptureState> {
            self.state_tx.subscribe()
        }
    }

    #[tokio::test]
    async fn dispatch_ptt_start_emits_recording_then_done() {
        let voice = Arc::new(MockVoice::new(Ok(()), Ok(String::new())));
        let state = state_with_voice(Arc::new(EchoBackend::new()), voice);
        let (tx, rx) = mpsc::channel::<Event>(8);

        state
            .dispatch(Request::PttStart { id: "p1".into() }, tx)
            .await
            .unwrap();
        let events = collect_events(rx).await;

        assert_eq!(events.len(), 2, "expected VoiceState+Done, got {events:?}");
        assert!(matches!(
            &events[0],
            Event::VoiceState { id, state: VoiceCaptureState::Recording } if id == "p1"
        ));
        assert!(matches!(&events[1], Event::Done { id } if id == "p1"));
    }

    #[tokio::test]
    async fn dispatch_ptt_start_error_emits_error_event() {
        let voice = Arc::new(MockVoice::new(
            Err(anyhow::anyhow!("no mic")),
            Ok(String::new()),
        ));
        let state = state_with_voice(Arc::new(EchoBackend::new()), voice);
        let (tx, rx) = mpsc::channel::<Event>(8);

        let err = state
            .dispatch(Request::PttStart { id: "p2".into() }, tx)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("no mic"));

        let events = collect_events(rx).await;
        assert_eq!(events.len(), 1);
        match &events[0] {
            Event::Error { id, message } => {
                assert_eq!(id, "p2");
                assert!(message.contains("no mic"));
            }
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn dispatch_ptt_stop_with_text_runs_query() {
        // Non-empty transcription should: emit Transcribing → Idle →
        // Transcription(text), then dispatch as a Query (EchoBackend
        // echoes the text), then terminal Done.
        let voice = Arc::new(MockVoice::new(Ok(()), Ok("hello world".into())));
        let state = state_with_voice(Arc::new(EchoBackend::new()), voice);
        let (tx, rx) = mpsc::channel::<Event>(16);

        state
            .dispatch(Request::PttStop { id: "p3".into() }, tx)
            .await
            .unwrap();
        let events = collect_events(rx).await;

        // Order: Transcribing, Idle, Transcription, Delta(echo), Done.
        assert!(matches!(
            &events[0],
            Event::VoiceState {
                state: VoiceCaptureState::Transcribing,
                ..
            }
        ));
        assert!(matches!(
            &events[1],
            Event::VoiceState {
                state: VoiceCaptureState::Idle,
                ..
            }
        ));
        assert!(matches!(
            &events[2],
            Event::Transcription { text, .. } if text == "hello world"
        ));
        assert!(matches!(
            &events[3],
            Event::Delta { text, .. } if text == "hello world"
        ));
        assert!(matches!(events.last(), Some(Event::Done { .. })));
    }

    #[tokio::test]
    async fn dispatch_ptt_stop_empty_transcription_skips_query() {
        // Empty (VAD trimmed) transcription should NOT dispatch a Query.
        let voice = Arc::new(MockVoice::new(Ok(()), Ok(String::new())));
        let state = state_with_voice(Arc::new(EchoBackend::new()), voice);
        let (tx, rx) = mpsc::channel::<Event>(8);

        state
            .dispatch(Request::PttStop { id: "p4".into() }, tx)
            .await
            .unwrap();
        let events = collect_events(rx).await;

        // No Delta event should appear — the query is skipped.
        assert!(
            !events.iter().any(|e| matches!(e, Event::Delta { .. })),
            "expected no Delta on empty transcription: {events:?}"
        );
        assert!(matches!(
            events.iter().find(|e| matches!(e, Event::Transcription { .. })),
            Some(Event::Transcription { text, .. }) if text.is_empty()
        ));
        assert!(matches!(events.last(), Some(Event::Done { .. })));
    }

    /// Scripted `ContinuousListener` used by the listen-handler tests.
    /// Tracks start/stop call counts and exposes a toggleable "should
    /// fail" mode so we can exercise the error paths.
    struct MockListener {
        active: std::sync::atomic::AtomicBool,
        start_fails: std::sync::atomic::AtomicBool,
        state_tx: tokio::sync::watch::Sender<bool>,
        utterances: tokio::sync::broadcast::Sender<String>,
    }

    impl MockListener {
        fn new() -> Self {
            let (state_tx, _) = tokio::sync::watch::channel(false);
            let (utterances, _) = tokio::sync::broadcast::channel(4);
            Self {
                active: std::sync::atomic::AtomicBool::new(false),
                start_fails: std::sync::atomic::AtomicBool::new(false),
                state_tx,
                utterances,
            }
        }
        fn with_start_fails(self) -> Self {
            self.start_fails
                .store(true, std::sync::atomic::Ordering::SeqCst);
            self
        }
    }

    #[async_trait::async_trait]
    impl assistd_voice::ContinuousListener for MockListener {
        async fn start(&self) -> anyhow::Result<()> {
            if self.start_fails.load(std::sync::atomic::Ordering::SeqCst) {
                anyhow::bail!("mock listener start failed");
            }
            self.active.store(true, std::sync::atomic::Ordering::SeqCst);
            let _ = self.state_tx.send(true);
            Ok(())
        }
        async fn stop(&self) -> anyhow::Result<()> {
            self.active
                .store(false, std::sync::atomic::Ordering::SeqCst);
            let _ = self.state_tx.send(false);
            Ok(())
        }
        fn is_active(&self) -> bool {
            self.active.load(std::sync::atomic::Ordering::SeqCst)
        }
        fn subscribe_utterances(&self) -> tokio::sync::broadcast::Receiver<String> {
            self.utterances.subscribe()
        }
        fn subscribe_state(&self) -> tokio::sync::watch::Receiver<bool> {
            self.state_tx.subscribe()
        }
    }

    #[tokio::test]
    async fn dispatch_listen_start_emits_listen_state_true() {
        let listener = Arc::new(MockListener::new());
        let state = state_with_listener(Arc::new(EchoBackend::new()), listener.clone());
        let (tx, rx) = mpsc::channel::<Event>(4);
        state
            .dispatch(Request::ListenStart { id: "l1".into() }, tx)
            .await
            .unwrap();
        let events = collect_events(rx).await;
        assert!(matches!(
            &events[0],
            Event::ListenState { id, active: true } if id == "l1"
        ));
        assert!(matches!(events.last(), Some(Event::Done { .. })));
        assert!(listener.is_active());
    }

    #[tokio::test]
    async fn dispatch_listen_stop_emits_listen_state_false() {
        let listener = Arc::new(MockListener::new());
        listener
            .active
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let state = state_with_listener(Arc::new(EchoBackend::new()), listener.clone());
        let (tx, rx) = mpsc::channel::<Event>(4);
        state
            .dispatch(Request::ListenStop { id: "l2".into() }, tx)
            .await
            .unwrap();
        let events = collect_events(rx).await;
        assert!(matches!(
            &events[0],
            Event::ListenState { id, active: false } if id == "l2"
        ));
        assert!(matches!(events.last(), Some(Event::Done { .. })));
        assert!(!listener.is_active());
    }

    #[tokio::test]
    async fn dispatch_listen_toggle_flips_state() {
        let listener = Arc::new(MockListener::new());
        let state = state_with_listener(Arc::new(EchoBackend::new()), listener.clone());
        // Off → on.
        let (tx, rx) = mpsc::channel::<Event>(4);
        state
            .clone()
            .dispatch(Request::ListenToggle { id: "t1".into() }, tx)
            .await
            .unwrap();
        let events = collect_events(rx).await;
        assert!(matches!(
            &events[0],
            Event::ListenState { active: true, .. }
        ));
        // On → off.
        let (tx, rx) = mpsc::channel::<Event>(4);
        state
            .clone()
            .dispatch(Request::ListenToggle { id: "t2".into() }, tx)
            .await
            .unwrap();
        let events = collect_events(rx).await;
        assert!(matches!(
            &events[0],
            Event::ListenState { active: false, .. }
        ));
    }

    #[tokio::test]
    async fn dispatch_get_listen_state_returns_current_value() {
        let listener = Arc::new(MockListener::new());
        listener
            .active
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let state = state_with_listener(Arc::new(EchoBackend::new()), listener);
        let (tx, rx) = mpsc::channel::<Event>(4);
        state
            .dispatch(Request::GetListenState { id: "g1".into() }, tx)
            .await
            .unwrap();
        let events = collect_events(rx).await;
        assert!(matches!(
            &events[0],
            Event::ListenState { id, active: true } if id == "g1"
        ));
        assert!(matches!(events.last(), Some(Event::Done { .. })));
    }

    #[tokio::test]
    async fn dispatch_listen_start_error_propagates() {
        let listener = Arc::new(MockListener::new().with_start_fails());
        let state = state_with_listener(Arc::new(EchoBackend::new()), listener);
        let (tx, rx) = mpsc::channel::<Event>(4);
        state
            .dispatch(Request::ListenStart { id: "l3".into() }, tx)
            .await
            .unwrap_err();
        let events = collect_events(rx).await;
        assert!(
            matches!(events.last(), Some(Event::Error { message, .. }) if message.contains("mock listener start failed"))
        );
    }

    #[tokio::test]
    async fn ptt_start_rejected_while_listening_active() {
        let listener = Arc::new(MockListener::new());
        listener
            .active
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let state = state_with_listener(Arc::new(EchoBackend::new()), listener);
        let (tx, rx) = mpsc::channel::<Event>(4);
        state
            .dispatch(Request::PttStart { id: "p-ex".into() }, tx)
            .await
            .unwrap();
        let events = collect_events(rx).await;
        assert!(
            matches!(&events[0], Event::Error { message, .. } if message.contains("continuous listening"))
        );
    }

    #[tokio::test]
    async fn dispatch_ptt_stop_error_emits_error_event() {
        let voice = Arc::new(MockVoice::new(
            Ok(()),
            Err(anyhow::anyhow!("device disappeared")),
        ));
        let state = state_with_voice(Arc::new(EchoBackend::new()), voice);
        let (tx, rx) = mpsc::channel::<Event>(8);

        let err = state
            .dispatch(Request::PttStop { id: "p5".into() }, tx)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("device disappeared"));

        let events = collect_events(rx).await;
        assert!(events.iter().any(|e| matches!(
            e,
            Event::VoiceState {
                state: VoiceCaptureState::Transcribing,
                ..
            }
        )));
        assert!(events.iter().any(|e| matches!(
            e,
            Event::VoiceState {
                state: VoiceCaptureState::Idle,
                ..
            }
        )));
        match events.last() {
            Some(Event::Error { id, message }) => {
                assert_eq!(id, "p5");
                assert!(message.contains("device disappeared"));
            }
            other => panic!("expected terminal Error, got {other:?}"),
        }
    }

    // ---- TTS streaming tests ----

    /// Records every speak() in arrival order. Tests assert order and
    /// content. wait_idle() is counted so cleanup behavior is testable.
    struct MockSpeechRecorder {
        calls: StdMutex<Vec<String>>,
        wait_idle_calls: AtomicUsize,
    }

    impl MockSpeechRecorder {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                calls: StdMutex::new(Vec::new()),
                wait_idle_calls: AtomicUsize::new(0),
            })
        }

        fn calls(&self) -> Vec<String> {
            self.calls.lock().unwrap().clone()
        }

        fn wait_idle_count(&self) -> usize {
            self.wait_idle_calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait::async_trait]
    impl assistd_voice::VoiceOutput for MockSpeechRecorder {
        async fn speak(&self, text: String) -> anyhow::Result<()> {
            self.calls.lock().unwrap().push(text);
            Ok(())
        }
        async fn wait_idle(&self) -> anyhow::Result<()> {
            self.wait_idle_calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    fn state_with_speech_recorder(
        backend: Arc<dyn LlmBackend>,
        recorder: Arc<MockSpeechRecorder>,
        config: Config,
    ) -> Arc<AppState> {
        state_with_speech_recorder_and_enabled(backend, recorder, config, true)
    }

    fn state_with_speech_recorder_and_enabled(
        backend: Arc<dyn LlmBackend>,
        recorder: Arc<MockSpeechRecorder>,
        config: Config,
        initially_enabled: bool,
    ) -> Arc<AppState> {
        let ctrl = VoiceOutputController::new(recorder, initially_enabled);
        Arc::new(AppState::new(
            config,
            backend,
            PresenceManager::stub(PresenceState::Active),
            Arc::new(ToolRegistry::default()),
            Arc::new(assistd_voice::NoVoiceInput::new()),
            Arc::new(assistd_voice::NoContinuousListener::new()),
            ctrl,
        ))
    }

    #[tokio::test]
    async fn dispatch_query_speaks_sentences_in_order() {
        // EchoBackend emits the input as a single Delta. SentenceBuffer
        // pushes that into 4 sentences (3 from push, 1 from finish).
        // The new per-query speech worker MUST consume them in arrival
        // order — the previous fire-and-forget tokio::spawn pattern
        // could scramble them based on synthesis time.
        let recorder = MockSpeechRecorder::new();
        let state = state_with_speech_recorder(
            Arc::new(EchoBackend::new()),
            recorder.clone(),
            Config::default(),
        );
        let (tx, rx) = mpsc::channel::<Event>(16);
        state
            .dispatch(
                Request::Query {
                    id: "ord".into(),
                    text: "First. Second. Third. End.".into(),
                    attachments: Vec::new(),
                    version: None,
                },
                tx,
            )
            .await
            .unwrap();
        let _ = collect_events(rx).await;

        let calls = recorder.calls();
        assert_eq!(
            calls,
            vec![
                "First.".to_string(),
                "Second.".to_string(),
                "Third.".to_string(),
                "End.".to_string(),
            ],
            "sentences must be spoken in arrival order"
        );
        // Worker drained before handle_query returned.
        assert_eq!(recorder.wait_idle_count(), 1);
    }

    /// Backend whose `step` emits a scripted sequence of deltas (with
    /// optional sleeps between them) on the first call, then `Final`
    /// on subsequent calls.
    struct StreamingDeltaBackend {
        script: StdMutex<Option<Vec<DeltaScript>>>,
    }

    enum DeltaScript {
        Text(&'static str),
        Sleep(Duration),
    }

    impl StreamingDeltaBackend {
        fn new(script: Vec<DeltaScript>) -> Arc<Self> {
            Arc::new(Self {
                script: StdMutex::new(Some(script)),
            })
        }
    }

    #[async_trait::async_trait]
    impl LlmBackend for StreamingDeltaBackend {
        async fn generate(
            &self,
            _prompt: String,
            _tx: mpsc::Sender<LlmEvent>,
        ) -> anyhow::Result<()> {
            unimplemented!("uses step path")
        }
        async fn push_user(
            &self,
            _text: String,
            _attachments: Vec<assistd_tools::Attachment>,
        ) -> anyhow::Result<()> {
            Ok(())
        }
        async fn push_tool_results(&self, _results: Vec<ToolResultPayload>) -> anyhow::Result<()> {
            Ok(())
        }
        async fn step(
            &self,
            _tools: Vec<serde_json::Value>,
            tx: mpsc::Sender<LlmEvent>,
        ) -> anyhow::Result<StepOutcome> {
            let script = self.script.lock().unwrap().take();
            if let Some(actions) = script {
                for action in actions {
                    match action {
                        DeltaScript::Text(s) => {
                            tx.send(LlmEvent::Delta { text: s.into() }).await.ok();
                        }
                        DeltaScript::Sleep(d) => tokio::time::sleep(d).await,
                    }
                }
            }
            Ok(StepOutcome::Final)
        }
    }

    fn config_with_partial_flush(ms: u32) -> Config {
        let mut cfg = Config::default();
        cfg.voice.synthesis.partial_flush_ms = ms;
        cfg
    }

    #[tokio::test]
    async fn dispatch_query_partial_flush_after_idle() {
        // Backend emits a partial sentence ("Half a sente"), then
        // *stalls* for >partial_flush_ms, then completes ("nce. End.").
        // With the idle flush enabled the partial cuts at the last
        // whitespace ("Half a") and is spoken during the stall;
        // "sentence." then completes naturally on resume; "End." on
        // finish.
        let recorder = MockSpeechRecorder::new();
        let backend = StreamingDeltaBackend::new(vec![
            DeltaScript::Text("Half a sente"),
            DeltaScript::Sleep(Duration::from_millis(150)),
            DeltaScript::Text("nce. End."),
        ]);
        let cfg = config_with_partial_flush(50);
        let state = state_with_speech_recorder(backend, recorder.clone(), cfg);
        let (tx, rx) = mpsc::channel::<Event>(16);
        state
            .dispatch(
                Request::Query {
                    id: "pf".into(),
                    text: "go".into(),
                    attachments: Vec::new(),
                    version: None,
                },
                tx,
            )
            .await
            .unwrap();
        let _ = collect_events(rx).await;

        let calls = recorder.calls();
        assert_eq!(
            calls,
            vec![
                "Half a".to_string(),
                "sentence.".to_string(),
                "End.".to_string(),
            ],
            "expected idle-flush followed by completed sentence"
        );
    }

    #[tokio::test]
    async fn dispatch_query_partial_flush_zero_disables() {
        // Same backend, same stall — but partial_flush_ms = 0 means no
        // idle flush. Only the completed sentence and the final tail
        // are spoken.
        let recorder = MockSpeechRecorder::new();
        let backend = StreamingDeltaBackend::new(vec![
            DeltaScript::Text("Half a sente"),
            DeltaScript::Sleep(Duration::from_millis(150)),
            DeltaScript::Text("nce. End."),
        ]);
        let cfg = config_with_partial_flush(0);
        let state = state_with_speech_recorder(backend, recorder.clone(), cfg);
        let (tx, rx) = mpsc::channel::<Event>(16);
        state
            .dispatch(
                Request::Query {
                    id: "pf0".into(),
                    text: "go".into(),
                    attachments: Vec::new(),
                    version: None,
                },
                tx,
            )
            .await
            .unwrap();
        let _ = collect_events(rx).await;

        let calls = recorder.calls();
        assert_eq!(
            calls,
            vec!["Half a sentence.".to_string(), "End.".to_string()],
            "partial_flush_ms=0 should hold the partial in-buffer until \
             the LLM resumes, completing the sentence whole; got {calls:?}"
        );
    }

    /// Tool-emitting scripted backend: step #1 emits one Delta then
    /// returns ToolCalls; step #N+ emits the queued post-tool deltas
    /// then returns Final.
    struct ToolCallBackend {
        pre_delta: &'static str,
        post_delta: &'static str,
        outcomes: StdMutex<Vec<StepOutcome>>,
    }

    impl ToolCallBackend {
        fn new(pre: &'static str, post: &'static str, outcomes: Vec<StepOutcome>) -> Arc<Self> {
            Arc::new(Self {
                pre_delta: pre,
                post_delta: post,
                outcomes: StdMutex::new(outcomes),
            })
        }
    }

    #[async_trait::async_trait]
    impl LlmBackend for ToolCallBackend {
        async fn generate(
            &self,
            _prompt: String,
            _tx: mpsc::Sender<LlmEvent>,
        ) -> anyhow::Result<()> {
            unimplemented!("uses step path")
        }
        async fn push_user(
            &self,
            _text: String,
            _attachments: Vec<assistd_tools::Attachment>,
        ) -> anyhow::Result<()> {
            Ok(())
        }
        async fn push_tool_results(&self, _results: Vec<ToolResultPayload>) -> anyhow::Result<()> {
            Ok(())
        }
        async fn step(
            &self,
            _tools: Vec<serde_json::Value>,
            tx: mpsc::Sender<LlmEvent>,
        ) -> anyhow::Result<StepOutcome> {
            let outcome = {
                let mut q = self.outcomes.lock().unwrap();
                if q.is_empty() {
                    StepOutcome::Final
                } else {
                    q.remove(0)
                }
            };
            // Emit pre-delta on the FIRST step (when ToolCalls is queued)
            // and post-delta on Final.
            match &outcome {
                StepOutcome::ToolCalls(_) => {
                    tx.send(LlmEvent::Delta {
                        text: self.pre_delta.into(),
                    })
                    .await
                    .ok();
                }
                StepOutcome::Final => {
                    tx.send(LlmEvent::Delta {
                        text: self.post_delta.into(),
                    })
                    .await
                    .ok();
                }
            }
            Ok(outcome)
        }
    }

    /// A tool that sleeps before returning. Used to span the
    /// partial_flush_ms window so we can verify the flush is
    /// inhibited while a tool call is in flight.
    struct SleepTool {
        ms: u64,
    }

    #[async_trait::async_trait]
    impl assistd_tools::Tool for SleepTool {
        fn name(&self) -> &str {
            "sleep"
        }
        fn description(&self) -> &str {
            "sleep for testing"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
        async fn invoke(&self, _args: serde_json::Value) -> anyhow::Result<serde_json::Value> {
            tokio::time::sleep(Duration::from_millis(self.ms)).await;
            Ok(serde_json::json!({
                "output": "slept",
                "exit_code": 0,
                "duration_ms": self.ms,
                "truncated": false,
            }))
        }
    }

    #[tokio::test]
    async fn dispatch_query_tool_call_inhibits_idle_flush() {
        // Stream: Delta("Half a ") → ToolCall(sleep 300ms) →
        //          ToolResult → Delta("done.") → Done.
        // partial_flush_ms = 50 — would fire 6+ times during the
        // 300ms tool dispatch if not inhibited. With proper
        // `awaiting_tool_result` tracking, only the final tail is
        // spoken: "Half a done."
        let recorder = MockSpeechRecorder::new();
        let backend = ToolCallBackend::new(
            "Half a ",
            "done.",
            vec![
                StepOutcome::ToolCalls(vec![ToolCall {
                    id: "c1".into(),
                    name: "sleep".into(),
                    arguments: serde_json::json!({}),
                }]),
                StepOutcome::Final,
            ],
        );
        let mut tools = ToolRegistry::new();
        tools.register(SleepTool { ms: 300 });
        let mut cfg = config_with_partial_flush(50);
        cfg.voice.synthesis.max_sentence_chars = 400;
        let state = Arc::new(AppState::new(
            cfg,
            backend,
            PresenceManager::stub(PresenceState::Active),
            Arc::new(tools),
            Arc::new(assistd_voice::NoVoiceInput::new()),
            Arc::new(assistd_voice::NoContinuousListener::new()),
            VoiceOutputController::new(recorder.clone(), true),
        ));
        let (tx, rx) = mpsc::channel::<Event>(16);
        state
            .dispatch(
                Request::Query {
                    id: "tc".into(),
                    text: "go".into(),
                    attachments: Vec::new(),
                    version: None,
                },
                tx,
            )
            .await
            .unwrap();
        let _ = collect_events(rx).await;

        let calls = recorder.calls();
        assert!(
            !calls.iter().any(|c| c == "Half a"),
            "idle flush fired during tool dispatch — inhibition broken: {calls:?}"
        );
        assert_eq!(
            calls,
            vec!["Half a done.".to_string()],
            "expected single combined utterance after tool resolves; got {calls:?}"
        );
    }

    #[tokio::test]
    async fn dispatch_query_speech_worker_drains_before_return() {
        // The worker's wait_idle() must be awaited before handle_query
        // returns — otherwise daemon shutdown could cut off audio.
        let recorder = MockSpeechRecorder::new();
        let state = state_with_speech_recorder(
            Arc::new(EchoBackend::new()),
            recorder.clone(),
            Config::default(),
        );
        let (tx, rx) = mpsc::channel::<Event>(8);
        state
            .dispatch(
                Request::Query {
                    id: "drain".into(),
                    text: "Hello world.".into(),
                    attachments: Vec::new(),
                    version: None,
                },
                tx,
            )
            .await
            .unwrap();
        let _ = collect_events(rx).await;
        // wait_idle invoked exactly once after the channel closes.
        assert_eq!(recorder.wait_idle_count(), 1);
    }

    fn ctx(
        class: Option<&str>,
        title: Option<&str>,
        ws: Option<&str>,
    ) -> assistd_wm::FocusedWindowContext {
        // PR 3b adds an `id: Option<WindowId>` field; the prompt
        // formatter doesn't render it, so the helper leaves it None.
        assistd_wm::FocusedWindowContext {
            id: None,
            class: class.map(str::to_string),
            title: title.map(str::to_string),
            workspace: ws.map(str::to_string),
        }
    }

    #[test]
    fn format_window_context_block_full_terminal_includes_hint() {
        let block = format_window_context_block(&ctx(
            Some("Alacritty"),
            Some("nvim ~ src/main.rs"),
            Some("2"),
        ))
        .expect("Some block expected for non-empty ctx");
        assert!(block.starts_with("Current desktop context:\n"));
        assert!(block.contains("- Focused window: Alacritty — nvim ~ src/main.rs\n"));
        assert!(block.contains("- Workspace: 2\n"));
        assert!(block.contains("interacting with a terminal window."));
        // AC#3: hint references the actual `run` sub-command names.
        assert!(block.contains("`command: \"bash\"`"));
        assert!(block.contains("`command: \"wm\"`"));
    }

    #[test]
    fn format_window_context_block_non_terminal_omits_hint() {
        let block = format_window_context_block(&ctx(
            Some("firefox"),
            Some("Anthropic - claude.ai"),
            Some("3"),
        ))
        .expect("Some block expected for non-empty ctx");
        assert!(block.contains("interacting with a non-terminal window."));
        assert!(!block.contains("command: \"bash\""));
        assert!(!block.contains("command: \"wm\""));
    }

    #[test]
    fn format_window_context_block_omits_missing_fields() {
        // Class only — no title bar, no workspace line.
        let block = format_window_context_block(&ctx(Some("Alacritty"), None, None)).expect("Some");
        assert!(block.contains("- Focused window: Alacritty\n"));
        assert!(!block.contains(" — "));
        assert!(!block.contains("- Workspace:"));
    }

    #[test]
    fn format_window_context_block_returns_none_for_empty_ctx() {
        assert!(format_window_context_block(&ctx(None, None, None)).is_none());
    }

    #[test]
    fn format_window_context_block_workspace_only() {
        // Edge case: focus event was cleared by a Close but the
        // active workspace is still tracked. Should still produce a
        // block (just the workspace line + non-terminal kind).
        let block = format_window_context_block(&ctx(None, None, Some("scratch"))).expect("Some");
        assert!(!block.contains("- Focused window:"));
        assert!(block.contains("- Workspace: scratch\n"));
        assert!(block.contains("non-terminal window."));
    }

    #[test]
    fn combine_context_blocks_merges_with_blank_line() {
        let merged = combine_context_blocks(
            Some("Relevant past context:\n- foo\n".into()),
            Some("Current desktop context:\n- bar".into()),
        )
        .expect("Some");
        // Trailing newline of semantic block trimmed; blank line inserted
        // between the two blocks so the LLM sees clean separation.
        assert_eq!(
            merged,
            "Relevant past context:\n- foo\nCurrent desktop context:\n- bar"
        );
    }

    #[test]
    fn combine_context_blocks_passes_through_singletons() {
        assert_eq!(
            combine_context_blocks(Some("a".into()), None),
            Some("a".into())
        );
        assert_eq!(
            combine_context_blocks(None, Some("b".into())),
            Some("b".into())
        );
    }

    #[test]
    fn combine_context_blocks_returns_none_for_both_none() {
        assert!(combine_context_blocks(None, None).is_none());
    }
}
