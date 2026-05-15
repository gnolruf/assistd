//! `handle_query`: the main per-turn driver. Re-probes vision, decodes
//! attachments, wakes presence, runs the agent loop, streams events
//! back through the IPC channel, and finalizes persistence.
//!
//! The work is split into eight named phases below so the orchestrator
//! reads as a sequence of preconditions, side-effects, and cleanup
//! rather than one 500-line block. Helpers stay on `impl AppState` so
//! the per-substruct field paths remain at one indentation level.

use super::AppState;
use super::context::combine_context_blocks;
use super::wire::decode_wire_attachments;
use crate::Agent;
use crate::presence::{LlmStreamGuard, RequestGuard};
use anyhow::Result;
use assistd_ipc::Event;
use assistd_llm::LlmEvent;
use assistd_memory::{PersistedMessage, SessionId, TurnId};
use assistd_tools::Attachment;
use assistd_voice::{SentenceBuffer, SpeakDecision};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::Instrument;

/// The two presence-side guards that must outlive the entire turn
/// (including post-Done audio playback). The agent-turn lock is held
/// separately because it's released as soon as the LLM stream ends —
/// concurrent queries shouldn't wait for audio.
struct QueryGuards {
    _request: RequestGuard,
    _stream: LlmStreamGuard,
}

impl AppState {
    /// Handle a single user query: wake the daemon if needed, run the agent
    /// loop, stream events back through `tx`, and persist the turn.
    ///
    /// This is the primary entry point for all LLM-backed queries, including
    /// those dispatched internally by [`Self::handle_ptt_stop`].
    pub async fn handle_query(
        self: Arc<Self>,
        id: String,
        text: String,
        wire_attachments: Vec<assistd_ipc::ImageAttachment>,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
        let attachments = match self
            .prepare_attachments(&id, &wire_attachments, &tx)
            .await?
        {
            Some(a) => a,
            // prepare_attachments only returns Ok(None) for an
            // explicitly-handled "decline" path that isn't reachable
            // today; collapse to Ok so the dispatch loop sees a clean
            // terminal Done.
            None => return Ok(()),
        };
        let Some(_session_guards) = self.acquire_query_guards(&id, &tx).await? else {
            return Ok(());
        };
        // Serialize agent turns. Concurrent queries each wait here so
        // one turn's assistant/tool_calls/tool_result triplet lands in
        // conversation state atomically.
        let _agent_guard = self.runtime.agent_turn_lock.clone().lock_owned().await;

        let (current_session, turn_id) = self.open_persistence_turn(&text).await;
        self.assemble_transient_context(&text).await;

        // Cancellation token: wired to the agent loop so a slow LLM
        // step or tool dispatch can be preempted promptly. Drop guard
        // fires the token when this function returns for any reason.
        let cancel = tokio_util::sync::CancellationToken::new();
        let _cancel_on_return = cancel.clone().drop_guard();

        // Snapshot the user's prompt so the post-turn title-generation
        // hook (Done arm in `drive_event_loop`) can summarise without
        // racing `run_turn`, which moves `text`.
        let title_user_text = text.clone();
        let (llm_tx, llm_rx) = mpsc::channel::<LlmEvent>(32);
        let generator = self.spawn_agent_task(text, attachments, llm_tx, cancel.clone());

        // Sentence buffer + speech worker. Each completed sentence is
        // sent to a per-query worker task that calls voice_output.speak
        // serially. Buffer = 32 ≈ 2 minutes of queued speech.
        let synthesis = &self.config.voice.synthesis;
        let sentence_buf = SentenceBuffer::new_with_mode(
            synthesis.max_sentence_chars as usize,
            synthesis.code_block_mode,
        );
        let partial_flush = if synthesis.partial_flush_ms > 0 {
            Some(Duration::from_millis(synthesis.partial_flush_ms as u64))
        } else {
            None
        };
        let (speech_tx, speech_rx) = mpsc::channel::<String>(32);
        let start_epoch = self.subsystems.voice_output.current_epoch();
        let speech_handle = self.spawn_speech_worker(start_epoch, speech_rx);

        // Drive the LlmEvent → Event loop. This both forwards wire
        // events and feeds the speech worker; speech_tx is dropped
        // inside the helper at end of stream so the worker drains
        // remaining audio before exiting.
        self.clone()
            .drive_event_loop(
                id.clone(),
                llm_rx,
                &tx,
                speech_tx,
                sentence_buf,
                partial_flush,
                turn_id,
                title_user_text,
                current_session,
            )
            .await;

        // The agent_turn_lock can be released as soon as the turn ends;
        // concurrent queries shouldn't wait for audio playback to finish.
        let gen_result = generator.await;
        drop(_agent_guard);

        self.finalize_turn(turn_id, gen_result, speech_handle, &tx, id)
            .await
    }

    /// Phase 1: revalidate vision support and decode wire attachments.
    /// Always called on every turn (the probe is a single local HTTP
    /// `GET /props` with a 2s timeout, mutating the gate only on
    /// model swap). Returns `Err` on a bad attachment payload so the
    /// caller fails before the GPU round-trip.
    async fn prepare_attachments(
        &self,
        id: &str,
        wire: &[assistd_ipc::ImageAttachment],
        tx: &mpsc::Sender<Event>,
    ) -> Result<Option<Vec<Attachment>>> {
        if let Some(rev) = self.subsystems.vision_revalidator.as_ref() {
            rev.revalidate().await;
        }
        match decode_wire_attachments(wire) {
            Ok(v) => Ok(Some(v)),
            Err(e) => {
                let _ = tx
                    .send(Event::Error {
                        id: id.to_string(),
                        message: format!("invalid attachment: {e}"),
                    })
                    .await;
                Err(anyhow::anyhow!("invalid attachment: {e}"))
            }
        }
    }

    /// Phase 2: take the request + stream guards that keep presence
    /// `Active` for the lifetime of the turn (including post-Done
    /// audio). The agent-turn lock is acquired by the caller because
    /// its drop timing differs (released early, after the LLM stream
    /// ends). Returns `Ok(None)` only if the wake fails after an
    /// `Event::Error` has already been emitted.
    async fn acquire_query_guards(
        &self,
        id: &str,
        tx: &mpsc::Sender<Event>,
    ) -> Result<Option<QueryGuards>> {
        let request = match self.subsystems.presence.acquire_request_guard().await {
            Ok(g) => g,
            Err(e) => {
                let _ = tx
                    .send(Event::Error {
                        id: id.to_string(),
                        message: format!("wake failed: {e:#}"),
                    })
                    .await;
                return Err(e);
            }
        };
        let stream = self.subsystems.presence.acquire_stream_guard();
        Ok(Some(QueryGuards {
            _request: request,
            _stream: stream,
        }))
    }

    /// Phase 3: open a persistence turn and fire-and-forget the user
    /// message. Returns `(session, turn_id)`. `turn_id` is `None`
    /// when persistence is disabled (`NoConversationStore`).
    async fn open_persistence_turn(&self, text: &str) -> (Arc<SessionId>, Option<TurnId>) {
        let (current_session, _current_branch) = self.runtime.conversation_ctx.current().await;
        let turn_id: Option<TurnId> = match self
            .memory
            .conversations
            .begin_turn(&current_session, text)
            .await
        {
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
        // way to the LLM; fire-and-forget so the dispatch loop never
        // waits on disk.
        self.persist_message_fire_and_forget(turn_id, PersistedMessage::user(text.to_string()));
        (current_session, turn_id)
    }

    /// Phase 4: build the LLM's per-turn transient system message from
    /// semantic recall + focused-window context, then push it. Both
    /// sources are best-effort: any failure debug-logs and continues
    /// (so a flaky compositor or embedder never blocks a turn).
    async fn assemble_transient_context(&self, text: &str) {
        let semantic = if self.memory.embedding_cfg.enabled && self.memory.embedding_cfg.auto_inject
        {
            match self.build_semantic_context(text).await {
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
            && let Err(e) = self.subsystems.llm.set_transient_context(block).await
        {
            tracing::debug!(
                target: "assistd::context",
                error = %e,
                "set_transient_context failed; continuing without it",
            );
        }
    }

    /// Phase 5: spawn the agent task. Returns the join handle so the
    /// orchestrator can await completion after the event loop drains.
    /// The agent owns its own LLM and tool handles (cloned `Arc`s) so
    /// the spawn doesn't borrow from `self`.
    fn spawn_agent_task(
        &self,
        text: String,
        attachments: Vec<Attachment>,
        llm_tx: mpsc::Sender<LlmEvent>,
        cancel: tokio_util::sync::CancellationToken,
    ) -> JoinHandle<Result<()>> {
        let llm = self.subsystems.llm.clone();
        let tools = self.subsystems.tools.clone();
        let max_iterations = self.config.agent.max_iterations;
        let health: Option<Arc<dyn assistd_llm::LlmHealthProbe>> = Some(Arc::new(
            crate::presence::PresenceLlmHealthProbe::new(self.subsystems.presence.clone()),
        ));
        let agent = Agent::new(llm, tools, max_iterations, health);
        tokio::spawn(
            async move { agent.run_turn(text, attachments, llm_tx, cancel).await }
                .in_current_span(),
        )
    }

    /// Phase 6: spawn the per-query speech worker. The worker drains
    /// the bounded `speech_rx` channel, honoring `should_speak` so a
    /// later `skip()` / `interrupt()` drops every remaining sentence
    /// for THIS query without affecting future ones. On channel close
    /// it calls `wait_idle()` so the daemon doesn't tear down audio
    /// mid-playback.
    fn spawn_speech_worker(
        &self,
        start_epoch: u64,
        mut speech_rx: mpsc::Receiver<String>,
    ) -> JoinHandle<()> {
        let ctrl = self.subsystems.voice_output.clone();
        tokio::spawn(
            async move {
                while let Some(sentence) = speech_rx.recv().await {
                    match ctrl.should_speak(start_epoch) {
                        SpeakDecision::Speak => {
                            // Truncate the snippet for logging; a long
                            // sentence in the log line clutters the journal.
                            let preview: String = sentence.chars().take(60).collect();
                            if let Err(e) = ctrl.inner().speak(sentence).await {
                                tracing::warn!(
                                    target: "assistd::voice",
                                    error = %e,
                                    sentence_preview = %preview,
                                    "voice_output.speak failed; sentence dropped"
                                );
                            }
                        }
                        SpeakDecision::DropSilent | SpeakDecision::DropForSkip => {
                            // TTS toggled off, or skipped: drain the channel
                            // without speaking.
                        }
                    }
                }
                if let Err(e) = ctrl.inner().wait_idle().await {
                    tracing::debug!(
                        target: "assistd::voice",
                        error = %e,
                        "voice_output.wait_idle failed (non-fatal)"
                    );
                }
            }
            .in_current_span(),
        )
    }

    /// Phase 7: the `LlmEvent` → `Event` translation loop. Consumes
    /// `llm_rx`, forwards wire events to the IPC `tx`, feeds completed
    /// sentences into `speech_tx`, accumulates assistant text for
    /// persistence boundaries, and inhibits the idle flush during a
    /// tool call. `speech_tx` is moved by value so it drops at end of
    /// this function — that signals the speech worker to wait_idle
    /// and exit.
    #[allow(clippy::too_many_arguments)]
    async fn drive_event_loop(
        self: Arc<Self>,
        id: String,
        mut llm_rx: mpsc::Receiver<LlmEvent>,
        tx: &mpsc::Sender<Event>,
        speech_tx: mpsc::Sender<String>,
        mut sentence_buf: SentenceBuffer,
        partial_flush: Option<Duration>,
        turn_id: Option<TurnId>,
        title_user_text: String,
        current_session: Arc<SessionId>,
    ) {
        // Tool calls inhibit the idle-flush: the LLM is waiting on a
        // tool, not stalled mid-prose. Without this, a long-running
        // bash command would trigger spurious mid-sentence flushes.
        let mut awaiting_tool_result = false;

        // Persistence accumulator: tee every `LlmEvent::Delta` into
        // this string so the assistant's full reply gets written as
        // one row at end-of-turn. Cleared on each Tool/Done flush.
        let mut assistant_accum = String::new();

        let mut client_alive = true;
        let mut first_sentence_emitted = false;

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

            // Wire events are forwarded before speech_tx.send so client
            // display latency stays independent of TTS backpressure.
            let (wire_event, sentences_to_speak): (Event, Vec<String>) = match llm_event {
                LlmEvent::Delta { text } => {
                    let sentences = sentence_buf.push(&text);
                    assistant_accum.push_str(&text);
                    (
                        Event::Delta {
                            id: id.clone(),
                            text,
                        },
                        sentences,
                    )
                }
                LlmEvent::ReasoningDelta { text } => {
                    // Reasoning is ephemeral display-only: never
                    // persisted to history, never fed to TTS.
                    (
                        Event::ReasoningDelta {
                            id: id.clone(),
                            text,
                        },
                        Vec::new(),
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
                    // onto the tool result downstream.
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
                LlmEvent::Status {
                    severity,
                    component,
                    event,
                    message,
                } => {
                    // On a recoverable restart, drop any partial
                    // accumulator state so a successful replay isn't
                    // persisted as `<partial><replay>` concatenated.
                    if event == "restarting" {
                        assistant_accum.clear();
                        let _ = sentence_buf.finish();
                    }
                    (
                        Event::Status {
                            id: id.clone(),
                            severity,
                            component,
                            event,
                            message,
                        },
                        Vec::new(),
                    )
                }
                LlmEvent::Done => {
                    let tail = sentence_buf.finish();
                    let sentences = tail.into_iter().collect();
                    if !assistant_accum.is_empty() {
                        let final_text = std::mem::take(&mut assistant_accum);
                        self.persist_message_fire_and_forget(
                            turn_id,
                            PersistedMessage::assistant_text(final_text),
                        );
                    }
                    // Kick off LLM title summarisation in the background.
                    // The helper short-circuits when a title is already
                    // set, so this is cheap on every-turn-but-the-first.
                    self.clone().spawn_session_title_generation(
                        current_session.clone(),
                        title_user_text.clone(),
                    );
                    (Event::Done { id: id.clone() }, sentences)
                }
            };

            if client_alive && tx.send(wire_event).await.is_err() {
                client_alive = false;
            }

            for s in sentences_to_speak {
                if !first_sentence_emitted {
                    tracing::debug!(
                        target: "assistd::voice::latency",
                        stage = "first_sentence_emitted",
                        "voice latency stage"
                    );
                    first_sentence_emitted = true;
                }
                if speech_tx.send(s).await.is_err() {
                    tracing::debug!(
                        target: "assistd::voice",
                        "speech worker channel closed; dropping sentence"
                    );
                    break;
                }
            }

            if !client_alive {
                // Stop pulling from llm_rx so the agent loop's
                // `tx.is_closed()` check fires at its next iteration
                // boundary. The speech worker keeps draining queued
                // audio (speech_tx still alive until function return).
                break;
            }
        }
        // speech_tx drops here, signalling the worker to wait_idle()
        // and exit; playback finishes naturally.
    }

    /// Phase 8: end the persisted turn (if any), await the speech
    /// worker so audio playback completes inside the still-active
    /// presence guards, and translate the agent's `JoinHandle` result
    /// into the dispatch loop's `Result`.
    async fn finalize_turn(
        &self,
        turn_id: Option<TurnId>,
        gen_result: std::result::Result<Result<()>, tokio::task::JoinError>,
        speech_handle: JoinHandle<()>,
        tx: &mpsc::Sender<Event>,
        id: String,
    ) -> Result<()> {
        // Every exit path lands an ended_at timestamp on the turn row.
        if let Some(t) = turn_id {
            let conv = self.memory.conversations.clone();
            self.runtime.persistence_tracker.spawn(async move {
                if let Err(e) = conv.end_turn(t).await {
                    tracing::warn!(
                        target: "assistd::memory",
                        error = %e,
                        "end_turn failed (continuing)"
                    );
                }
            });
        }

        // _session_guards (held by caller) are still alive so presence
        // stays Active through playback.
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
}
