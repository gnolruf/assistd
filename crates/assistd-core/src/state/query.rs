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
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::Instrument;

/// Caps the per-turn `Event::LastDelta` republish rate so a fast
/// token stream doesn't flood passive subscribers.
const LAST_DELTA_DEBOUNCE: Duration = Duration::from_millis(100);

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
            None => return Ok(()),
        };
        let Some(_session_guards) = self.acquire_query_guards(&id, &tx).await? else {
            return Ok(());
        };
        let _agent_guard = self.runtime.agent_turn_lock.clone().lock_owned().await;

        let (current_session, turn_id) = self.open_persistence_turn(&text).await;
        self.assemble_transient_context(&text).await;

        let cancel = tokio_util::sync::CancellationToken::new();
        let _cancel_on_return = cancel.clone().drop_guard();
        // Publish the token so an external `InterruptTurn` can reach it.
        // The slot is held only while this turn is in flight; we clear
        // it before releasing `agent_turn_lock` so the next queued turn
        // starts with an empty slot.
        *self.runtime.current_cancel.lock().await = Some(cancel.clone());

        let title_user_text = text.clone();
        let (llm_tx, llm_rx) = mpsc::channel::<LlmEvent>(32);
        let generator = self.spawn_agent_task(text, attachments, llm_tx, cancel.clone());

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
        let speech_handle = self.spawn_speech_worker(id.clone(), start_epoch, speech_rx);

        let done_emitted = self
            .clone()
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

        let gen_result = generator.await;
        *self.runtime.current_cancel.lock().await = None;
        drop(_agent_guard);

        self.finalize_turn(turn_id, gen_result, speech_handle, &tx, id, done_emitted)
            .await
    }

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

    async fn acquire_query_guards(
        &self,
        id: &str,
        tx: &mpsc::Sender<Event>,
    ) -> Result<Option<QueryGuards>> {
        let request = match self
            .subsystems
            .presence
            .acquire_request_guard_with_progress(id.to_string(), tx.clone())
            .await
        {
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

    async fn open_persistence_turn(&self, text: &str) -> (Arc<SessionId>, Option<TurnId>) {
        let (current_session, _current_branch) = self.runtime.conversation_ctx.current().await;
        let turn_id: Option<TurnId> = match self
            .memory
            .conversations
            .begin_turn(&current_session, text)
            .await
        {
            Ok(t) if t.0 != 0 => Some(t),
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
        self.persist_message_fire_and_forget(turn_id, PersistedMessage::user(text.to_string()));
        (current_session, turn_id)
    }

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

    fn spawn_speech_worker(
        &self,
        id: String,
        start_epoch: u64,
        mut speech_rx: mpsc::Receiver<String>,
    ) -> JoinHandle<()> {
        let ctrl = self.subsystems.voice_output.clone();
        let events_bus = self.runtime.events_bus().clone();
        tokio::spawn(
            async move {
                // Emit SpeakingState{true} on the first sentence we
                // actually play; flip back to false after wait_idle.
                // Turns that never play a sentence (TTS off, skipped,
                // empty reply) emit nothing — passive subscribers like
                // the tray popup only care about audible playback.
                let mut emitted_start = false;
                while let Some(sentence) = speech_rx.recv().await {
                    match ctrl.should_speak(start_epoch) {
                        SpeakDecision::Speak => {
                            let preview: String = sentence.chars().take(60).collect();
                            if !emitted_start {
                                let _ = events_bus.send(Event::SpeakingState {
                                    id: id.clone(),
                                    speaking: true,
                                });
                                emitted_start = true;
                            }
                            if let Err(e) = ctrl.inner().speak(sentence).await {
                                tracing::warn!(
                                    target: "assistd::voice",
                                    error = %e,
                                    sentence_preview = %preview,
                                    "voice_output.speak failed; sentence dropped"
                                );
                            }
                            // `speak()` synthesises then enqueues; if a
                            // skip fired mid-synthesis the controller's
                            // playback.clear() already ran, then this
                            // call appended the just-synthesised PCM
                            // *after* the clear. Re-check the epoch and
                            // cancel again so the late-arriving audio
                            // doesn't leak through.
                            if matches!(ctrl.should_speak(start_epoch), SpeakDecision::DropForSkip)
                            {
                                ctrl.inner().cancel().await;
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
                if emitted_start {
                    let _ = events_bus.send(Event::SpeakingState {
                        id,
                        speaking: false,
                    });
                }
            }
            .in_current_span(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    /// Returns `true` when the loop emitted a terminal `Event::Done`
    /// from the LLM stream. Returns `false` when the loop broke early
    /// (cancelled turn, client gone, llm_rx dropped before
    /// `LlmEvent::Done`). The caller uses this in
    /// [`Self::finalize_turn`] to decide whether to emit a synthetic
    /// `Done` so the IPC client never hangs.
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
    ) -> bool {
        let mut awaiting_tool_result = false;

        let mut assistant_accum = String::new();

        let mut client_alive = true;
        let mut first_sentence_emitted = false;
        let mut done_emitted = false;

        let events_bus = self.runtime.events_bus().clone();
        // Backdate so the first Delta emits without waiting a window.
        let mut last_emit_at = Instant::now()
            .checked_sub(LAST_DELTA_DEBOUNCE)
            .unwrap_or_else(Instant::now);

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

            let (wire_event, sentences_to_speak): (Event, Vec<String>) = match llm_event {
                LlmEvent::Delta { text } => {
                    let sentences = sentence_buf.push(&text);
                    assistant_accum.push_str(&text);
                    if last_emit_at.elapsed() >= LAST_DELTA_DEBOUNCE {
                        let _ = events_bus.send(Event::LastDelta {
                            id: id.clone(),
                            text: assistant_accum.clone(),
                        });
                        last_emit_at = Instant::now();
                    }
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
                    if !assistant_accum.is_empty() {
                        let pre_text = std::mem::take(&mut assistant_accum);
                        let _ = events_bus.send(Event::LastDelta {
                            id: id.clone(),
                            text: pre_text.clone(),
                        });
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
                        let _ = events_bus.send(Event::LastDelta {
                            id: id.clone(),
                            text: final_text.clone(),
                        });
                        self.persist_message_fire_and_forget(
                            turn_id,
                            PersistedMessage::assistant_text(final_text),
                        );
                    }
                    self.clone().spawn_session_title_generation(
                        current_session.clone(),
                        title_user_text.clone(),
                    );
                    done_emitted = true;
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
                break;
            }
        }
        done_emitted
    }

    async fn finalize_turn(
        &self,
        turn_id: Option<TurnId>,
        gen_result: std::result::Result<Result<()>, tokio::task::JoinError>,
        speech_handle: JoinHandle<()>,
        tx: &mpsc::Sender<Event>,
        id: String,
        done_emitted: bool,
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

        let _ = speech_handle.await;

        match gen_result {
            Ok(Ok(())) => {
                // Cancellation (e.g. `Request::InterruptTurn`) makes
                // `Agent::run_turn` return `Ok(())` mid-stream without
                // emitting a `LlmEvent::Done`. The IPC contract still
                // requires a terminal event per request, so synthesise
                // one here when the event loop never observed Done.
                if !done_emitted {
                    let _ = tx.send(Event::Done { id }).await;
                }
                Ok(())
            }
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
