use crate::{Config, PresenceManager, run_agent_turn};
use anyhow::Result;
use assistd_ipc::{Event, PresenceState, Request, VoiceCaptureState};
use assistd_llm::{LlmBackend, LlmEvent};
use assistd_tools::ToolRegistry;
use assistd_voice::{ContinuousListener, SentenceBuffer, VoiceInput, VoiceOutput};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, mpsc};

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
    /// Speaks LLM responses aloud, sentence by sentence. When TTS is
    /// disabled in config or the build, this is `NoVoiceOutput` which
    /// silently accepts every speak() call.
    pub voice_output: Arc<dyn VoiceOutput>,
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
        voice_output: Arc<dyn VoiceOutput>,
    ) -> Self {
        Self {
            config,
            llm,
            presence,
            tools,
            voice,
            listener,
            voice_output,
            agent_turn_lock: Arc::new(Mutex::new(())),
        }
    }

    /// Route a single incoming request to the appropriate subsystem.
    ///
    /// Events are streamed back through `tx`. When this function returns
    /// (either `Ok` or `Err`), no further events will be sent. The caller
    /// is responsible for surfacing `Err` to the client as an
    /// [`Event::Error`] if one hasn't been emitted already.
    pub async fn dispatch(self: Arc<Self>, req: Request, tx: mpsc::Sender<Event>) -> Result<()> {
        match req {
            Request::Query { id, text } => self.handle_query(id, text, tx).await,
            Request::SetPresence { id, target } => self.handle_set_presence(id, target, tx).await,
            Request::GetPresence { id } => self.handle_get_presence(id, tx).await,
            Request::Cycle { id } => self.handle_cycle(id, tx).await,
            Request::PttStart { id } => self.handle_ptt_start(id, tx).await,
            Request::PttStop { id } => self.handle_ptt_stop(id, tx).await,
            Request::ListenStart { id } => self.handle_listen_start(id, tx).await,
            Request::ListenStop { id } => self.handle_listen_stop(id, tx).await,
            Request::ListenToggle { id } => self.handle_listen_toggle(id, tx).await,
            Request::GetListenState { id } => self.handle_get_listen_state(id, tx).await,
        }
    }

    pub async fn handle_query(
        self: Arc<Self>,
        id: String,
        text: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<()> {
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

        let (llm_tx, mut llm_rx) = mpsc::channel::<LlmEvent>(32);
        let llm = self.llm.clone();
        let tools = self.tools.clone();
        let max_iterations = self.config.agent.max_iterations;
        let generator =
            tokio::spawn(
                async move { run_agent_turn(llm, tools, max_iterations, text, llm_tx).await },
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
        let voice = self.voice_output.clone();
        let speech_handle = tokio::spawn(async move {
            while let Some(sentence) = speech_rx.recv().await {
                if let Err(e) = voice.speak(sentence).await {
                    tracing::debug!(
                        target: "assistd::voice",
                        error = %e,
                        "voice_output.speak failed (non-fatal)"
                    );
                }
            }
            // Channel closed: drain anything still in the audio queue
            // before the worker returns. Any failure here is logged
            // inside wait_idle and downgraded to Ok.
            if let Err(e) = voice.wait_idle().await {
                tracing::debug!(
                    target: "assistd::voice",
                    error = %e,
                    "voice_output.wait_idle failed (non-fatal)"
                );
            }
        });

        // Tool calls inhibit the idle-flush — the LLM is *waiting on a
        // tool*, not stalled mid-prose. Without this, a long-running
        // bash command would trigger spurious mid-sentence flushes.
        let mut awaiting_tool_result = false;
        let partial_flush = if partial_flush_ms > 0 {
            Some(Duration::from_millis(partial_flush_ms as u64))
        } else {
            None
        };

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
                    (
                        Event::Delta {
                            id: id.clone(),
                            text,
                        },
                        sentences,
                    )
                }
                LlmEvent::ToolCall {
                    id: _call_id,
                    name,
                    arguments,
                } => {
                    awaiting_tool_result = true;
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
                    id: _call_id,
                    name,
                    result,
                } => {
                    awaiting_tool_result = false;
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
        self.handle_query(id, text, tx).await
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
            Arc::new(assistd_voice::NoVoiceOutput),
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
            Arc::new(assistd_voice::NoVoiceOutput),
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
            Arc::new(assistd_voice::NoVoiceOutput),
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
        async fn push_user(&self, _text: String) -> anyhow::Result<()> {
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
            Arc::new(assistd_voice::NoVoiceOutput),
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
    impl VoiceOutput for MockSpeechRecorder {
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
        Arc::new(AppState::new(
            config,
            backend,
            PresenceManager::stub(PresenceState::Active),
            Arc::new(ToolRegistry::default()),
            Arc::new(assistd_voice::NoVoiceInput::new()),
            Arc::new(assistd_voice::NoContinuousListener::new()),
            recorder,
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
        async fn push_user(&self, _text: String) -> anyhow::Result<()> {
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
        async fn push_user(&self, _text: String) -> anyhow::Result<()> {
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
            recorder.clone(),
        ));
        let (tx, rx) = mpsc::channel::<Event>(16);
        state
            .dispatch(
                Request::Query {
                    id: "tc".into(),
                    text: "go".into(),
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
                },
                tx,
            )
            .await
            .unwrap();
        let _ = collect_events(rx).await;
        // wait_idle invoked exactly once after the channel closes.
        assert_eq!(recorder.wait_idle_count(), 1);
    }
}
