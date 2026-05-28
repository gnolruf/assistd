use crate::{Config, PresenceManager};
use anyhow::Result;
use assistd_ipc::{Event, Request};
use assistd_llm::LlmBackend;
use assistd_tools::ToolRegistry;
use assistd_voice::{ContinuousListener, VoiceInput, VoiceOutputController};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;

pub(crate) mod branches;
pub(crate) mod capabilities;
pub(crate) mod context;
pub(crate) mod memory_handlers;
pub(crate) mod memory_stack;
pub(crate) mod persistence;
pub(crate) mod presence_handlers;
pub(crate) mod query;
pub(crate) mod runtime;
pub(crate) mod subscribe;
pub(crate) mod subsystems;
pub(crate) mod voice_handlers;
pub(crate) mod wire;

pub use self::memory_stack::MemoryStack;
pub use self::runtime::{ConversationContext, RuntimeState};
pub use self::subsystems::{McpStartupFailure, Subsystems};

/// Shared, long-lived daemon state handed to every request handler.
///
/// Decomposed into four orthogonal slots:
/// - [`Self::config`]: static daemon configuration.
/// - [`Self::subsystems`]: live backend handles (LLM, presence, voice,
///   tools, window manager, vision revalidator).
/// - [`Self::memory`]: persistence + embedding stack.
/// - [`Self::runtime`]: per-process bookkeeping (locks, trackers,
///   active conversation pointer).
///
/// All four substructs hold their internals as `Arc<...>` / cheap-clone
/// types so the parent `Arc<AppState>` provides cross-handler sharing
/// without extra indirection.
pub struct AppState {
    pub config: Config,
    pub subsystems: Subsystems,
    pub memory: MemoryStack,
    pub runtime: RuntimeState,
}

impl AppState {
    /// Construct a minimal `AppState` with stub memory and embedding backends.
    ///
    /// Subsystems are wired up from the supplied backends; memory + runtime
    /// inherit no-op placeholders. Production callers (`crates/assistd/src/daemon.rs`)
    /// assemble `AppState` from substruct builders directly (see [`Subsystems`],
    /// [`MemoryStack`], [`RuntimeState`]); this constructor is retained for
    /// tests that only need a thin shell.
    pub fn new(
        config: Config,
        llm: Arc<dyn LlmBackend>,
        presence: Arc<PresenceManager>,
        tools: Arc<ToolRegistry>,
        voice: Arc<dyn VoiceInput>,
        listener: Arc<dyn ContinuousListener>,
        voice_output: Arc<VoiceOutputController>,
    ) -> Self {
        let subsystems = Subsystems::new(llm, presence, tools, voice, listener, voice_output);
        let memory = MemoryStack::disabled(config.embedding.clone());
        let runtime = RuntimeState::new();
        Self {
            config,
            subsystems,
            memory,
            runtime,
        }
    }

    /// Route a single incoming request to the appropriate subsystem.
    ///
    /// Events are streamed back through `tx`. When this function returns
    /// (either `Ok` or `Err`), no further events will be sent. The caller
    /// is responsible for surfacing `Err` to the client as an
    /// [`Event::Error`] if one hasn't been emitted already.
    pub async fn dispatch(self: Arc<Self>, req: Request, tx: mpsc::Sender<Event>) -> Result<()> {
        // Subscribe lives for the client's lifetime; no envelope cap.
        if matches!(req, Request::Subscribe { .. }) {
            return self.dispatch_inner(req, tx).await;
        }
        let envelope = Duration::from_secs(self.config.timeouts.dispatch_envelope_secs);
        let req_id = req.id().to_string();
        let req_kind = req.kind();
        let tx_for_timeout = tx.clone();
        let inner = self.clone().dispatch_inner(req, tx);
        match tokio::time::timeout(envelope, inner).await {
            Ok(result) => result,
            Err(_) => {
                tracing::warn!(
                    target: "assistd::state",
                    id = %req_id,
                    kind = req_kind,
                    timeout_secs = self.config.timeouts.dispatch_envelope_secs,
                    "dispatch envelope timeout exceeded; aborting request"
                );
                let _ = tx_for_timeout
                    .send(Event::Error {
                        id: req_id,
                        message: format!(
                            "request exceeded {}s envelope timeout",
                            self.config.timeouts.dispatch_envelope_secs
                        ),
                    })
                    .await;
                Ok(())
            }
        }
    }

    async fn dispatch_inner(self: Arc<Self>, req: Request, tx: mpsc::Sender<Event>) -> Result<()> {
        match req {
            Request::Query {
                id,
                text,
                attachments,
            } => self.handle_query(id, text, attachments, tx).await,
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
            Request::InterruptTurn { id } => self.handle_interrupt_turn(id, tx).await,
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
            Request::Fork { id, name } => self.handle_fork(id, name, tx).await,
            Request::Branches { id } => self.handle_branches(id, tx).await,
            Request::Switch { id, target } => self.handle_switch(id, target, tx).await,
            Request::Undo { id } => self.handle_undo(id, tx).await,
            Request::ResumeOrNew { id, recency_secs } => {
                self.handle_resume_or_new(id, recency_secs, tx).await
            }
            Request::NewSession { id } => self.handle_new_session(id, tx).await,
            Request::Subscribe { id, filter } => self.handle_subscribe(id, filter, tx).await,
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Config;
    use crate::state::branches::clean_generated_title;
    use crate::state::context::{combine_context_blocks, format_window_context_block};
    use assistd_config::ToolsOutputConfig;
    use assistd_ipc::{PresenceState, VoiceCaptureState};
    use assistd_llm::{
        EchoBackend, FailedBackend, LlmEvent, StepOutcome, ToolCall, ToolResultPayload,
    };
    use assistd_memory::ConversationStore;
    use assistd_tools::{CommandRegistry, RunTool, commands::EchoCommand};
    use parking_lot::Mutex as StdMutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn clean_generated_title_strips_quotes_and_first_lines_only() {
        assert_eq!(clean_generated_title("\"Cats and dogs\""), "Cats and dogs");
        assert_eq!(
            clean_generated_title("Title: weather in Berlin\n(extra explanation)"),
            "Title: weather in Berlin"
        );
        assert_eq!(clean_generated_title("\n\n  hello world.  "), "hello world");
        assert_eq!(clean_generated_title("**bolded title**"), "bolded title");
        assert_eq!(clean_generated_title(""), "");
    }

    #[test]
    fn clean_generated_title_caps_length() {
        let raw = "x".repeat(200);
        let out = clean_generated_title(&raw);
        assert_eq!(out.chars().count(), 80);
    }

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
    async fn persistence_tracker_drains_in_flight_tasks() {
        let state = test_state(Arc::new(EchoBackend::new()), PresenceState::Active);
        let tracker = state.runtime.persistence_tracker_handle();
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        tracker.spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            c.fetch_add(1, Ordering::SeqCst);
        });
        tracker.close();
        tokio::time::timeout(Duration::from_secs(1), tracker.wait())
            .await
            .expect("tracker.wait must complete within the budget");
        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "tracker.wait returned before the spawned task incremented the counter"
        );
    }

    #[tokio::test]
    async fn dispatch_query_emits_delta_then_done() {
        let state = test_state(Arc::new(EchoBackend::new()), PresenceState::Active);
        let (tx, rx) = mpsc::channel::<Event>(8);
        let req = Request::Query {
            id: "q1".into(),
            text: "hello".into(),
            attachments: Vec::new(),
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
        assert_eq!(state.subsystems.presence.state(), PresenceState::Sleeping);
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
        // State must be unchanged: GetPresence is a read-only snapshot.
        assert_eq!(state.subsystems.presence.state(), PresenceState::Drowsy);
    }

    #[tokio::test]
    async fn dispatch_cycle_advances_to_next_state() {
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
        assert_eq!(state.subsystems.presence.state(), PresenceState::Sleeping);
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
        outcomes: parking_lot::Mutex<Vec<StepOutcome>>,
    }

    #[async_trait::async_trait]
    impl LlmBackend for ScriptedBackend {
        async fn generate(
            &self,
            _prompt: String,
            _tx: mpsc::Sender<LlmEvent>,
        ) -> assistd_llm::LlmResult<()> {
            unimplemented!("uses step path")
        }
        async fn push_user(
            &self,
            _text: String,
            _attachments: Vec<assistd_tools::Attachment>,
        ) -> assistd_llm::LlmResult<()> {
            Ok(())
        }
        async fn push_tool_results(
            &self,
            _results: Vec<ToolResultPayload>,
        ) -> assistd_llm::LlmResult<()> {
            Ok(())
        }
        async fn step(
            &self,
            _tools: Vec<serde_json::Value>,
            _tx: mpsc::Sender<LlmEvent>,
        ) -> assistd_llm::LlmResult<StepOutcome> {
            let outcome = {
                let mut q = self.outcomes.lock();
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
        let backend = Arc::new(ScriptedBackend {
            outcomes: parking_lot::Mutex::new(vec![
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
        start_result: parking_lot::Mutex<Option<anyhow::Result<()>>>,
        stop_result: parking_lot::Mutex<Option<anyhow::Result<String>>>,
        state_tx: tokio::sync::watch::Sender<assistd_voice::VoiceCaptureState>,
    }

    impl MockVoice {
        fn new(start: anyhow::Result<()>, stop: anyhow::Result<String>) -> Self {
            let (state_tx, _) = tokio::sync::watch::channel(assistd_voice::VoiceCaptureState::Idle);
            Self {
                start_result: parking_lot::Mutex::new(Some(start)),
                stop_result: parking_lot::Mutex::new(Some(stop)),
                state_tx,
            }
        }
    }

    #[async_trait::async_trait]
    impl assistd_voice::VoiceInput for MockVoice {
        async fn start_recording(&self) -> anyhow::Result<()> {
            self.start_result.lock().take().unwrap_or_else(|| Ok(()))
        }
        async fn stop_and_transcribe(&self) -> anyhow::Result<String> {
            self.stop_result
                .lock()
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

        // No Delta event should appear: the query is skipped.
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
            self.calls.lock().clone()
        }

        fn wait_idle_count(&self) -> usize {
            self.wait_idle_calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait::async_trait]
    impl assistd_voice::VoiceOutput for MockSpeechRecorder {
        async fn speak(&self, text: String) -> anyhow::Result<()> {
            self.calls.lock().push(text);
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
        ) -> assistd_llm::LlmResult<()> {
            unimplemented!("uses step path")
        }
        async fn push_user(
            &self,
            _text: String,
            _attachments: Vec<assistd_tools::Attachment>,
        ) -> assistd_llm::LlmResult<()> {
            Ok(())
        }
        async fn push_tool_results(
            &self,
            _results: Vec<ToolResultPayload>,
        ) -> assistd_llm::LlmResult<()> {
            Ok(())
        }
        async fn step(
            &self,
            _tools: Vec<serde_json::Value>,
            tx: mpsc::Sender<LlmEvent>,
        ) -> assistd_llm::LlmResult<StepOutcome> {
            let script = self.script.lock().take();
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
        ) -> assistd_llm::LlmResult<()> {
            unimplemented!("uses step path")
        }
        async fn push_user(
            &self,
            _text: String,
            _attachments: Vec<assistd_tools::Attachment>,
        ) -> assistd_llm::LlmResult<()> {
            Ok(())
        }
        async fn push_tool_results(
            &self,
            _results: Vec<ToolResultPayload>,
        ) -> assistd_llm::LlmResult<()> {
            Ok(())
        }
        async fn step(
            &self,
            _tools: Vec<serde_json::Value>,
            tx: mpsc::Sender<LlmEvent>,
        ) -> assistd_llm::LlmResult<StepOutcome> {
            let outcome = {
                let mut q = self.outcomes.lock();
                if q.is_empty() {
                    StepOutcome::Final
                } else {
                    q.remove(0)
                }
            };
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
                },
                tx,
            )
            .await
            .unwrap();
        let _ = collect_events(rx).await;

        let calls = recorder.calls();
        assert!(
            !calls.iter().any(|c| c == "Half a"),
            "idle flush fired during tool dispatch; inhibition broken: {calls:?}"
        );
        assert_eq!(
            calls,
            vec!["Half a done.".to_string()],
            "expected single combined utterance after tool resolves; got {calls:?}"
        );
    }

    #[tokio::test]
    async fn dispatch_query_speech_worker_drains_before_return() {
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
        assert!(block.contains("- Focused window: Alacritty - nvim ~ src/main.rs\n"));
        assert!(block.contains("- Workspace: 2\n"));
        assert!(block.contains("interacting with a terminal window."));
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
        let block = format_window_context_block(&ctx(Some("Alacritty"), None, None)).expect("Some");
        assert!(block.contains("- Focused window: Alacritty\n"));
        assert!(!block.contains(" - "));
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

    // --- Branch-handler integration tests --------------------------------
    //
    // These exercise the four `/fork` `/branches` `/switch` `/undo`
    // handlers against an SQLite-backed ConversationStore so the
    // dispatcher → writer-task → DB → in-memory replay round trip is
    // covered end-to-end. Uses an in-memory backend (EchoBackend, which
    // gets `replace_history` / `truncate_to_last_real_user` no-op
    // defaults) so the tests don't require llama-server.

    async fn fresh_branch_state() -> (
        Arc<AppState>,
        Arc<dyn ConversationStore>,
        assistd_memory::SessionId,
        assistd_memory::BranchId,
    ) {
        use assistd_memory::{SqliteConversationStore, SqliteHandle};
        use tokio::sync::watch;
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("memory.db");
        std::mem::forget(temp);
        let (_tx, rx) = watch::channel(false);
        let (handle, _writer_handle) = SqliteHandle::open(&path, rx).await.unwrap();
        let handle = Arc::new(handle);
        let conv: Arc<dyn ConversationStore> =
            Arc::new(SqliteConversationStore::new(handle.clone()));
        let (session, branch) = conv.begin_session_with_main_branch(123).await.unwrap();
        let ctx = Arc::new(ConversationContext::new(session.clone(), branch));
        let config = Config::default();
        let subsystems = Subsystems::new(
            Arc::new(EchoBackend::new()),
            PresenceManager::stub(PresenceState::Active),
            Arc::new(ToolRegistry::default()),
            Arc::new(assistd_voice::NoVoiceInput::new()),
            Arc::new(assistd_voice::NoContinuousListener::new()),
            VoiceOutputController::new(Arc::new(assistd_voice::NoVoiceOutput), true),
        );
        let memory =
            MemoryStack::disabled(config.embedding.clone()).with_conversations(conv.clone());
        let runtime = RuntimeState::new().with_conversation_ctx(ctx);
        let state = Arc::new(AppState {
            config,
            subsystems,
            memory,
            runtime,
        });
        (state, conv, session, branch)
    }

    async fn drain_events(mut rx: mpsc::Receiver<Event>) -> Vec<Event> {
        let mut out = Vec::new();
        while let Some(ev) = rx.recv().await {
            out.push(ev);
        }
        out
    }

    #[tokio::test]
    async fn fork_creates_branch_and_switches() {
        let (state, conv, session, _main_branch) = fresh_branch_state().await;
        let (tx, rx) = mpsc::channel::<Event>(16);
        state
            .clone()
            .dispatch(
                Request::Fork {
                    id: "rq".into(),
                    name: "experiment".into(),
                },
                tx,
            )
            .await
            .unwrap();
        let events = drain_events(rx).await;
        // Last event must be Done; some prior event must be BranchSwitched.
        assert!(matches!(events.last(), Some(Event::Done { .. })));
        let switched = events
            .iter()
            .find(|e| matches!(e, Event::BranchSwitched { .. }))
            .expect("expected BranchSwitched");
        if let Event::BranchSwitched {
            name,
            parent_branch_name,
            ..
        } = switched
        {
            assert_eq!(name, "experiment");
            assert_eq!(parent_branch_name.as_deref(), Some("main"));
        } else {
            unreachable!()
        }
        // Active branch in DB now points at the new branch.
        let new_current = conv.get_current_branch(&session).await.unwrap();
        assert!(new_current.is_some());
        // ConversationContext also rotated.
        let (active_session, _) = state.runtime.conversation_ctx.current().await;
        assert_eq!(active_session.0, session.0);
    }

    #[tokio::test]
    async fn fork_with_empty_name_emits_error() {
        let (state, _conv, _session, _branch) = fresh_branch_state().await;
        let (tx, rx) = mpsc::channel::<Event>(16);
        state
            .clone()
            .dispatch(
                Request::Fork {
                    id: "rq".into(),
                    name: "   ".into(),
                },
                tx,
            )
            .await
            .unwrap();
        let events = drain_events(rx).await;
        assert!(events.iter().any(|e| matches!(e, Event::Error { .. })));
    }

    #[tokio::test]
    async fn branches_lists_active_session_first() {
        let (state, conv, session, _main_branch) = fresh_branch_state().await;
        // Create an extra branch so list isn't trivial.
        let _alt = conv
            .create_branch(&session, "alt", None, None)
            .await
            .unwrap();
        let (tx, rx) = mpsc::channel::<Event>(16);
        state
            .clone()
            .dispatch(Request::Branches { id: "rq".into() }, tx)
            .await
            .unwrap();
        let events = drain_events(rx).await;
        let infos: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                Event::BranchInfo { name, .. } => Some(name.clone()),
                _ => None,
            })
            .collect();
        assert!(infos.contains(&"main".to_string()));
        assert!(infos.contains(&"alt".to_string()));
        assert!(matches!(events.last(), Some(Event::Done { .. })));
    }

    #[tokio::test]
    async fn switch_replays_history_into_event_stream() {
        let (state, conv, session, main_branch) = fresh_branch_state().await;
        // Append a couple messages to main so /switch has content.
        let turn = conv.begin_turn(&session, "hello").await.unwrap();
        conv.append_message_to_branch(
            &session,
            main_branch,
            Some(turn),
            assistd_memory::PersistedMessage::user("hello"),
        )
        .await
        .unwrap();
        conv.append_message_to_branch(
            &session,
            main_branch,
            Some(turn),
            assistd_memory::PersistedMessage::assistant_text("world"),
        )
        .await
        .unwrap();
        // Fork into "alt", then switch back to main.
        let _alt = conv.fork_branch(main_branch, "alt").await.unwrap();
        let (tx, rx) = mpsc::channel::<Event>(16);
        state
            .clone()
            .dispatch(
                Request::Switch {
                    id: "rq".into(),
                    target: "alt".into(),
                },
                tx,
            )
            .await
            .unwrap();
        let events = drain_events(rx).await;
        let history_count = events
            .iter()
            .filter(|e| matches!(e, Event::HistoryEntry { .. }))
            .count();
        assert_eq!(history_count, 2, "fork's branch_messages also has 2 rows");
        assert!(matches!(events.last(), Some(Event::Done { .. })));
    }

    #[tokio::test]
    async fn switch_unknown_branch_emits_error() {
        let (state, _conv, _session, _branch) = fresh_branch_state().await;
        let (tx, rx) = mpsc::channel::<Event>(16);
        state
            .clone()
            .dispatch(
                Request::Switch {
                    id: "rq".into(),
                    target: "no-such-branch".into(),
                },
                tx,
            )
            .await
            .unwrap();
        let events = drain_events(rx).await;
        assert!(events.iter().any(|e| matches!(e, Event::Error { .. })));
    }

    #[tokio::test]
    async fn undo_drops_last_turn_and_emits_count() {
        let (state, conv, session, main_branch) = fresh_branch_state().await;
        // Two turns on main: "first" and "second".
        let t1 = conv.begin_turn(&session, "first").await.unwrap();
        conv.append_message_to_branch(
            &session,
            main_branch,
            Some(t1),
            assistd_memory::PersistedMessage::user("first"),
        )
        .await
        .unwrap();
        conv.append_message_to_branch(
            &session,
            main_branch,
            Some(t1),
            assistd_memory::PersistedMessage::assistant_text("a"),
        )
        .await
        .unwrap();
        conv.end_turn(t1).await.unwrap();
        let t2 = conv.begin_turn(&session, "second").await.unwrap();
        conv.append_message_to_branch(
            &session,
            main_branch,
            Some(t2),
            assistd_memory::PersistedMessage::user("second"),
        )
        .await
        .unwrap();
        conv.append_message_to_branch(
            &session,
            main_branch,
            Some(t2),
            assistd_memory::PersistedMessage::assistant_text("b"),
        )
        .await
        .unwrap();
        conv.end_turn(t2).await.unwrap();
        let (tx, rx) = mpsc::channel::<Event>(16);
        state
            .clone()
            .dispatch(Request::Undo { id: "rq".into() }, tx)
            .await
            .unwrap();
        let events = drain_events(rx).await;
        let applied = events
            .iter()
            .find_map(|e| match e {
                Event::UndoApplied {
                    removed_messages,
                    last_user_text,
                    ..
                } => Some((*removed_messages, last_user_text.clone())),
                _ => None,
            })
            .expect("expected UndoApplied");
        assert_eq!(applied.0, 2);
        assert_eq!(applied.1.as_deref(), Some("second"));
        let history = conv.load_branch_history(main_branch).await.unwrap();
        assert_eq!(history.len(), 2);
    }

    #[tokio::test]
    async fn undo_on_empty_branch_returns_zero() {
        let (state, _conv, _session, _branch) = fresh_branch_state().await;
        let (tx, rx) = mpsc::channel::<Event>(16);
        state
            .clone()
            .dispatch(Request::Undo { id: "rq".into() }, tx)
            .await
            .unwrap();
        let events = drain_events(rx).await;
        let applied = events
            .iter()
            .find_map(|e| match e {
                Event::UndoApplied {
                    removed_messages, ..
                } => Some(*removed_messages),
                _ => None,
            })
            .expect("expected UndoApplied even on empty branch");
        assert_eq!(applied, 0);
    }
}
