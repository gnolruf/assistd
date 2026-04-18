use crate::{Config, PresenceManager, run_agent_turn};
use anyhow::Result;
use assistd_ipc::{Event, PresenceState, Request};
use assistd_llm::{LlmBackend, LlmEvent};
use assistd_tools::ToolRegistry;
use std::sync::Arc;
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
    ) -> Self {
        Self {
            config,
            llm,
            presence,
            tools,
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
        }
    }

    async fn handle_query(
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

        while let Some(llm_event) = llm_rx.recv().await {
            let wire_event = match llm_event {
                LlmEvent::Delta { text } => Event::Delta {
                    id: id.clone(),
                    text,
                },
                LlmEvent::ToolCall {
                    id: _call_id,
                    name,
                    arguments,
                } => Event::ToolCall {
                    id: id.clone(),
                    name,
                    args: arguments,
                },
                LlmEvent::ToolResult {
                    id: _call_id,
                    name,
                    result,
                } => Event::ToolResult {
                    id: id.clone(),
                    name,
                    result,
                },
                LlmEvent::Done => Event::Done { id: id.clone() },
            };
            if tx.send(wire_event).await.is_err() {
                // Client has disconnected; stop forwarding events.
                break;
            }
        }

        match generator.await {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Config;
    use assistd_config::ToolsOutputConfig;
    use assistd_llm::{EchoBackend, FailedBackend, StepOutcome, ToolCall, ToolResultPayload};
    use assistd_tools::{CommandRegistry, RunTool, commands::EchoCommand};

    fn test_state(backend: Arc<dyn LlmBackend>, initial_state: PresenceState) -> Arc<AppState> {
        Arc::new(AppState::new(
            Config::default(),
            backend,
            PresenceManager::stub(initial_state),
            Arc::new(ToolRegistry::default()),
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
}
