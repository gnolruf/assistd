use super::*;
use assistd_config::ToolsOutputConfig;
use assistd_llm::{LlmBackend, LlmEvent, StepOutcome, ToolCall};
use assistd_tools::{CommandRegistry, RunTool};
use async_trait::async_trait;
use parking_lot::Mutex as StdMutex;

/// Scripted mock backend: returns queued step outcomes in order.
/// Records what was pushed so tests can assert the loop fed results
/// back correctly.
struct MockBackend {
    outcomes: StdMutex<Vec<StepOutcome>>,
    pushed_users: StdMutex<Vec<String>>,
    pushed_attachments: StdMutex<Vec<Vec<Attachment>>>,
    pushed_results: StdMutex<Vec<Vec<ToolResultPayload>>>,
    /// Counts every entry into `step` (including ones cancelled
    /// before they finish). Used by cancellation tests to verify
    /// the loop didn't iterate past a cancellation point.
    step_calls: std::sync::atomic::AtomicUsize,
    /// Optional artificial delay inside `step` so cancellation
    /// tests can fire while the step is still pending.
    slow_step: StdMutex<Option<std::time::Duration>>,
    /// Number of tool schemas offered on each completed `step`.
    step_tool_counts: StdMutex<Vec<usize>>,
    transient_notes: StdMutex<Vec<String>>,
}

impl MockBackend {
    fn with(outcomes: Vec<StepOutcome>) -> Arc<Self> {
        Arc::new(Self {
            outcomes: StdMutex::new(outcomes),
            pushed_users: StdMutex::new(Vec::new()),
            pushed_attachments: StdMutex::new(Vec::new()),
            pushed_results: StdMutex::new(Vec::new()),
            step_calls: std::sync::atomic::AtomicUsize::new(0),
            slow_step: StdMutex::new(None),
            step_tool_counts: StdMutex::new(Vec::new()),
            transient_notes: StdMutex::new(Vec::new()),
        })
    }

    fn slow_step_ms(self: Arc<Self>, ms: u64) -> Arc<Self> {
        *self.slow_step.lock() = Some(std::time::Duration::from_millis(ms));
        self
    }
}

#[async_trait]
impl LlmBackend for MockBackend {
    async fn generate(
        &self,
        _prompt: String,
        _tx: mpsc::Sender<LlmEvent>,
    ) -> assistd_llm::LlmResult<()> {
        unimplemented!("mock uses step path only")
    }

    async fn push_user(
        &self,
        text: String,
        attachments: Vec<Attachment>,
    ) -> assistd_llm::LlmResult<()> {
        self.pushed_users.lock().push(text);
        self.pushed_attachments.lock().push(attachments);
        Ok(())
    }

    async fn push_tool_results(
        &self,
        results: Vec<ToolResultPayload>,
    ) -> assistd_llm::LlmResult<()> {
        self.pushed_results.lock().push(results);
        Ok(())
    }

    async fn step(
        &self,
        tools: Vec<Value>,
        tx: mpsc::Sender<LlmEvent>,
    ) -> assistd_llm::LlmResult<StepOutcome> {
        self.step_calls
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let delay = *self.slow_step.lock();
        if let Some(d) = delay {
            tokio::time::sleep(d).await;
        }
        self.step_tool_counts.lock().push(tools.len());
        let outcome = {
            let mut q = self.outcomes.lock();
            if q.is_empty() {
                StepOutcome::Final
            } else {
                q.remove(0)
            }
        };
        if matches!(outcome, StepOutcome::Final) {
            let _ = tx.send(LlmEvent::Delta { text: "ok".into() }).await;
        }
        Ok(outcome)
    }

    async fn set_transient_note(&self, text: String) -> assistd_llm::LlmResult<()> {
        self.transient_notes.lock().push(text);
        Ok(())
    }
}

const TOOL_DEADLINE: Duration = Duration::from_secs(300);

/// A `run` tool whose invocation never completes.
struct HangingTool {
    entered: Arc<tokio::sync::Notify>,
}

#[async_trait]
impl assistd_tools::Tool for HangingTool {
    fn name(&self) -> &str {
        "run"
    }
    fn description(&self) -> &str {
        "never returns"
    }
    fn parameters_schema(&self) -> Value {
        serde_json::json!({"type":"object"})
    }
    async fn invoke(&self, _args: Value) -> Result<Value, assistd_tools::ToolError> {
        self.entered.notify_one();
        std::future::pending::<()>().await;
        unreachable!("hanging tool must never resolve")
    }
}

fn call(id: &str, command: &str) -> ToolCall {
    ToolCall {
        id: id.into(),
        name: "run".into(),
        arguments: serde_json::json!({ "command": command }),
    }
}

fn tools_with_echo() -> Arc<ToolRegistry> {
    use assistd_tools::commands::EchoCommand;
    let mut reg = CommandRegistry::new();
    reg.register(EchoCommand);
    let mut tools = ToolRegistry::new();
    tools.register(RunTool::new(
        Arc::new(reg),
        &ToolsOutputConfig::default(),
        std::env::temp_dir().join(format!("assistd-agent-test-{}", std::process::id())),
    ));
    Arc::new(tools)
}

async fn collect(rx: &mut mpsc::Receiver<LlmEvent>) -> Vec<LlmEvent> {
    let mut out = Vec::new();
    while let Some(ev) = rx.recv().await {
        out.push(ev);
    }
    out
}

#[tokio::test]
async fn simple_query_one_step_final() {
    let backend = MockBackend::with(vec![StepOutcome::Final]);
    let tools = tools_with_echo();
    let (tx, mut rx) = mpsc::channel(16);
    Agent::new(backend.clone(), tools, None, TOOL_DEADLINE)
        .run_turn(
            "what is 2+2?".into(),
            Vec::new(),
            tx,
            CancellationToken::new(),
        )
        .await
        .unwrap();
    let events = collect(&mut rx).await;
    assert!(matches!(events.last(), Some(LlmEvent::Done)));
    assert_eq!(backend.pushed_users.lock().len(), 1);
    // No tool calls dispatched → no tool_results pushed.
    assert!(backend.pushed_results.lock().is_empty());
}

#[tokio::test]
async fn multi_step_tool_then_final() {
    let backend = MockBackend::with(vec![
        StepOutcome::ToolCalls(vec![call("c-1", "echo hello")]),
        StepOutcome::Final,
    ]);
    let tools = tools_with_echo();
    let (tx, mut rx) = mpsc::channel(16);
    Agent::new(backend.clone(), tools, None, TOOL_DEADLINE)
        .run_turn("say hello".into(), Vec::new(), tx, CancellationToken::new())
        .await
        .unwrap();
    let events = collect(&mut rx).await;

    assert!(
        events
            .iter()
            .any(|e| matches!(e, LlmEvent::ToolCall { name, .. } if name == "run"))
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, LlmEvent::ToolResult { name, .. } if name == "run"))
    );
    assert!(matches!(events.last(), Some(LlmEvent::Done)));

    let pushed = backend.pushed_results.lock();
    assert_eq!(pushed.len(), 1);
    assert_eq!(pushed[0].len(), 1);
    assert!(
        pushed[0][0].content.contains("hello"),
        "result content should contain echo output: {:?}",
        pushed[0][0].content
    );
}

#[tokio::test]
async fn piped_command_completes_in_one_iteration() {
    // A single `run` call with pipes; no need for multiple iterations.
    let backend = MockBackend::with(vec![
        StepOutcome::ToolCalls(vec![call(
            "c-1",
            "echo \"alpha\nbeta\nalpha\" | grep alpha | wc -l",
        )]),
        StepOutcome::Final,
    ]);
    use assistd_tools::commands::{EchoCommand, GrepCommand, WcCommand};
    let mut reg = CommandRegistry::new();
    reg.register(EchoCommand);
    reg.register(GrepCommand);
    reg.register(WcCommand);
    let mut tools = ToolRegistry::new();
    tools.register(RunTool::new(
        Arc::new(reg),
        &ToolsOutputConfig::default(),
        std::env::temp_dir().join(format!("assistd-agent-test-pipe-{}", std::process::id())),
    ));
    let tools = Arc::new(tools);

    let (tx, mut rx) = mpsc::channel(16);
    Agent::new(backend.clone(), tools, None, TOOL_DEADLINE)
        .run_turn("how many?".into(), Vec::new(), tx, CancellationToken::new())
        .await
        .unwrap();
    let events = collect(&mut rx).await;
    // Exactly one ToolCall/ToolResult pair.
    let calls = events
        .iter()
        .filter(|e| matches!(e, LlmEvent::ToolCall { .. }))
        .count();
    assert_eq!(calls, 1);

    let results = backend.pushed_results.lock();
    assert_eq!(results.len(), 1);
    assert!(
        results[0][0].content.contains("[exit:0"),
        "expected success footer: {:?}",
        results[0][0].content
    );
}

fn withdrawn_status(events: &[LlmEvent]) -> bool {
    events.iter().any(|e| {
        matches!(
            e,
            LlmEvent::Status {
                event: StatusKind::ToolsWithdrawn,
                ..
            }
        )
    })
}

#[tokio::test]
async fn repeated_identical_calls_withdraw_tools_then_answer() {
    // Same command every step; once the queue drains the mock
    // answers with text, standing in for a model that honours the
    // withdrawal note.
    let outcomes: Vec<StepOutcome> = (0..DUPLICATE_CALL_LIMIT)
        .map(|i| StepOutcome::ToolCalls(vec![call(&format!("c-{i}"), "echo same")]))
        .collect();
    let backend = MockBackend::with(outcomes);
    let tools = tools_with_echo();
    let (tx, mut rx) = mpsc::channel(64);
    Agent::new(backend.clone(), tools, None, TOOL_DEADLINE)
        .run_turn("loop it".into(), Vec::new(), tx, CancellationToken::new())
        .await
        .unwrap();
    let events = collect(&mut rx).await;

    assert!(withdrawn_status(&events), "expected tools_withdrawn status");
    assert!(matches!(events.last(), Some(LlmEvent::Done)));
    // Three dispatched duplicates, then one answer step. The schema is
    // still offered on it so a disobedient call is parsed, not leaked.
    assert_eq!(backend.pushed_results.lock().len(), DUPLICATE_CALL_LIMIT);
    assert_eq!(*backend.step_tool_counts.lock(), vec![1, 1, 1, 1]);
    let notes = backend.transient_notes.lock();
    assert_eq!(notes.len(), 1);
    assert!(
        notes[0].contains("repeated"),
        "note should say why: {notes:?}"
    );
}

#[tokio::test]
async fn distinct_calls_are_not_treated_as_repeats() {
    let outcomes: Vec<StepOutcome> = (0..5)
        .map(|i| StepOutcome::ToolCalls(vec![call(&format!("c-{i}"), &format!("echo {i}"))]))
        .collect();
    let backend = MockBackend::with(outcomes);
    let tools = tools_with_echo();
    let (tx, mut rx) = mpsc::channel(64);
    Agent::new(backend.clone(), tools, None, TOOL_DEADLINE)
        .run_turn("go".into(), Vec::new(), tx, CancellationToken::new())
        .await
        .unwrap();
    let events = collect(&mut rx).await;

    assert!(!withdrawn_status(&events));
    assert_eq!(backend.pushed_results.lock().len(), 5);
    assert!(backend.transient_notes.lock().is_empty());
}

#[tokio::test]
async fn step_ceiling_withdraws_tools_then_answer() {
    let outcomes: Vec<StepOutcome> = (0..MAX_TOOL_STEPS + 10)
        .map(|i| StepOutcome::ToolCalls(vec![call(&format!("c-{i}"), &format!("echo {i}"))]))
        .collect();
    let backend = MockBackend::with(outcomes);
    let tools = tools_with_echo();
    let (tx, mut rx) = mpsc::channel(64);
    let agent = Agent::new(backend.clone(), tools, None, TOOL_DEADLINE);
    let turn = agent.run_turn("go".into(), Vec::new(), tx, CancellationToken::new());
    let (result, events) = tokio::join!(turn, collect(&mut rx));
    result.unwrap();

    assert!(withdrawn_status(&events));
    assert!(matches!(events.last(), Some(LlmEvent::Done)));
    let counts = backend.step_tool_counts.lock();
    assert_eq!(counts.len(), MAX_TOOL_STEPS as usize + 1);
    assert_eq!(counts.last(), Some(&1));
    let notes = backend.transient_notes.lock();
    assert!(
        notes[0].contains("ceiling"),
        "note should say why: {notes:?}"
    );
}

#[tokio::test]
async fn tool_request_after_withdrawal_ends_turn_with_synthetic_results() {
    let outcomes: Vec<StepOutcome> = (0..4)
        .map(|i| StepOutcome::ToolCalls(vec![call(&format!("c-{i}"), "echo same")]))
        .collect();
    let backend = MockBackend::with(outcomes);
    let tools = tools_with_echo();
    let (tx, mut rx) = mpsc::channel(64);
    Agent::new(backend.clone(), tools, None, TOOL_DEADLINE)
        .run_turn("loop it".into(), Vec::new(), tx, CancellationToken::new())
        .await
        .unwrap();
    let events = collect(&mut rx).await;

    assert!(matches!(events.last(), Some(LlmEvent::Done)));
    let pushed = backend.pushed_results.lock();
    assert_eq!(pushed.len(), DUPLICATE_CALL_LIMIT + 1);
    assert!(
        pushed[DUPLICATE_CALL_LIMIT][0]
            .content
            .contains("tools are unavailable"),
        "expected synthetic result: {:?}",
        pushed[DUPLICATE_CALL_LIMIT][0].content
    );
    assert_eq!(
        backend.step_calls.load(std::sync::atomic::Ordering::SeqCst),
        DUPLICATE_CALL_LIMIT + 1
    );
}

#[tokio::test]
async fn unknown_tool_passes_error_to_next_step() {
    let backend = MockBackend::with(vec![
        StepOutcome::ToolCalls(vec![ToolCall {
            id: "c-1".into(),
            name: "nonexistent".into(),
            arguments: serde_json::json!({}),
        }]),
        StepOutcome::Final,
    ]);
    let tools = tools_with_echo();
    let (tx, mut rx) = mpsc::channel(16);
    Agent::new(backend.clone(), tools, None, TOOL_DEADLINE)
        .run_turn("go".into(), Vec::new(), tx, CancellationToken::new())
        .await
        .unwrap();
    drop(collect(&mut rx).await);
    let pushed = backend.pushed_results.lock();
    assert_eq!(pushed.len(), 1);
    let payload = &pushed[0][0];
    assert!(
        payload.content.starts_with("[error] agent: unknown tool"),
        "expected unknown-tool error prefix: {:?}",
        payload.content
    );
}

#[tokio::test]
async fn tool_invoke_err_becomes_synthetic_error_result() {
    struct ErrTool;
    #[async_trait]
    impl assistd_tools::Tool for ErrTool {
        fn name(&self) -> &str {
            "run"
        }
        fn description(&self) -> &str {
            "errors on invoke"
        }
        fn parameters_schema(&self) -> Value {
            serde_json::json!({"type":"object"})
        }
        async fn invoke(&self, _args: Value) -> Result<Value, assistd_tools::ToolError> {
            Err(assistd_tools::ToolError::InvalidArgs("boom".into()))
        }
    }
    let mut reg = ToolRegistry::new();
    reg.register(ErrTool);
    let reg = Arc::new(reg);

    let backend = MockBackend::with(vec![
        StepOutcome::ToolCalls(vec![call("c-1", "whatever")]),
        StepOutcome::Final,
    ]);
    let (tx, mut rx) = mpsc::channel(16);
    Agent::new(backend.clone(), reg, None, TOOL_DEADLINE)
        .run_turn("go".into(), Vec::new(), tx, CancellationToken::new())
        .await
        .unwrap();
    drop(collect(&mut rx).await);
    let pushed = backend.pushed_results.lock();
    let payload = &pushed[0][0];
    assert!(
        payload
            .content
            .starts_with("[error] run: tool invocation failed"),
        "expected tool-error prefix: {:?}",
        payload.content
    );
}

#[tokio::test]
async fn client_disconnect_between_iterations_stops_loop() {
    let backend = MockBackend::with(vec![
        StepOutcome::ToolCalls(vec![call("c-1", "echo a")]),
        // If we ever reach this outcome it's a bug.
        StepOutcome::ToolCalls(vec![call("c-2", "echo b")]),
    ]);
    let tools = tools_with_echo();
    let (tx, rx) = mpsc::channel::<LlmEvent>(16);
    drop(rx);
    // Channel is closed from the start, so the very first is_closed
    // check bails the loop before any step runs.
    Agent::new(backend.clone(), tools, None, TOOL_DEADLINE)
        .run_turn("go".into(), Vec::new(), tx, CancellationToken::new())
        .await
        .unwrap();
    let pushed = backend.pushed_results.lock();
    assert!(
        pushed.is_empty(),
        "no tool results should be pushed after disconnect: {pushed:?}"
    );
}

#[tokio::test]
async fn explicit_cancel_before_first_step_stops_loop_immediately() {
    let backend = MockBackend::with(vec![
        StepOutcome::Final,
        // If we get here the cancellation didn't take effect.
        StepOutcome::Final,
    ]);
    let tools = tools_with_echo();
    let (tx, _rx) = mpsc::channel::<LlmEvent>(16);
    let token = CancellationToken::new();
    token.cancel();
    Agent::new(backend.clone(), tools, None, TOOL_DEADLINE)
        .run_turn("go".into(), Vec::new(), tx, token)
        .await
        .unwrap();
    assert_eq!(backend.pushed_users.lock().len(), 1);
    assert_eq!(
        backend.step_calls.load(std::sync::atomic::Ordering::SeqCst),
        0
    );
}

#[tokio::test]
async fn agent_loop_routes_remember_then_recall() {
    use assistd_memory::{
        ConversationStore, MemoryStore, SqliteConversationStore, SqliteHandle, SqliteMemoryStore,
    };
    use assistd_tools::{MemoryOps, RecallTool, RememberTool};
    use tokio::sync::watch;

    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("memory.db");
    // Leak the tempdir for the test: `path` must outlive the
    // handle, and cleanup happens at process exit.
    std::mem::forget(temp);
    let (_tx, rx) = watch::channel(false);
    let (handle, _writer) = SqliteHandle::open(&path, rx).await.unwrap();
    let handle = Arc::new(handle);
    let mem: Arc<dyn MemoryStore> = Arc::new(SqliteMemoryStore::new(handle.clone()));
    let conv: Arc<dyn ConversationStore> = Arc::new(SqliteConversationStore::new(handle));
    let memory_ops = Arc::new(MemoryOps::new(mem.clone(), conv));

    let mut tools = ToolRegistry::new();
    let (embed_tx, embed_rx) = tokio::sync::mpsc::channel::<assistd_embed::EmbedJob>(1);
    drop(embed_rx);
    let no_embedder: Arc<dyn assistd_embed::Embedder> = Arc::new(assistd_embed::NoEmbedder);
    let no_semantic: Arc<dyn assistd_memory::SemanticStore> =
        Arc::new(assistd_memory::NoSemanticStore);
    tools.register(RememberTool::new(memory_ops, embed_tx));
    tools.register(RecallTool::new(no_embedder, no_semantic, String::new()));
    let tools = Arc::new(tools);

    let backend = MockBackend::with(vec![
        StepOutcome::ToolCalls(vec![ToolCall {
            id: "c-1".into(),
            name: "remember".into(),
            arguments: serde_json::json!({
                "key": "editor_preference",
                "value": "vim",
            }),
        }]),
        StepOutcome::ToolCalls(vec![ToolCall {
            id: "c-2".into(),
            name: "recall".into(),
            arguments: serde_json::json!({"query": "what editor do I prefer"}),
        }]),
        StepOutcome::Final,
    ]);

    let (tx, mut rx) = mpsc::channel(32);
    Agent::new(backend.clone(), tools, None, TOOL_DEADLINE)
        .run_turn(
            "I prefer vim over emacs".into(),
            Vec::new(),
            tx,
            CancellationToken::new(),
        )
        .await
        .unwrap();
    let events = collect(&mut rx).await;

    // Two ToolCalls emitted, one for each step.
    let names: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            LlmEvent::ToolCall { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(names, vec!["remember", "recall"]);

    assert_eq!(
        mem.load("editor_preference").await.unwrap().as_deref(),
        Some("vim")
    );

    let recall_output = events
        .iter()
        .find_map(|e| match e {
            LlmEvent::ToolResult { name, result, .. } if name == "recall" => Some(
                result
                    .get("output")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
            ),
            _ => None,
        })
        .expect("expected a recall ToolResult");
    assert_eq!(recall_output, "(no memories)");

    let pushed = backend.pushed_results.lock();
    assert_eq!(pushed.len(), 2);
    assert_eq!(pushed[1][0].content, "(no memories)");
}

struct FakeMcpTool {
    name: String,
    result: Value,
}

#[async_trait]
impl assistd_tools::Tool for FakeMcpTool {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        "fake mcp tool"
    }
    fn parameters_schema(&self) -> Value {
        serde_json::json!({"type": "object"})
    }
    async fn invoke(&self, _args: Value) -> Result<Value, assistd_tools::ToolError> {
        Ok(self.result.clone())
    }
}

#[tokio::test]
async fn agent_loop_mixes_native_and_mcp_calls() {
    use assistd_tools::commands::EchoCommand;
    let mut reg = CommandRegistry::new();
    reg.register(EchoCommand);
    let mut tools = ToolRegistry::new();
    tools.register(RunTool::new(
        Arc::new(reg),
        &ToolsOutputConfig::default(),
        std::env::temp_dir().join(format!("assistd-agent-test-mix-{}", std::process::id())),
    ));
    tools.register(FakeMcpTool {
        name: "mcp__google_calendar__list_events".into(),
        result: serde_json::json!({
            "type": "text",
            "output": "10:00 standup\n14:00 review",
            "exit_code": 0,
            "duration_ms": 12,
            "truncated": false,
        }),
    });
    let tools = Arc::new(tools);

    let backend = MockBackend::with(vec![
        // Step 1: model calls the MCP tool.
        StepOutcome::ToolCalls(vec![ToolCall {
            id: "c-mcp".into(),
            name: "mcp__google_calendar__list_events".into(),
            arguments: serde_json::json!({"date": "tomorrow"}),
        }]),
        // Step 2: model follows up with a native `run` call,
        // proving the loop accepts a different tool on the next
        // step without state leaking from the prior MCP call.
        StepOutcome::ToolCalls(vec![call("c-run", "echo hello")]),
        StepOutcome::Final,
    ]);

    let (tx, mut rx) = mpsc::channel(32);
    Agent::new(backend.clone(), tools, None, TOOL_DEADLINE)
        .run_turn(
            "what's on my calendar tomorrow?".into(),
            Vec::new(),
            tx,
            CancellationToken::new(),
        )
        .await
        .unwrap();
    let events = collect(&mut rx).await;

    let names: Vec<String> = events
        .iter()
        .filter_map(|e| match e {
            LlmEvent::ToolCall { name, .. } => Some(name.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        names,
        vec!["mcp__google_calendar__list_events", "run"],
        "tool calls didn't fire in the scripted order"
    );

    let pushed = backend.pushed_results.lock();
    assert_eq!(pushed.len(), 2, "two push_tool_results round-trips");
    assert!(
        pushed[0][0].content.contains("standup"),
        "MCP body must reach next step: {:?}",
        pushed[0][0].content
    );
    assert!(
        pushed[1][0].content.contains("hello"),
        "native echo body must reach next step: {:?}",
        pushed[1][0].content
    );
}

#[tokio::test]
async fn agent_loop_propagates_mcp_error_envelope_to_next_step() {
    let mut tools = ToolRegistry::new();
    tools.register(FakeMcpTool {
        name: "mcp__google_calendar__list_events".into(),
        result: serde_json::json!({
            "type": "error",
            "output": "[error] mcp__google_calendar__list_events: \
                       MCP server returned error code -32602: Invalid params. \
                       Check: the arguments and try again\n",
            "exit_code": -1,
            "duration_ms": 7,
            "truncated": false,
        }),
    });
    let tools = Arc::new(tools);

    let backend = MockBackend::with(vec![
        StepOutcome::ToolCalls(vec![ToolCall {
            id: "c-mcp".into(),
            name: "mcp__google_calendar__list_events".into(),
            arguments: serde_json::json!({}),
        }]),
        StepOutcome::Final,
    ]);
    let (tx, mut rx) = mpsc::channel(16);
    Agent::new(backend.clone(), tools, None, TOOL_DEADLINE)
        .run_turn(
            "what's on my calendar?".into(),
            Vec::new(),
            tx,
            CancellationToken::new(),
        )
        .await
        .unwrap();
    drop(collect(&mut rx).await);

    let pushed = backend.pushed_results.lock();
    assert_eq!(pushed.len(), 1);
    let payload = &pushed[0][0];
    assert!(
        payload
            .content
            .starts_with("[error] mcp__google_calendar__list_events: "),
        "convention prefix lost: {:?}",
        payload.content
    );
    assert!(
        payload.content.contains("Check:"),
        "recovery hint stripped before reaching the model: {:?}",
        payload.content
    );
    assert!(
        payload.content.contains("-32602"),
        "rpc code lost: {:?}",
        payload.content
    );
}

#[tokio::test]
async fn cancellation_during_slow_step_preempts_loop() {
    let backend = MockBackend::with(vec![StepOutcome::Final]).slow_step_ms(2_000);
    let tools = tools_with_echo();
    let (tx, _rx) = mpsc::channel::<LlmEvent>(16);
    let token = CancellationToken::new();
    let token_for_kicker = token.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        token_for_kicker.cancel();
    });
    let started = std::time::Instant::now();
    Agent::new(backend.clone(), tools, None, TOOL_DEADLINE)
        .run_turn("go".into(), Vec::new(), tx, token)
        .await
        .unwrap();
    let elapsed = started.elapsed();
    assert!(
        elapsed < std::time::Duration::from_millis(500),
        "cancellation did not preempt slow step (elapsed: {elapsed:?})"
    );
}

#[tokio::test]
async fn cancellation_during_hung_tool_preempts_dispatch() {
    let entered = Arc::new(tokio::sync::Notify::new());
    let mut reg = ToolRegistry::new();
    reg.register(HangingTool {
        entered: entered.clone(),
    });
    let backend = MockBackend::with(vec![
        StepOutcome::ToolCalls(vec![call("c-1", "hang")]),
        // Reaching this outcome would mean the loop iterated past
        // the cancellation point.
        StepOutcome::ToolCalls(vec![call("c-2", "echo b")]),
    ]);
    let (tx, mut rx) = mpsc::channel::<LlmEvent>(16);
    let token = CancellationToken::new();
    let token_for_kicker = token.clone();
    tokio::spawn(async move {
        entered.notified().await;
        token_for_kicker.cancel();
    });

    let agent = Agent::new(backend.clone(), Arc::new(reg), None, TOOL_DEADLINE);
    let turn = agent.run_turn("go".into(), Vec::new(), tx, token);
    tokio::time::timeout(std::time::Duration::from_secs(5), turn)
        .await
        .expect("cancellation did not preempt hung tool")
        .unwrap();

    assert_eq!(
        backend.step_calls.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "loop iterated past the cancellation point"
    );

    let events = collect(&mut rx).await;
    assert!(
        events
            .iter()
            .any(|e| matches!(e, LlmEvent::ToolResult { id, .. } if id == "c-1")),
        "abandoned tool call left without a ToolResult event: {events:?}"
    );

    let pushed = backend.pushed_results.lock();
    assert_eq!(pushed.len(), 1, "cancelled call was not answered");
    assert_eq!(pushed[0].len(), 1);
    assert_eq!(pushed[0][0].call_id, "c-1");
    assert!(
        pushed[0][0].content.contains("cancelled during dispatch"),
        "unexpected cancelled payload: {:?}",
        pushed[0][0].content
    );
}

#[tokio::test]
async fn hung_tool_past_deadline_becomes_error_result_and_turn_continues() {
    let mut reg = ToolRegistry::new();
    reg.register(HangingTool {
        entered: Arc::new(tokio::sync::Notify::new()),
    });
    let backend = MockBackend::with(vec![
        StepOutcome::ToolCalls(vec![call("c-1", "hang")]),
        StepOutcome::Final,
    ]);
    let (tx, _rx) = mpsc::channel::<LlmEvent>(16);

    let agent = Agent::new(
        backend.clone(),
        Arc::new(reg),
        None,
        Duration::from_millis(100),
    );
    let turn = agent.run_turn("go".into(), Vec::new(), tx, CancellationToken::new());
    tokio::time::timeout(Duration::from_secs(5), turn)
        .await
        .expect("tool deadline did not fire")
        .unwrap();

    assert_eq!(
        backend.step_calls.load(std::sync::atomic::Ordering::SeqCst),
        2,
        "turn did not continue after the abandoned call"
    );
    let pushed = backend.pushed_results.lock();
    assert_eq!(pushed.len(), 1);
    assert_eq!(pushed[0][0].call_id, "c-1");
    assert!(
        pushed[0][0].content.contains("call abandoned"),
        "unexpected payload: {:?}",
        pushed[0][0].content
    );
}

struct MockProbe {
    pid: StdMutex<Option<u32>>,
    state: StdMutex<assistd_llm::ReadyState>,
    wait_result: StdMutex<Option<Result<(), assistd_llm::HealthWaitError>>>,
    wait_calls: std::sync::atomic::AtomicUsize,
}

impl MockProbe {
    fn ready_with_wait_ok(pid: u32) -> Arc<Self> {
        Arc::new(Self {
            pid: StdMutex::new(Some(pid)),
            state: StdMutex::new(assistd_llm::ReadyState::Ready),
            wait_result: StdMutex::new(Some(Ok(()))),
            wait_calls: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    fn ready_with_wait_err(err: assistd_llm::HealthWaitError) -> Arc<Self> {
        Arc::new(Self {
            pid: StdMutex::new(Some(99)),
            state: StdMutex::new(assistd_llm::ReadyState::Degraded),
            wait_result: StdMutex::new(Some(Err(err))),
            wait_calls: std::sync::atomic::AtomicUsize::new(0),
        })
    }
}

#[async_trait]
impl LlmHealthProbe for MockProbe {
    fn pid(&self) -> Option<u32> {
        *self.pid.lock()
    }

    fn state(&self) -> Option<assistd_llm::ReadyState> {
        Some(*self.state.lock())
    }

    async fn wait_for_ready(
        &self,
        _budget: std::time::Duration,
    ) -> Result<(), assistd_llm::HealthWaitError> {
        self.wait_calls
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        (*self.wait_result.lock()).unwrap_or(Ok(()))
    }
}

struct ErrorInjectingBackend {
    errors: StdMutex<Vec<LlmError>>,
    outcomes: StdMutex<Vec<StepOutcome>>,
    step_calls: std::sync::atomic::AtomicUsize,
    pushed_users: StdMutex<Vec<String>>,
}

impl ErrorInjectingBackend {
    fn new(errors: Vec<LlmError>, outcomes: Vec<StepOutcome>) -> Arc<Self> {
        Arc::new(Self {
            errors: StdMutex::new(errors),
            outcomes: StdMutex::new(outcomes),
            step_calls: std::sync::atomic::AtomicUsize::new(0),
            pushed_users: StdMutex::new(Vec::new()),
        })
    }
}

#[async_trait]
impl LlmBackend for ErrorInjectingBackend {
    async fn generate(
        &self,
        _prompt: String,
        _tx: mpsc::Sender<LlmEvent>,
    ) -> assistd_llm::LlmResult<()> {
        unimplemented!("mock uses step path only")
    }

    async fn push_user(
        &self,
        text: String,
        _attachments: Vec<Attachment>,
    ) -> assistd_llm::LlmResult<()> {
        self.pushed_users.lock().push(text);
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
        _tools: Vec<Value>,
        tx: mpsc::Sender<LlmEvent>,
    ) -> assistd_llm::LlmResult<StepOutcome> {
        self.step_calls
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let next_err = self.errors.lock().pop();
        if let Some(e) = next_err {
            return Err(e);
        }
        let outcome = self.outcomes.lock().pop().unwrap_or(StepOutcome::Final);
        if matches!(outcome, StepOutcome::Final) {
            let _ = tx.send(LlmEvent::Delta { text: "ok".into() }).await;
        }
        Ok(outcome)
    }
}

#[tokio::test]
async fn replay_once_on_server_restarting() {
    let backend = ErrorInjectingBackend::new(
        vec![LlmError::ServerRestarting("boom".into())],
        vec![StepOutcome::Final],
    );
    let probe = MockProbe::ready_with_wait_ok(123);
    let tools = tools_with_echo();
    let (tx, mut rx) = mpsc::channel(32);
    Agent::new(backend.clone(), tools, Some(probe.clone()), TOOL_DEADLINE)
        .run_turn("hello".into(), Vec::new(), tx, CancellationToken::new())
        .await
        .expect("replay should succeed");

    let events = collect(&mut rx).await;
    let restart_count = events
        .iter()
        .filter(|e| {
            matches!(
                e,
                LlmEvent::Status {
                    event: StatusKind::Restarting,
                    ..
                }
            )
        })
        .count();
    let replaying_count = events
        .iter()
        .filter(|e| {
            matches!(
                e,
                LlmEvent::Status {
                    event: StatusKind::Replaying,
                    ..
                }
            )
        })
        .count();
    assert_eq!(restart_count, 1, "expected one restarting Status event");
    assert_eq!(replaying_count, 1, "expected one replaying Status event");
    assert!(matches!(events.last(), Some(LlmEvent::Done)));
    assert_eq!(
        probe.wait_calls.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "wait_for_ready should be called exactly once"
    );
    // Two step calls: the failing one and the successful retry.
    assert_eq!(
        backend.step_calls.load(std::sync::atomic::Ordering::SeqCst),
        2
    );
}

#[tokio::test]
async fn replay_does_not_loop_on_repeated_server_restarting() {
    let backend = ErrorInjectingBackend::new(
        vec![
            LlmError::ServerRestarting("second".into()),
            LlmError::ServerRestarting("first".into()),
        ],
        vec![],
    );
    let probe = MockProbe::ready_with_wait_ok(123);
    let tools = tools_with_echo();
    let (tx, mut rx) = mpsc::channel(32);
    let result = Agent::new(backend.clone(), tools, Some(probe), TOOL_DEADLINE)
        .run_turn("hello".into(), Vec::new(), tx, CancellationToken::new())
        .await;
    assert!(result.is_err(), "second ServerRestarting must be terminal");

    let events = collect(&mut rx).await;
    let restart_count = events
        .iter()
        .filter(|e| {
            matches!(
                e,
                LlmEvent::Status {
                    event: StatusKind::Restarting,
                    ..
                }
            )
        })
        .count();
    assert_eq!(
        restart_count, 1,
        "should attempt replay only once, then surface error"
    );
    assert!(matches!(events.last(), Some(LlmEvent::Done)));
    // Two step calls: original + one replay attempt.
    assert_eq!(
        backend.step_calls.load(std::sync::atomic::Ordering::SeqCst),
        2
    );
}

#[tokio::test]
async fn replay_abandons_on_degraded_supervisor() {
    let backend =
        ErrorInjectingBackend::new(vec![LlmError::ServerRestarting("dead".into())], vec![]);
    let probe = MockProbe::ready_with_wait_err(assistd_llm::HealthWaitError::Degraded);
    let tools = tools_with_echo();
    let (tx, mut rx) = mpsc::channel(32);
    let result = Agent::new(backend.clone(), tools, Some(probe), TOOL_DEADLINE)
        .run_turn("hello".into(), Vec::new(), tx, CancellationToken::new())
        .await;
    assert!(result.is_err(), "Degraded probe must surface as error");

    let events = collect(&mut rx).await;
    assert!(
        events.iter().any(|e| matches!(
            e,
            LlmEvent::Status {
                event: StatusKind::Degraded,
                ..
            }
        )),
        "expected a degraded Status event"
    );
    assert!(matches!(events.last(), Some(LlmEvent::Done)));
    assert_eq!(
        backend.step_calls.load(std::sync::atomic::Ordering::SeqCst),
        1
    );
}
