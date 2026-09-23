use super::*;
use assistd_config::ToolsOutputConfig;
use assistd_llm::{LlmBackend, LlmEvent, StepOutcome, ToolCall};
use assistd_tools::{CommandRegistry, RunTool};
use async_trait::async_trait;
use parking_lot::Mutex as StdMutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Scripted mock backend: returns queued step outcomes in order, then
/// `Final`. Records what the loop pushed back.
struct MockBackend {
    outcomes: StdMutex<Vec<StepOutcome>>,
    pushed_users: StdMutex<Vec<String>>,
    pushed_results: StdMutex<Vec<Vec<ToolResultPayload>>>,
    /// Counts every entry into `step`, including ones cancelled before
    /// they finish.
    step_calls: AtomicUsize,
    slow_step: StdMutex<Option<Duration>>,
    /// Number of tool schemas offered on each completed `step`.
    step_tool_counts: StdMutex<Vec<usize>>,
    transient_notes: StdMutex<Vec<String>>,
}

impl MockBackend {
    fn with(outcomes: Vec<StepOutcome>) -> Arc<Self> {
        Arc::new(Self {
            outcomes: StdMutex::new(outcomes),
            pushed_users: StdMutex::new(Vec::new()),
            pushed_results: StdMutex::new(Vec::new()),
            step_calls: AtomicUsize::new(0),
            slow_step: StdMutex::new(None),
            step_tool_counts: StdMutex::new(Vec::new()),
            transient_notes: StdMutex::new(Vec::new()),
        })
    }

    fn slow_step_ms(self: Arc<Self>, ms: u64) -> Arc<Self> {
        *self.slow_step.lock() = Some(Duration::from_millis(ms));
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
        _attachments: Vec<Attachment>,
    ) -> assistd_llm::LlmResult<()> {
        self.pushed_users.lock().push(text);
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
        self.step_calls.fetch_add(1, Ordering::SeqCst);
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

/// Run one uncancelled turn, draining events concurrently so a long
/// turn never blocks on channel capacity.
async fn run_turn(agent: Agent, user_text: &str) -> (Result<(), LlmError>, Vec<LlmEvent>) {
    let (tx, mut rx) = mpsc::channel(16);
    let turn = agent.run_turn(user_text.into(), Vec::new(), tx, CancellationToken::new());
    tokio::join!(turn, collect(&mut rx))
}

fn status_kinds(events: &[LlmEvent]) -> Vec<StatusKind> {
    events
        .iter()
        .filter_map(|e| match e {
            LlmEvent::Status { event, .. } => Some(*event),
            _ => None,
        })
        .collect()
}

fn ok_then_done() -> [LlmEvent; 2] {
    [LlmEvent::Delta { text: "ok".into() }, LlmEvent::Done]
}

#[tokio::test]
async fn simple_query_one_step_final() {
    let backend = MockBackend::with(vec![StepOutcome::Final]);
    let (res, events) = run_turn(
        Agent::new(backend.clone(), tools_with_echo(), None, TOOL_DEADLINE),
        "what is 2+2?",
    )
    .await;
    res.unwrap();
    assert_eq!(events, ok_then_done());
    assert_eq!(*backend.pushed_users.lock(), ["what is 2+2?"]);
    assert!(backend.pushed_results.lock().is_empty());
}

#[tokio::test]
async fn tool_call_result_is_fed_back_before_final_answer() {
    let backend = MockBackend::with(vec![
        StepOutcome::ToolCalls(vec![call("c-1", "echo hello")]),
        StepOutcome::Final,
    ]);
    let (res, events) = run_turn(
        Agent::new(backend.clone(), tools_with_echo(), None, TOOL_DEADLINE),
        "say hello",
    )
    .await;
    res.unwrap();

    let [requested, tool_call, tool_result, rest @ ..] = events.as_slice() else {
        panic!("too few events: {events:?}");
    };
    assert_eq!(
        *requested,
        LlmEvent::ToolCallsRequested {
            calls: vec![call("c-1", "echo hello")]
        }
    );
    assert_eq!(
        *tool_call,
        LlmEvent::ToolCall {
            id: "c-1".into(),
            name: "run".into(),
            arguments: serde_json::json!({ "command": "echo hello" }),
        }
    );
    assert!(
        matches!(tool_result, LlmEvent::ToolResult { id, name, .. } if id == "c-1" && name == "run"),
        "{tool_result:?}"
    );
    assert_eq!(rest, ok_then_done());

    let pushed = backend.pushed_results.lock();
    let [step] = pushed.as_slice() else {
        panic!("expected one push: {pushed:?}");
    };
    let [payload] = step.as_slice() else {
        panic!("expected one result: {step:?}");
    };
    assert_eq!(payload.call_id, "c-1");
    assert!(payload.content.contains("hello"), "{:?}", payload.content);
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
    let (res, events) = run_turn(
        Agent::new(backend.clone(), tools_with_echo(), None, TOOL_DEADLINE),
        "loop it",
    )
    .await;
    res.unwrap();

    assert_eq!(status_kinds(&events), [StatusKind::ToolsWithdrawn]);
    assert_eq!(events.last(), Some(&LlmEvent::Done));
    // Three dispatched duplicates, then one answer step. The schema is
    // still offered on it so a disobedient call is parsed, not leaked.
    assert_eq!(backend.pushed_results.lock().len(), DUPLICATE_CALL_LIMIT);
    assert_eq!(*backend.step_tool_counts.lock(), [1, 1, 1, 1]);
    assert_eq!(
        *backend.transient_notes.lock(),
        [ToolBudgetExhausted::Repeating.model_note()]
    );
}

#[tokio::test]
async fn distinct_calls_are_not_treated_as_repeats() {
    let outcomes: Vec<StepOutcome> = (0..5)
        .map(|i| StepOutcome::ToolCalls(vec![call(&format!("c-{i}"), &format!("echo {i}"))]))
        .collect();
    let backend = MockBackend::with(outcomes);
    let (res, events) = run_turn(
        Agent::new(backend.clone(), tools_with_echo(), None, TOOL_DEADLINE),
        "go",
    )
    .await;
    res.unwrap();

    assert!(status_kinds(&events).is_empty());
    assert_eq!(backend.pushed_results.lock().len(), 5);
    assert!(backend.transient_notes.lock().is_empty());
}

#[tokio::test]
async fn step_ceiling_withdraws_tools_then_answer() {
    let outcomes: Vec<StepOutcome> = (0..MAX_TOOL_STEPS + 10)
        .map(|i| StepOutcome::ToolCalls(vec![call(&format!("c-{i}"), &format!("echo {i}"))]))
        .collect();
    let backend = MockBackend::with(outcomes);
    let (res, events) = run_turn(
        Agent::new(backend.clone(), tools_with_echo(), None, TOOL_DEADLINE),
        "go",
    )
    .await;
    res.unwrap();

    assert_eq!(status_kinds(&events), [StatusKind::ToolsWithdrawn]);
    assert_eq!(events.last(), Some(&LlmEvent::Done));
    let counts = backend.step_tool_counts.lock();
    assert_eq!(counts.len(), MAX_TOOL_STEPS as usize + 1);
    assert_eq!(counts.last(), Some(&1));
    assert_eq!(
        *backend.transient_notes.lock(),
        [ToolBudgetExhausted::StepCeiling.model_note()]
    );
}

#[tokio::test]
async fn tool_request_after_withdrawal_ends_turn_with_synthetic_results() {
    let outcomes: Vec<StepOutcome> = (0..=DUPLICATE_CALL_LIMIT)
        .map(|i| StepOutcome::ToolCalls(vec![call(&format!("c-{i}"), "echo same")]))
        .collect();
    let backend = MockBackend::with(outcomes);
    let (res, events) = run_turn(
        Agent::new(backend.clone(), tools_with_echo(), None, TOOL_DEADLINE),
        "loop it",
    )
    .await;
    res.unwrap();

    assert_eq!(events.last(), Some(&LlmEvent::Done));
    let pushed = backend.pushed_results.lock();
    assert_eq!(pushed.len(), DUPLICATE_CALL_LIMIT + 1);
    let [synthetic] = pushed[DUPLICATE_CALL_LIMIT].as_slice() else {
        panic!("expected one synthetic result: {pushed:?}");
    };
    assert_eq!(synthetic.call_id, format!("c-{DUPLICATE_CALL_LIMIT}"));
    assert_eq!(
        synthetic.content,
        "[error] run: agent turn ended; tools are unavailable.\n[exit:-1 | 0ms]"
    );
    assert_eq!(
        backend.step_calls.load(Ordering::SeqCst),
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
    let (res, _) = run_turn(
        Agent::new(backend.clone(), tools_with_echo(), None, TOOL_DEADLINE),
        "go",
    )
    .await;
    res.unwrap();
    let pushed = backend.pushed_results.lock();
    assert_eq!(pushed.len(), 1);
    let payload = &pushed[0][0];
    assert!(
        payload
            .content
            .starts_with("[error] agent: unknown tool 'nonexistent'. Available: run.\n[exit:-1 |"),
        "{:?}",
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

    let backend = MockBackend::with(vec![
        StepOutcome::ToolCalls(vec![call("c-1", "whatever")]),
        StepOutcome::Final,
    ]);
    let (res, _) = run_turn(
        Agent::new(backend.clone(), Arc::new(reg), None, TOOL_DEADLINE),
        "go",
    )
    .await;
    res.unwrap();
    let pushed = backend.pushed_results.lock();
    let payload = &pushed[0][0];
    assert!(
        payload.content.starts_with(
            "[error] run: tool invocation failed. Check: boom. Try: a different command.\n[exit:-1 |"
        ),
        "{:?}",
        payload.content
    );
}

#[tokio::test]
async fn closed_event_channel_stops_before_first_step() {
    let backend = MockBackend::with(vec![StepOutcome::ToolCalls(vec![call("c-1", "echo a")])]);
    let (tx, rx) = mpsc::channel::<LlmEvent>(16);
    drop(rx);
    Agent::new(backend.clone(), tools_with_echo(), None, TOOL_DEADLINE)
        .run_turn("go".into(), Vec::new(), tx, CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(backend.step_calls.load(Ordering::SeqCst), 0);
    assert!(backend.pushed_results.lock().is_empty());
}

#[tokio::test]
async fn explicit_cancel_before_first_step_stops_loop_immediately() {
    let backend = MockBackend::with(vec![StepOutcome::Final]);
    let (tx, _rx) = mpsc::channel::<LlmEvent>(16);
    let token = CancellationToken::new();
    token.cancel();
    Agent::new(backend.clone(), tools_with_echo(), None, TOOL_DEADLINE)
        .run_turn("go".into(), Vec::new(), tx, token)
        .await
        .unwrap();
    assert_eq!(backend.pushed_users.lock().len(), 1);
    assert_eq!(backend.step_calls.load(Ordering::SeqCst), 0);
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

    let backend = MockBackend::with(vec![
        StepOutcome::ToolCalls(vec![ToolCall {
            id: "c-mcp".into(),
            name: "mcp__google_calendar__list_events".into(),
            arguments: serde_json::json!({"date": "tomorrow"}),
        }]),
        StepOutcome::ToolCalls(vec![call("c-run", "echo hello")]),
        StepOutcome::Final,
    ]);

    let (res, events) = run_turn(
        Agent::new(backend.clone(), Arc::new(tools), None, TOOL_DEADLINE),
        "what's on my calendar tomorrow?",
    )
    .await;
    res.unwrap();

    let names: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            LlmEvent::ToolCall { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(names, ["mcp__google_calendar__list_events", "run"]);

    let pushed = backend.pushed_results.lock();
    assert_eq!(pushed.len(), 2);
    assert_eq!(pushed[0][0].content, "10:00 standup\n14:00 review");
    assert!(pushed[1][0].content.contains("hello"), "{:?}", pushed[1][0]);
}

#[tokio::test(start_paused = true)]
async fn cancellation_during_slow_step_preempts_loop() {
    let backend = MockBackend::with(vec![StepOutcome::Final]).slow_step_ms(2_000);
    let (tx, _rx) = mpsc::channel::<LlmEvent>(16);
    let token = CancellationToken::new();
    let kicker = token.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(20)).await;
        kicker.cancel();
    });
    Agent::new(backend.clone(), tools_with_echo(), None, TOOL_DEADLINE)
        .run_turn("go".into(), Vec::new(), tx, token)
        .await
        .unwrap();
    assert_eq!(backend.step_calls.load(Ordering::SeqCst), 1);
    assert!(
        backend.step_tool_counts.lock().is_empty(),
        "slow step ran to completion instead of being preempted"
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
        StepOutcome::ToolCalls(vec![call("c-2", "echo b")]),
    ]);
    let (tx, mut rx) = mpsc::channel::<LlmEvent>(16);
    let token = CancellationToken::new();
    let kicker = token.clone();
    tokio::spawn(async move {
        entered.notified().await;
        kicker.cancel();
    });

    let agent = Agent::new(backend.clone(), Arc::new(reg), None, TOOL_DEADLINE);
    let turn = agent.run_turn("go".into(), Vec::new(), tx, token);
    tokio::time::timeout(Duration::from_secs(5), turn)
        .await
        .expect("cancellation did not preempt hung tool")
        .unwrap();

    assert_eq!(
        backend.step_calls.load(Ordering::SeqCst),
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
    let [step] = pushed.as_slice() else {
        panic!("cancelled call was not answered exactly once: {pushed:?}");
    };
    let [payload] = step.as_slice() else {
        panic!("expected one result: {step:?}");
    };
    assert_eq!(payload.call_id, "c-1");
    assert_eq!(
        payload.content,
        "[error] run: agent turn cancelled during dispatch.\n[exit:-1 | 0ms]"
    );
}

#[tokio::test(start_paused = true)]
async fn hung_tool_past_deadline_becomes_error_result_and_turn_continues() {
    let mut reg = ToolRegistry::new();
    reg.register(HangingTool {
        entered: Arc::new(tokio::sync::Notify::new()),
    });
    let backend = MockBackend::with(vec![
        StepOutcome::ToolCalls(vec![call("c-1", "hang")]),
        StepOutcome::Final,
    ]);
    let agent = Agent::new(backend.clone(), Arc::new(reg), None, Duration::from_secs(2));
    let (res, _) = tokio::time::timeout(Duration::from_secs(5), run_turn(agent, "go"))
        .await
        .expect("tool deadline did not fire");
    res.unwrap();

    assert_eq!(
        backend.step_calls.load(Ordering::SeqCst),
        2,
        "turn did not continue after the abandoned call"
    );
    let pushed = backend.pushed_results.lock();
    assert_eq!(pushed.len(), 1);
    assert_eq!(pushed[0][0].call_id, "c-1");
    assert!(
        pushed[0][0]
            .content
            .starts_with("[error] run: no result after 2s; call abandoned."),
        "{:?}",
        pushed[0][0].content
    );
}

struct MockProbe {
    wait_result: Result<(), HealthWaitError>,
    wait_calls: AtomicUsize,
}

impl MockProbe {
    fn new(wait_result: Result<(), HealthWaitError>) -> Arc<Self> {
        Arc::new(Self {
            wait_result,
            wait_calls: AtomicUsize::new(0),
        })
    }
}

#[async_trait]
impl LlmHealthProbe for MockProbe {
    fn pid(&self) -> Option<u32> {
        None
    }

    fn state(&self) -> Option<assistd_llm::ReadyState> {
        None
    }

    async fn wait_for_ready(&self, _budget: Duration) -> Result<(), HealthWaitError> {
        self.wait_calls.fetch_add(1, Ordering::SeqCst);
        self.wait_result
    }
}

/// Fails each `step` with the next queued error, then answers `Final`.
struct ErrorInjectingBackend {
    errors: StdMutex<Vec<LlmError>>,
    step_calls: AtomicUsize,
}

impl ErrorInjectingBackend {
    fn new(errors: Vec<LlmError>) -> Arc<Self> {
        Arc::new(Self {
            errors: StdMutex::new(errors),
            step_calls: AtomicUsize::new(0),
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
        _text: String,
        _attachments: Vec<Attachment>,
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
        _tools: Vec<Value>,
        tx: mpsc::Sender<LlmEvent>,
    ) -> assistd_llm::LlmResult<StepOutcome> {
        self.step_calls.fetch_add(1, Ordering::SeqCst);
        let next_err = {
            let mut errors = self.errors.lock();
            (!errors.is_empty()).then(|| errors.remove(0))
        };
        if let Some(e) = next_err {
            return Err(e);
        }
        let _ = tx.send(LlmEvent::Delta { text: "ok".into() }).await;
        Ok(StepOutcome::Final)
    }
}

fn restart_error(err: &LlmError) -> Option<&str> {
    match err {
        LlmError::ServerRestarting(reason) => Some(reason),
        _ => None,
    }
}

#[tokio::test]
async fn replay_once_on_server_restarting() {
    let backend = ErrorInjectingBackend::new(vec![LlmError::ServerRestarting("boom".into())]);
    let probe = MockProbe::new(Ok(()));
    let (res, events) = run_turn(
        Agent::new(
            backend.clone(),
            tools_with_echo(),
            Some(probe.clone()),
            TOOL_DEADLINE,
        ),
        "hello",
    )
    .await;
    res.expect("replay should succeed");

    assert_eq!(
        status_kinds(&events),
        [StatusKind::Restarting, StatusKind::Replaying]
    );
    assert!(events.ends_with(&ok_then_done()), "{events:?}");
    assert_eq!(probe.wait_calls.load(Ordering::SeqCst), 1);
    assert_eq!(backend.step_calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn replay_does_not_loop_on_repeated_server_restarting() {
    let backend = ErrorInjectingBackend::new(vec![
        LlmError::ServerRestarting("first".into()),
        LlmError::ServerRestarting("second".into()),
    ]);
    let probe = MockProbe::new(Ok(()));
    let (res, events) = run_turn(
        Agent::new(
            backend.clone(),
            tools_with_echo(),
            Some(probe),
            TOOL_DEADLINE,
        ),
        "hello",
    )
    .await;
    let err = res.expect_err("second ServerRestarting must be terminal");
    assert_eq!(restart_error(&err), Some("second"), "{err}");

    assert_eq!(
        status_kinds(&events),
        [StatusKind::Restarting, StatusKind::Replaying]
    );
    assert_eq!(events.last(), Some(&LlmEvent::Done));
    assert_eq!(backend.step_calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn replay_abandons_on_degraded_supervisor() {
    let backend = ErrorInjectingBackend::new(vec![LlmError::ServerRestarting("dead".into())]);
    let probe = MockProbe::new(Err(HealthWaitError::Degraded));
    let (res, events) = run_turn(
        Agent::new(
            backend.clone(),
            tools_with_echo(),
            Some(probe),
            TOOL_DEADLINE,
        ),
        "hello",
    )
    .await;
    let err = res.expect_err("Degraded probe must surface as error");
    assert_eq!(
        restart_error(&err),
        Some("LLM supervisor entered degraded state; restart abandoned")
    );

    assert_eq!(
        status_kinds(&events),
        [StatusKind::Restarting, StatusKind::Degraded]
    );
    assert_eq!(events.last(), Some(&LlmEvent::Done));
    assert_eq!(backend.step_calls.load(Ordering::SeqCst), 1);
}
