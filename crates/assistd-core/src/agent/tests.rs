use std::borrow::Cow;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::{Notify, mpsc};
use tokio_util::sync::CancellationToken;

use assistd_config::MAX_TOOL_STEPS;
use assistd_ipc::StatusKind;
use assistd_llm::{
    HealthSnapshot, HealthWaitError, LlmError, LlmEvent, LlmHealthProbe, ScriptedBackend,
    SlotAssignment, SlotHolder, Step, ToolCall,
};
use assistd_tools::commands::EchoCommand;
use assistd_tools::{
    CommandRegistry, OutputCap, PresentSpec, RunTool, Tool, ToolError, ToolOutput, ToolRegistry,
    TurnContext,
};

use super::definition::DUPLICATE_CALL_LIMIT;
use super::recovery::{TRUNCATION_NOTE, TRUNCATION_RETRY_LIMIT};
use super::run::ToolBudgetExhausted;
use super::{Agent, AgentDefinition, AgentHost};

const TOOL_DEADLINE: Duration = Duration::from_secs(300);

/// A backend that plays `steps`, then answers "ok".
fn scripted(steps: impl IntoIterator<Item = Step>) -> Arc<ScriptedBackend> {
    Arc::new(ScriptedBackend::new(steps).then_replying("ok"))
}

fn agent_with(
    backend: Arc<ScriptedBackend>,
    tools: Arc<ToolRegistry>,
    health: Option<Arc<dyn LlmHealthProbe>>,
    tool_deadline: Duration,
) -> Agent {
    let main = SlotAssignment::Shared(SlotHolder::Main);
    AgentHost::new(backend, health, tool_deadline).main(AgentDefinition::new(
        String::new(),
        tools,
        main,
    ))
}

/// A `run` tool whose invocation never completes.
#[derive(Debug)]
struct HangingTool {
    entered: Arc<Notify>,
}

#[async_trait]
impl Tool for HangingTool {
    fn name(&self) -> &'static str {
        "run"
    }
    fn description(&self) -> Cow<'_, str> {
        "never returns".into()
    }
    fn parameters_schema(&self) -> Value {
        serde_json::json!({"type":"object"})
    }
    async fn invoke(&self, _turn: &TurnContext, _args: Value) -> Result<ToolOutput, ToolError> {
        self.entered.notify_one();
        std::future::pending::<()>().await;
        unreachable!("hanging tool must never resolve")
    }
}

/// A `run` tool that answers after `delay` and opts out of the tool deadline.
#[derive(Debug)]
struct SelfBoundedTool {
    delay: Duration,
}

#[async_trait]
impl Tool for SelfBoundedTool {
    fn name(&self) -> &'static str {
        "run"
    }
    fn description(&self) -> Cow<'_, str> {
        "slow but bounded".into()
    }
    fn parameters_schema(&self) -> Value {
        serde_json::json!({"type":"object"})
    }
    fn bounded_by_tool_deadline(&self) -> bool {
        false
    }
    async fn invoke(&self, _turn: &TurnContext, _args: Value) -> Result<ToolOutput, ToolError> {
        tokio::time::sleep(self.delay).await;
        Ok(ToolOutput::text("finished", self.delay))
    }
}

/// A `run` tool whose invocation panics.
#[derive(Debug)]
struct PanickingTool;

#[async_trait]
impl Tool for PanickingTool {
    fn name(&self) -> &'static str {
        "run"
    }
    fn description(&self) -> Cow<'_, str> {
        "panics".into()
    }
    fn parameters_schema(&self) -> Value {
        serde_json::json!({"type":"object"})
    }
    async fn invoke(&self, _turn: &TurnContext, _args: Value) -> Result<ToolOutput, ToolError> {
        panic!("not implemented")
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
    let mut commands = CommandRegistry::new();
    commands.register(EchoCommand);
    let output_cap = Arc::new(OutputCap::new(PresentSpec {
        overflow_dir: std::env::temp_dir()
            .join(format!("assistd-agent-test-{}", std::process::id())),
        ..PresentSpec::default()
    }));
    let mut tools = ToolRegistry::with_output_cap(Arc::clone(&output_cap));
    tools.register(RunTool::new(Arc::new(commands), output_cap));
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
    let context = TurnContext::default();
    let turn = agent.run(
        user_text.to_string(),
        &context,
        tx,
        CancellationToken::new(),
    );
    let (outcome, events) = tokio::join!(turn, collect(&mut rx));
    (outcome.end.into_result(), events)
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

/// A failed turn streams no text for its failure and never sends `Done`.
fn assert_failed_without_done(events: &[LlmEvent]) {
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, LlmEvent::Done | LlmEvent::Delta { .. })),
        "{events:?}"
    );
}

fn ok_then_done() -> [LlmEvent; 2] {
    [LlmEvent::Delta { text: "ok".into() }, LlmEvent::Done]
}

#[tokio::test]
async fn tool_call_result_is_fed_back_before_final_answer() {
    let backend = scripted(vec![
        Step::tool_calls(vec![call("c-1", "echo hello")]),
        Step::reply("ok"),
    ]);
    let (res, events) = run_turn(
        agent_with(backend.clone(), tools_with_echo(), None, TOOL_DEADLINE),
        "say hello",
    )
    .await;
    res.unwrap();

    let [
        first_step,
        requested,
        tool_call,
        tool_result,
        second_step,
        rest @ ..,
    ] = events.as_slice()
    else {
        panic!("too few events: {events:?}");
    };
    assert_eq!(
        [first_step, second_step],
        [&LlmEvent::StepStarted, &LlmEvent::StepStarted]
    );
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

    let pushed = backend.seen().tool_results;
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
    let outcomes: Vec<Step> = (0..DUPLICATE_CALL_LIMIT)
        .map(|i| Step::tool_calls(vec![call(&format!("c-{i}"), "echo same")]))
        .collect();
    let backend = scripted(outcomes);
    let (res, events) = run_turn(
        agent_with(backend.clone(), tools_with_echo(), None, TOOL_DEADLINE),
        "loop it",
    )
    .await;
    res.unwrap();

    assert_eq!(status_kinds(&events), [StatusKind::ToolsWithdrawn]);
    assert_eq!(events.last(), Some(&LlmEvent::Done));
    assert_eq!(backend.seen().tool_results.len(), DUPLICATE_CALL_LIMIT);
    assert_eq!(
        backend.seen().offered_tools,
        [1, 1, 1, 1],
        "the answer step must still offer the schema so a stray call is parsed"
    );
    assert_eq!(
        backend.seen().transient_notes,
        [ToolBudgetExhausted::Repeating {
            times: DUPLICATE_CALL_LIMIT
        }
        .model_note()]
    );
}

#[tokio::test]
async fn step_ceiling_withdraws_tools_then_answer() {
    let outcomes: Vec<Step> = (0..MAX_TOOL_STEPS + 10)
        .map(|i| Step::tool_calls(vec![call(&format!("c-{i}"), &format!("echo {i}"))]))
        .collect();
    let backend = scripted(outcomes);
    let (res, events) = run_turn(
        agent_with(backend.clone(), tools_with_echo(), None, TOOL_DEADLINE),
        "go",
    )
    .await;
    res.unwrap();

    assert_eq!(status_kinds(&events), [StatusKind::ToolsWithdrawn]);
    assert_eq!(events.last(), Some(&LlmEvent::Done));
    let counts = backend.seen().offered_tools;
    assert_eq!(counts.len(), MAX_TOOL_STEPS as usize + 1);
    assert_eq!(counts.last(), Some(&1));
    assert_eq!(
        backend.seen().transient_notes,
        [ToolBudgetExhausted::StepCeiling {
            steps: MAX_TOOL_STEPS
        }
        .model_note()]
    );
}

#[tokio::test]
async fn tool_request_after_withdrawal_ends_turn_with_synthetic_results() {
    let outcomes: Vec<Step> = (0..=DUPLICATE_CALL_LIMIT)
        .map(|i| Step::tool_calls(vec![call(&format!("c-{i}"), "echo same")]))
        .collect();
    let backend = scripted(outcomes);
    let (res, events) = run_turn(
        agent_with(backend.clone(), tools_with_echo(), None, TOOL_DEADLINE),
        "loop it",
    )
    .await;
    res.unwrap();

    assert_eq!(events.last(), Some(&LlmEvent::Done));
    let pushed = backend.seen().tool_results;
    assert_eq!(pushed.len(), DUPLICATE_CALL_LIMIT + 1);
    let [synthetic] = pushed[DUPLICATE_CALL_LIMIT].as_slice() else {
        panic!("expected one synthetic result: {pushed:?}");
    };
    assert_eq!(synthetic.call_id, format!("c-{DUPLICATE_CALL_LIMIT}"));
    assert_eq!(
        synthetic.content,
        "[error] run: agent turn ended; tools are unavailable.\n[exit:-1 | 0ms]"
    );
    assert_eq!(backend.seen().offered_tools.len(), DUPLICATE_CALL_LIMIT + 1);
}

#[tokio::test]
async fn truncated_step_is_retried_with_a_note() {
    let backend = scripted(vec![Step::truncated(), Step::reply("ok")]);
    let (res, events) = run_turn(
        agent_with(backend.clone(), tools_with_echo(), None, TOOL_DEADLINE),
        "go",
    )
    .await;
    res.unwrap();

    assert_eq!(status_kinds(&events), [StatusKind::OutputTruncated]);
    assert!(events.ends_with(&ok_then_done()), "{events:?}");
    assert_eq!(backend.seen().transient_notes, [TRUNCATION_NOTE]);
    assert!(backend.seen().tool_results.is_empty());
}

#[tokio::test]
async fn truncations_past_the_retry_limit_fail_the_turn() {
    let backend = scripted((0..=TRUNCATION_RETRY_LIMIT).map(|_| Step::truncated()));
    let (res, events) = run_turn(
        agent_with(backend.clone(), tools_with_echo(), None, TOOL_DEADLINE),
        "go",
    )
    .await;

    assert!(
        matches!(res, Err(LlmError::OutputLimit(n)) if n == TRUNCATION_RETRY_LIMIT + 1),
        "{res:?}"
    );
    assert_eq!(
        status_kinds(&events),
        [StatusKind::OutputTruncated; TRUNCATION_RETRY_LIMIT as usize]
    );
    assert_failed_without_done(&events);
}

#[tokio::test]
async fn truncation_after_withdrawal_restates_the_withdrawal() {
    let mut outcomes: Vec<Step> = (0..DUPLICATE_CALL_LIMIT)
        .map(|i| Step::tool_calls(vec![call(&format!("c-{i}"), "echo same")]))
        .collect();
    outcomes.push(Step::truncated());
    let backend = scripted(outcomes);
    let (res, _events) = run_turn(
        agent_with(backend.clone(), tools_with_echo(), None, TOOL_DEADLINE),
        "loop it",
    )
    .await;
    res.unwrap();

    let withdrawal = ToolBudgetExhausted::Repeating {
        times: DUPLICATE_CALL_LIMIT,
    }
    .model_note();
    assert_eq!(
        backend.seen().transient_notes,
        [
            withdrawal.clone(),
            format!("{TRUNCATION_NOTE} {withdrawal}")
        ]
    );
}

#[tokio::test]
async fn unknown_tool_passes_error_to_next_step() {
    let backend = scripted(vec![
        Step::tool_calls(vec![ToolCall {
            id: "c-1".into(),
            name: "nonexistent".into(),
            arguments: serde_json::json!({}),
        }]),
        Step::reply("ok"),
    ]);
    let (res, _) = run_turn(
        agent_with(backend.clone(), tools_with_echo(), None, TOOL_DEADLINE),
        "go",
    )
    .await;
    res.unwrap();
    let pushed = backend.seen().tool_results;
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

#[tokio::test(start_paused = true)]
async fn cancellation_during_slow_step_preempts_loop() {
    let slow_step = Step::reply("ok").pausing(Duration::from_secs(2));
    let backend = scripted([slow_step]);
    let (tx, mut rx) = mpsc::channel::<LlmEvent>(16);
    let token = CancellationToken::new();
    let kicker = token.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(20)).await;
        kicker.cancel();
    });
    agent_with(backend.clone(), tools_with_echo(), None, TOOL_DEADLINE)
        .run("go".to_string(), &TurnContext::default(), tx, token)
        .await
        .end
        .into_result()
        .unwrap();
    assert_eq!(backend.seen().offered_tools.len(), 1);
    let events = collect(&mut rx).await;
    assert!(
        !events.contains(&LlmEvent::Done),
        "slow step ran to completion instead of being preempted: {events:?}"
    );
}

#[tokio::test]
async fn cancellation_mid_batch_answers_every_remaining_call() {
    let entered = Arc::new(Notify::new());
    let mut tools = ToolRegistry::new();
    tools.register(HangingTool {
        entered: entered.clone(),
    });
    let backend = scripted(vec![Step::tool_calls(vec![
        call("c-1", "hang"),
        call("c-2", "echo b"),
        call("c-3", "echo c"),
    ])]);
    let (tx, mut rx) = mpsc::channel::<LlmEvent>(16);
    let token = CancellationToken::new();
    let kicker = token.clone();
    tokio::spawn(async move {
        entered.notified().await;
        kicker.cancel();
    });

    let agent = agent_with(backend.clone(), Arc::new(tools), None, TOOL_DEADLINE);
    let context = TurnContext::default();
    let turn = agent.run("go".to_string(), &context, tx, token);
    tokio::time::timeout(Duration::from_secs(5), turn)
        .await
        .expect("cancellation did not preempt hung tool")
        .end
        .into_result()
        .unwrap();
    assert_eq!(backend.seen().offered_tools.len(), 1);

    let events = collect(&mut rx).await;
    let answered: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            LlmEvent::ToolResult { id, .. } => Some(id.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(answered, ["c-1", "c-2", "c-3"], "{events:?}");

    let pushed = backend.seen().tool_results;
    let [step] = pushed.as_slice() else {
        panic!("batch was not answered exactly once: {pushed:?}");
    };
    let contents: Vec<(&str, &str)> = step
        .iter()
        .map(|payload| (payload.call_id.as_str(), payload.content.as_str()))
        .collect();
    assert_eq!(
        contents,
        [
            (
                "c-1",
                "[error] run: agent turn cancelled during dispatch.\n[exit:-1 | 0ms]"
            ),
            (
                "c-2",
                "[error] run: agent turn cancelled before dispatch.\n[exit:-1 | 0ms]"
            ),
            (
                "c-3",
                "[error] run: agent turn cancelled before dispatch.\n[exit:-1 | 0ms]"
            ),
        ]
    );
}

#[tokio::test(start_paused = true)]
async fn hung_tool_past_deadline_becomes_error_result_and_turn_continues() {
    let mut tools = ToolRegistry::new();
    tools.register(HangingTool {
        entered: Arc::new(Notify::new()),
    });
    let backend = scripted(vec![
        Step::tool_calls(vec![call("c-1", "hang")]),
        Step::reply("ok"),
    ]);
    let agent = agent_with(
        backend.clone(),
        Arc::new(tools),
        None,
        Duration::from_secs(2),
    );
    let (res, _) = tokio::time::timeout(Duration::from_secs(5), run_turn(agent, "go"))
        .await
        .expect("tool deadline did not fire");
    res.unwrap();

    assert_eq!(
        backend.seen().offered_tools.len(),
        2,
        "turn did not continue after the abandoned call"
    );
    let pushed = backend.seen().tool_results;
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

#[tokio::test]
async fn panicking_tool_becomes_error_result_and_turn_continues() {
    let mut tools = ToolRegistry::new();
    tools.register(PanickingTool);
    let backend = scripted(vec![
        Step::tool_calls(vec![call("c-1", "web fetch https://example.com")]),
        Step::reply("ok"),
    ]);
    let agent = agent_with(
        backend.clone(),
        Arc::new(tools),
        None,
        Duration::from_secs(2),
    );

    run_turn(agent, "go").await.0.unwrap();

    assert_eq!(backend.seen().offered_tools.len(), 2);
    let pushed = backend.seen().tool_results;
    assert!(
        pushed[0][0]
            .content
            .starts_with("[error] run: the tool crashed (not implemented)."),
        "{:?}",
        pushed[0][0].content
    );
}

#[tokio::test(start_paused = true)]
async fn self_bounded_tool_outlives_the_tool_deadline() {
    let mut tools = ToolRegistry::new();
    tools.register(SelfBoundedTool {
        delay: Duration::from_secs(60),
    });
    let backend = scripted(vec![
        Step::tool_calls(vec![call("c-1", "relay")]),
        Step::reply("ok"),
    ]);
    let agent = agent_with(
        backend.clone(),
        Arc::new(tools),
        None,
        Duration::from_secs(2),
    );

    run_turn(agent, "go").await.0.unwrap();

    let pushed = backend.seen().tool_results;
    assert_eq!(pushed[0][0].content, "finished");
}

#[derive(Debug)]
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
    async fn snapshot(&self) -> Option<HealthSnapshot> {
        None
    }

    async fn wait_for_ready(&self, _budget: Duration) -> Result<(), HealthWaitError> {
        self.wait_calls.fetch_add(1, Ordering::SeqCst);
        self.wait_result
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
    let backend = scripted([Step::error(LlmError::ServerRestarting("boom".into()))]);
    let probe = MockProbe::new(Ok(()));
    let (res, events) = run_turn(
        agent_with(
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
    assert_eq!(backend.seen().offered_tools.len(), 2);
}

#[tokio::test]
async fn replay_does_not_loop_on_repeated_server_restarting() {
    let backend = scripted([
        Step::error(LlmError::ServerRestarting("first".into())),
        Step::error(LlmError::ServerRestarting("second".into())),
    ]);
    let probe = MockProbe::new(Ok(()));
    let (res, events) = run_turn(
        agent_with(
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
    assert_failed_without_done(&events);
    assert_eq!(backend.seen().offered_tools.len(), 2);
}
