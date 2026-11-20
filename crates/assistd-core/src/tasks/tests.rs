use std::borrow::Cow;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use parking_lot::Mutex as StdMutex;
use tokio::sync::{Notify, mpsc, watch};

use assistd_config::Config;
use assistd_ipc::{
    Event, EventBody, EventKind, HostName, PermissionMode, PresenceTarget, Request, RequestBody,
    SubscribeFilter, TaskQueueOp, TaskRequest, TaskRunState,
};
use assistd_llm::{EchoBackend, ScriptedBackend, Step, ToolCall, TurnBackend};
use assistd_memory::{HistoryScope, SessionId};
use assistd_tools::{
    Approval, ConfirmationGate, ConfirmationRequest, IpcConfirmationGate, JudgeContext,
    JudgedClass, Risk, Summary, ToolError, ToolOutput, ToolRegistry, TurnContext, Unattended,
};
use assistd_utils::readiness::Readiness;
use assistd_voice::VoiceManager;

use crate::presence::tests::stub_presence;
use crate::state::tests::{
    HangingTool, MockSpeechRecorder, await_task_queue, dispatch, done, listed_run_states,
    main_backend, paused_prompt, queue_task_run, queued_runs, runs_end_unasked, scripted,
    settle_task_runs, task_state, task_state_with,
};
use crate::state::{AppState, MemoryStack, Requester, Subsystems};
use crate::transcript::SessionContext;

#[tokio::test]
async fn a_finished_task_run_leaves_the_main_backend_untouched() {
    let backend = main_backend(Arc::new(EchoBackend::new()));
    let (state, conv, branch, task) =
        task_state(backend.clone(), Arc::new(ToolRegistry::default())).await;
    let before = conv.load_branch(branch, HistoryScope::Whole).await.unwrap();

    queue_task_run(&state, task).await;
    runs_end_unasked(&state, None).await;
    settle_task_runs(&state).await;

    let runs = state.memory.tasks.runs(task, 1).await.unwrap();
    assert_eq!(runs[0].reply.as_deref(), Some("summarize my inbox"));
    assert!(backend.seen().users.is_empty());
    assert_eq!(
        conv.load_branch(branch, HistoryScope::Whole).await.unwrap(),
        before
    );
    assert_eq!(state.session_ctx.current().await.1, Some(branch));
    assert_eq!(listed_run_states(&conv).await, [TaskRunState::Succeeded]);
}

#[tokio::test]
async fn an_announcing_task_broadcasts_and_speaks_its_final_reply() {
    let backend = main_backend(Arc::new(EchoBackend::new()));
    let (state, _conv, _branch, task) =
        task_state(backend, Arc::new(ToolRegistry::default())).await;
    let recorder = MockSpeechRecorder::new();
    state
        .subsystems
        .voice
        .set_speech(Readiness::Ready(recorder.clone()));
    let mut spec = state.memory.tasks.get(task).await.unwrap().spec;
    spec.announce = true;
    state.memory.tasks.save(Some(task), spec).await.unwrap();
    let mut bus = state.bus.subscribe(SubscribeFilter {
        kinds: vec![EventKind::TaskReply],
    });

    queue_task_run(&state, task).await;
    let announced = tokio::time::timeout(Duration::from_secs(5), bus.recv())
        .await
        .expect("no TaskReply")
        .unwrap();
    tokio::time::timeout(Duration::from_secs(5), async {
        while recorder.calls().is_empty() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the reply was not spoken");
    settle_task_runs(&state).await;

    assert!(matches!(
        announced.body,
        EventBody::TaskReply { reply, .. } if reply == "summarize my inbox"
    ));
    assert_eq!(recorder.calls(), ["summarize my inbox"]);
}

#[tokio::test]
async fn a_cancelled_task_run_leaves_the_main_backend_untouched() {
    let entered = Arc::new(Notify::new());
    let dropped = Arc::new(AtomicBool::new(false));
    let task_backend = scripted(
        [Step::tool_calls(vec![ToolCall {
            id: "c1".into(),
            name: "hang".into(),
            arguments: serde_json::json!({}),
        }])],
        "done.",
    );
    let mut tools = ToolRegistry::new();
    tools.register(HangingTool {
        entered: entered.clone(),
        dropped: dropped.clone(),
    });
    let backend = main_backend(task_backend);
    let (state, conv, branch, task) = task_state(backend.clone(), Arc::new(tools)).await;
    let before = conv.load_branch(branch, HistoryScope::Whole).await.unwrap();

    let run = queue_task_run(&state, task).await;
    entered.notified().await;
    let (res, events) = dispatch(
        &state,
        Request::new("c", RequestBody::Task(TaskRequest::Cancel { run })),
    )
    .await;
    res.unwrap();
    assert_eq!(events, [done("c")]);
    settle_task_runs(&state).await;

    assert!(
        dropped.load(Ordering::SeqCst),
        "the hung tool outlived the run"
    );
    assert!(backend.seen().users.is_empty());
    assert_eq!(
        conv.load_branch(branch, HistoryScope::Whole).await.unwrap(),
        before
    );
    assert_eq!(listed_run_states(&conv).await, [TaskRunState::Cancelled]);
}

async fn change_task_queue(state: &Arc<AppState>, op: TaskQueueOp) {
    let label = format!("{op:?}");
    let (res, events) = dispatch(
        state,
        Request::new("q", RequestBody::Task(TaskRequest::Queue { op })),
    )
    .await;
    res.unwrap();
    assert_eq!(events, [done("q")], "{label}");
}

#[tokio::test]
async fn queue_order_and_pause_follow_moves_and_cancels_across_a_reload() {
    let backend = main_backend(Arc::new(EchoBackend::new()));
    let (state, _conv, _branch, task) =
        task_state(backend, Arc::new(ToolRegistry::default())).await;
    change_task_queue(&state, TaskQueueOp::Pause).await;
    let mut runs = Vec::new();
    for _ in 0..4 {
        runs.push(queue_task_run(&state, task).await);
    }
    let [first, second, third, fourth] = runs[..] else {
        unreachable!()
    };
    change_task_queue(&state, TaskQueueOp::Move { run: fourth, to: 0 }).await;
    change_task_queue(&state, TaskQueueOp::Move { run: first, to: 9 }).await;
    change_task_queue(&state, TaskQueueOp::Cancel { run: second }).await;
    let expected = [fourth, third, first].map(|run| (run, TaskRunState::Queued));
    assert_eq!(queued_runs(&state), expected);
    settle_task_runs(&state).await;

    let reloaded = Arc::new(AppState::from_parts(
        Config::default(),
        Subsystems::new(
            Arc::new(EchoBackend::new()),
            stub_presence(PresenceTarget::Active),
            Arc::new(ToolRegistry::default()),
            VoiceManager::new(true),
        ),
        MemoryStack::new(
            Arc::clone(&state.memory.memory),
            Arc::clone(&state.memory.transcripts),
            Arc::clone(&state.memory.tasks),
            state.config.embedding.clone(),
        ),
        Arc::new(SessionContext::new(SessionId::new(), None)),
        watch::channel(NonZeroU32::MIN).1,
    ));
    reloaded.tasks.start_task_queue().await;
    assert_eq!(queued_runs(&reloaded), expected);
    assert!(reloaded.tasks.deps.admission.queue_snapshot().paused);
    settle_task_runs(&reloaded).await;
}

/// Asks to fetch from the `host` in its arguments, as `web` asks about a host not yet
/// approved, and records how each call ended.
#[derive(Debug, Default)]
struct FetchTool {
    outcomes: Arc<StdMutex<Vec<&'static str>>>,
}

#[async_trait::async_trait]
impl assistd_tools::Tool for FetchTool {
    fn name(&self) -> &'static str {
        "fetch"
    }
    fn description(&self) -> Cow<'_, str> {
        "fetches a page".into()
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({"type": "object"})
    }
    async fn invoke(
        &self,
        turn: &TurnContext,
        args: serde_json::Value,
    ) -> Result<ToolOutput, ToolError> {
        let host = args["host"].as_str().unwrap_or_default().to_string();
        let request = ConfirmationRequest {
            tool: "web".into(),
            script: format!("web fetch https://{host}"),
            matched_pattern: format!("fetches from {host}, which is not yet approved"),
            summary: Summary::of("a fetch"),
            always_allow: vec![host.clone()],
            risk: Risk::Elevated,
            unattended: Unattended::WebHost(host),
            source: assistd_tools::ConfirmSource::Main,
        };
        let approved = IpcConfirmationGate.confirm(turn, request).await != Approval::Deny;
        let outcome = if approved { "fetched" } else { "denied" };
        self.outcomes.lock().push(outcome);
        Ok(ToolOutput::text(outcome, Duration::ZERO))
    }
}

/// Asks before running a script that only the safety judge can approve unasked, and
/// records how each call ended.
#[derive(Debug, Default)]
struct UnvettedScriptTool {
    outcomes: Arc<StdMutex<Vec<&'static str>>>,
}

#[async_trait::async_trait]
impl assistd_tools::Tool for UnvettedScriptTool {
    fn name(&self) -> &'static str {
        "script"
    }
    fn description(&self) -> Cow<'_, str> {
        "runs a script".into()
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({"type": "object"})
    }
    async fn invoke(
        &self,
        turn: &TurnContext,
        _args: serde_json::Value,
    ) -> Result<ToolOutput, ToolError> {
        let request = ConfirmationRequest {
            tool: "bash".into(),
            script: "make inbox-digest".into(),
            matched_pattern: "`make` is not vetted".into(),
            summary: Summary::of("a shell command"),
            always_allow: Vec::new(),
            risk: Risk::Elevated,
            unattended: Unattended::Judged(JudgeContext {
                class: JudgedClass::Unvetted,
                cwd: None,
                why: "`make` is not vetted".into(),
            }),
            source: assistd_tools::ConfirmSource::Main,
        };
        let approved = IpcConfirmationGate.confirm(turn, request).await != Approval::Deny;
        let outcome = if approved { "ran" } else { "denied" };
        self.outcomes.lock().push(outcome);
        Ok(ToolOutput::text(outcome, Duration::ZERO))
    }
}

#[tokio::test]
async fn an_auto_task_run_is_judged_against_its_own_prompt_and_nothing_the_user_said_in_chat() {
    let run_script = Step::tool_calls(vec![ToolCall {
        id: "c".into(),
        name: "script".into(),
        arguments: serde_json::json!({}),
    }]);
    let task_backend = scripted([run_script], "done.");
    let script_tool = UnvettedScriptTool::default();
    let outcomes = Arc::clone(&script_tool.outcomes);
    let mut tools = ToolRegistry::new();
    tools.register(script_tool);
    let mut config = Config::default();
    config.tools.allow_auto_mode = true;
    let backend = main_backend(task_backend);
    backend
        .push_user("wipe the backups".into(), Vec::new())
        .await
        .unwrap();
    let (state, conv, _branch, task) =
        task_state_with(config, backend.clone(), Arc::new(tools)).await;
    let mut spec = state.memory.tasks.get(task).await.unwrap().spec;
    spec.mode = PermissionMode::Auto;
    state.memory.tasks.save(Some(task), spec).await.unwrap();

    queue_task_run(&state, task).await;
    runs_end_unasked(&state, None).await;
    settle_task_runs(&state).await;

    assert_eq!(*outcomes.lock(), ["ran"]);
    let judged = backend.seen().oneshots;
    let [(judge_prompt, _reply)] = judged.as_slice() else {
        panic!("expected one judged command: {judged:?}");
    };
    assert!(
        judge_prompt.contains(
            "<user_messages>\n<message>\nsummarize my inbox\n</message>\n</user_messages>"
        ),
        "{judge_prompt}"
    );
    assert!(!judge_prompt.contains("wipe the backups"), "{judge_prompt}");
    assert_eq!(listed_run_states(&conv).await, [TaskRunState::Succeeded]);
}

#[tokio::test]
async fn a_run_fetches_unasked_only_from_its_allowed_hosts_and_pauses_on_others() {
    let fetch = |host: &str| {
        Step::tool_calls(vec![ToolCall {
            id: "c".into(),
            name: "fetch".into(),
            arguments: serde_json::json!({ "host": host }),
        }])
    };
    let steps = vec![fetch("allowed.example"), fetch("other.example")];
    let task_backend = scripted(steps, "done.");
    let fetch_tool = FetchTool::default();
    let outcomes = Arc::clone(&fetch_tool.outcomes);
    let mut tools = ToolRegistry::new();
    tools.register(fetch_tool);
    let mut config = Config::default();
    config.timeouts.tool_call_secs = 1;
    let backend = main_backend(task_backend);
    let (state, conv, _branch, task) = task_state_with(config, backend, Arc::new(tools)).await;
    let mut spec = state.memory.tasks.get(task).await.unwrap().spec;
    spec.allowed_hosts = vec!["allowed.example".to_string().try_into().unwrap()];
    state.memory.tasks.save(Some(task), spec).await.unwrap();

    let run = queue_task_run(&state, task).await;
    let prompt = await_task_queue(&state, paused_prompt).await;
    assert_eq!(prompt.script, "web fetch https://other.example");
    assert_eq!(
        *outcomes.lock(),
        ["fetched"],
        "the allowed host was fetched unasked"
    );
    assert_eq!(queued_runs(&state), [(run, TaskRunState::NeedsInput)]);
    assert!(
        state
            .tasks
            .deps
            .admission
            .agent_turn_lock
            .try_lock()
            .is_ok(),
        "a paused run holds no lane"
    );
    tokio::time::sleep(Duration::from_millis(1500)).await;

    let (res, events) = dispatch(
        &state,
        Request::new(
            "a",
            RequestBody::Task(TaskRequest::Confirm {
                run,
                confirm_id: prompt.confirm_id.clone(),
                allow: true,
                always: false,
            }),
        ),
    )
    .await;
    res.unwrap();
    assert_eq!(events, [done("a")]);
    runs_end_unasked(&state, Some(&prompt.confirm_id)).await;
    settle_task_runs(&state).await;

    assert_eq!(
        *outcomes.lock(),
        ["fetched", "fetched"],
        "the parked fetch outlived the tool deadline"
    );
    assert_eq!(listed_run_states(&conv).await, [TaskRunState::Succeeded]);
}

#[tokio::test]
async fn always_allowing_a_paused_fetch_adds_its_host_to_the_task_and_stops_asking() {
    let fetch = || {
        Step::tool_calls(vec![ToolCall {
            id: "c".into(),
            name: "fetch".into(),
            arguments: serde_json::json!({ "host": "new.example" }),
        }])
    };
    let task_backend = scripted([fetch(), fetch()], "done.");
    let fetch_tool = FetchTool::default();
    let outcomes = Arc::clone(&fetch_tool.outcomes);
    let mut tools = ToolRegistry::new();
    tools.register(fetch_tool);
    let backend = main_backend(task_backend);
    let (state, _conv, _branch, task) =
        task_state_with(Config::default(), backend, Arc::new(tools)).await;

    let run = queue_task_run(&state, task).await;
    let prompt = await_task_queue(&state, paused_prompt).await;
    assert_eq!(
        prompt.host.as_ref().map(HostName::as_str),
        Some("new.example")
    );

    let (res, events) = dispatch(
        &state,
        Request::new(
            "a",
            RequestBody::Task(TaskRequest::Confirm {
                run,
                confirm_id: prompt.confirm_id.clone(),
                allow: true,
                always: true,
            }),
        ),
    )
    .await;
    res.unwrap();
    let [
        Event {
            body: EventBody::Task { spec, .. },
            ..
        },
        Event {
            body: EventBody::Done,
            ..
        },
    ] = events.as_slice()
    else {
        panic!("unexpected events: {events:?}");
    };
    assert_eq!(spec.allowed_hosts, [prompt.host.clone().unwrap()]);
    runs_end_unasked(&state, Some(&prompt.confirm_id)).await;
    settle_task_runs(&state).await;
    assert_eq!(*outcomes.lock(), ["fetched", "fetched"]);
}

#[tokio::test]
async fn a_lagged_subscriber_is_resent_the_queue_with_the_prompt_of_a_paused_run() {
    let fetch = Step::tool_calls(vec![ToolCall {
        id: "c".into(),
        name: "fetch".into(),
        arguments: serde_json::json!({ "host": "new.example" }),
    }]);
    let task_backend = scripted([fetch], "done.");
    let mut tools = ToolRegistry::new();
    tools.register(FetchTool::default());
    let backend = main_backend(task_backend);
    let (state, _conv, _branch, task) =
        task_state_with(Config::default(), backend, Arc::new(tools)).await;
    let run = queue_task_run(&state, task).await;
    let paused_on = await_task_queue(&state, paused_prompt).await;

    let (tx, mut rx) = mpsc::channel(1);
    let daemon = Arc::clone(&state);
    let subscriber = tokio::spawn(async move {
        let filter = SubscribeFilter {
            kinds: vec![EventKind::TaskQueue],
        };
        let subscribe = Request::new("sub", RequestBody::Subscribe { filter });
        daemon.dispatch(subscribe, Requester::default(), tx).await
    });
    assert!(matches!(
        rx.recv().await,
        Some(Event {
            body: EventBody::TaskQueue { .. },
            ..
        })
    ));
    for _ in 0..1024 {
        state.bus.publish(&Event::new(
            "flood",
            EventBody::TaskQueue {
                snapshot: state.tasks.deps.admission.queue_snapshot(),
            },
        ));
    }

    let Some(Event {
        id,
        body: EventBody::TaskQueue { snapshot },
    }) = rx.recv().await
    else {
        panic!("no queue snapshot after the lag");
    };
    assert_eq!(id, "sub");
    let prompt = paused_prompt(snapshot).expect("the snapshot carries the paused prompt");
    assert_eq!(prompt, paused_on);

    drop(rx);
    subscriber.await.unwrap().unwrap();
    let (res, _) = dispatch(
        &state,
        Request::new(
            "a",
            RequestBody::Task(TaskRequest::Confirm {
                run,
                confirm_id: prompt.confirm_id,
                allow: false,
                always: false,
            }),
        ),
    )
    .await;
    res.unwrap();
    settle_task_runs(&state).await;
}

#[tokio::test]
async fn a_refine_sends_the_rewrite_and_none_of_its_planning_turn() {
    let draft = serde_json::json!({
        "prompt": "Summarize the unread mail in the inbox.",
        "summary": "Reads the inbox.",
        "steps": [],
    });
    let planning = ScriptedBackend::default()
        .then_replying("Found the inbox.")
        .answering_schemas(draft.to_string());
    let (state, _conv, _branch, _task) = task_state(
        main_backend(Arc::new(planning)),
        Arc::new(ToolRegistry::default()),
    )
    .await;
    let mut bus = state.bus.subscribe(SubscribeFilter {
        kinds: vec![EventKind::Delta, EventKind::LastDelta, EventKind::Done],
    });
    let refine = TaskRequest::Refine {
        prompt: "summarize my inbox".to_string().try_into().unwrap(),
        mode: PermissionMode::Manual,
        allowed_hosts: Vec::new(),
        feedback: None,
    };

    let (res, events) = dispatch(&state, Request::new("r", RequestBody::Task(refine))).await;
    res.unwrap();

    let [refined, end] = events.as_slice() else {
        panic!("unexpected events: {events:?}");
    };
    assert_eq!(end, &done("r"));
    assert!(
        matches!(&refined.body, EventBody::TaskRefined { prompt, .. }
            if prompt.as_str() == "Summarize the unread mail in the inbox."),
        "{refined:?}"
    );
    assert_eq!(
        bus.recv().await.unwrap(),
        done("r"),
        "the turn was published"
    );
}
