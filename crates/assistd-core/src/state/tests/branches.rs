use serde_json::{Value, json};

use assistd_ipc::{RelayStepEnd, RelayStepPlan};

use super::*;
use crate::relay::tests::relay_tool;

fn history(events: &[Event]) -> Vec<(Role, &str)> {
    events
        .iter()
        .filter_map(|e| match &e.body {
            EventBody::HistoryEntry { role, content, .. } => Some((*role, content.as_str())),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn fork_creates_branch_and_switches() {
    let (state, conv, session, main_branch) = fresh_branch_state().await;
    let (res, events) = dispatch(
        &state,
        Request::new(
            "rq",
            RequestBody::Fork {
                name: "experiment".into(),
            },
        ),
    )
    .await;
    res.unwrap();
    assert_eq!(events.last(), Some(&done("rq")));

    let (active_session, active_branch) = state.session_ctx.current().await;
    assert_eq!(active_session, session);
    let active_branch = active_branch.expect("fork lands on a saved branch");
    assert_ne!(active_branch, main_branch);
    assert_eq!(
        conv.current_branch(session).await.unwrap(),
        Some(active_branch)
    );
    assert!(
        events.iter().any(|e| matches!(
            &e.body,
            EventBody::BranchSwitched { branch_id, name, parent_branch_name, .. }
                if *branch_id == Some(active_branch.0)
                    && name == "experiment"
                    && parent_branch_name.as_deref() == Some("main")
        )),
        "{events:?}"
    );
}

#[tokio::test]
async fn branches_lists_active_session_first() {
    let (state, conv, _session, main_branch) = fresh_branch_state().await;
    conv.fork_and_switch(main_branch, "alt").await.unwrap();
    conv.begin_session_with_main_branch(SessionId::new())
        .await
        .unwrap();

    let (res, events) = dispatch(&state, Request::new("rq", RequestBody::Branches)).await;
    res.unwrap();
    let infos: Vec<_> = events
        .iter()
        .filter_map(|e| match &e.body {
            EventBody::BranchInfo {
                name,
                is_active_session,
                ..
            } => Some((name.as_str(), *is_active_session)),
            _ => None,
        })
        .collect();
    assert_eq!(infos, [("alt", true), ("main", true), ("main", false)]);
    assert_eq!(events.last(), Some(&done("rq")));
}

#[tokio::test]
async fn switch_replays_history_into_event_stream() {
    let (state, conv, _, main_branch) = fresh_branch_state().await;
    append_turn(&conv, main_branch, "hello", "world").await;
    conv.fork_and_switch(main_branch, "alt").await.unwrap();

    let (res, events) = dispatch(
        &state,
        Request::new(
            "rq",
            RequestBody::Switch {
                target: "alt".into(),
            },
        ),
    )
    .await;
    res.unwrap();
    assert!(
        matches!(events.first(), Some(Event { body: EventBody::BranchSwitched { name, .. }, .. }) if name == "alt"),
        "{events:?}"
    );
    assert_eq!(
        history(&events),
        [(Role::User, "hello"), (Role::Assistant, "world")]
    );
    assert_eq!(events.last(), Some(&done("rq")));
}

#[tokio::test]
async fn resume_or_new_with_huge_window_resumes_instead_of_panicking() {
    let (state, conv, _, main_branch) = fresh_branch_state().await;
    append_turn(&conv, main_branch, "hello", "world").await;

    let (res, events) = dispatch(
        &state,
        Request::new(
            "rq",
            RequestBody::ResumeOrNew {
                recency_secs: i64::MAX as u64,
            },
        ),
    )
    .await;
    res.unwrap();
    assert_eq!(
        history(&events),
        [(Role::User, "hello"), (Role::Assistant, "world")],
        "an unbounded window must keep the branch"
    );
    assert_eq!(events.last(), Some(&done("rq")));
}

#[tokio::test]
async fn undo_drops_last_turn_and_emits_count() {
    let (state, conv, _, main_branch) = fresh_branch_state().await;
    append_turn(&conv, main_branch, "first", "a").await;
    append_turn(&conv, main_branch, "second", "b").await;

    let (res, events) = dispatch(&state, Request::new("rq", RequestBody::Undo)).await;
    res.unwrap();
    let applied = events
        .iter()
        .find_map(|e| match &e.body {
            EventBody::UndoApplied {
                removed_messages,
                last_user_text,
            } => Some((*removed_messages, last_user_text.as_deref())),
            _ => None,
        })
        .expect("expected UndoApplied");
    assert_eq!(applied, (2, Some("second")));

    let remaining: Vec<_> = conv
        .load_branch(main_branch, HistoryScope::Whole)
        .await
        .unwrap()
        .rows
        .into_iter()
        .map(|r| r.message.content)
        .collect();
    assert_eq!(remaining, ["first", "a"]);
}

#[tokio::test]
async fn branch_operations_do_not_wait_for_the_model_lane() {
    let (state, _conv, _session, _branch) = fresh_branch_state().await;
    let task_run_lane = state
        .turns
        .deps
        .admission
        .task_queue
        .queue_refine()
        .granted()
        .await
        .expect("the main lane was free");
    for request in [
        Request::new("undo", RequestBody::Undo),
        Request::new("new", RequestBody::NewSession),
    ] {
        let (res, _) = tokio::time::timeout(Duration::from_secs(5), dispatch(&state, request))
            .await
            .expect("a branch operation waited behind the model lane");
        res.unwrap();
    }
    drop(task_run_lane);
}

#[tokio::test]
async fn new_session_writes_no_rows_until_the_first_message() {
    let (state, conv, session, _branch) = fresh_branch_state().await;
    let (res, events) = dispatch(&state, Request::new("rq", RequestBody::NewSession)).await;
    res.unwrap();

    let (unsaved_session, unsaved_branch) = state.session_ctx.current().await;
    assert_ne!(unsaved_session, session);
    assert_eq!(unsaved_branch, None);
    assert_eq!(
        events,
        [
            Event::new(
                "rq",
                EventBody::BranchSwitched {
                    branch_id: None,
                    session_id: unsaved_session.to_string(),
                    session_title: None,
                    name: "main".into(),
                    parent_branch_name: None,
                    fork_point_seq: None
                }
            ),
            done("rq"),
        ]
    );
    assert_eq!(conv.list_branches().await.unwrap().len(), 1);

    let (res, _) = dispatch(&state, query("q", "hello")).await;
    res.unwrap();
    state.transcript.drain_inflight().await;

    let (saved_session, saved_branch) = state.session_ctx.current().await;
    assert_eq!(saved_session, unsaved_session);
    let saved_branch = saved_branch.expect("the first message saves the session");
    assert_eq!(
        conv.current_branch(saved_session).await.unwrap(),
        Some(saved_branch)
    );
    assert_eq!(conv.list_branches().await.unwrap().len(), 2);
    let rows = conv
        .load_branch(saved_branch, HistoryScope::Whole)
        .await
        .unwrap()
        .rows;
    assert_eq!(
        rows.first().map(|row| row.message.content.as_str()),
        Some("hello")
    );
}

#[tokio::test]
async fn resume_or_new_keeps_an_unsaved_session() {
    let (state, conv, _session, _branch) = fresh_branch_state().await;
    let (res, _) = dispatch(&state, Request::new("new", RequestBody::NewSession)).await;
    res.unwrap();
    let (unsaved_session, _) = state.session_ctx.current().await;

    let (res, events) = dispatch(
        &state,
        Request::new("rq", RequestBody::ResumeOrNew { recency_secs: 0 }),
    )
    .await;
    res.unwrap();

    let (session, branch) = state.session_ctx.current().await;
    assert_eq!(session, unsaved_session);
    assert_eq!(branch, None);
    assert!(
        matches!(
            events.first(),
            Some(Event { body: EventBody::BranchSwitched { branch_id: None, session_id, .. }, .. })
                if *session_id == unsaved_session.to_string()
        ),
        "{events:?}"
    );
    assert_eq!(events.last(), Some(&done("rq")));
    assert_eq!(conv.list_branches().await.unwrap().len(), 1);
}

#[tokio::test]
async fn narration_from_a_truncated_step_is_not_persisted() {
    let backend = scripted([Step::truncated().saying("Cut off mid-")], "All done.");
    let (state, conv, _session, branch) = branch_state_with(backend, echo_tools()).await;
    let (res, _) = dispatch(&state, query("rq", "go")).await;
    res.unwrap();
    state.transcript.drain_inflight().await;

    let rows = conv
        .load_branch(branch, HistoryScope::Whole)
        .await
        .unwrap()
        .rows;
    let shape: Vec<_> = rows
        .iter()
        .map(|r| (r.message.role(), r.message.content.as_str()))
        .collect();
    assert_eq!(
        shape,
        [
            (PersistedRole::User, "go"),
            (PersistedRole::Assistant, "All done.")
        ]
    );
}

#[tokio::test]
async fn step_with_parallel_calls_persists_as_one_assistant_row() {
    let backend = scripted(
        [Step::tool_calls(vec![
            run_call("call-a", "echo a"),
            run_call("call-b", "echo b"),
        ])
        .saying("Checking both.")],
        "All done.",
    );
    let (state, conv, _session, branch) = branch_state_with(backend, echo_tools()).await;
    let (res, _) = dispatch(&state, query("rq", "go")).await;
    res.unwrap();
    state.transcript.drain_inflight().await;

    let rows = conv
        .load_branch(branch, HistoryScope::Whole)
        .await
        .unwrap()
        .rows;
    let shape_ignoring_tool_output: Vec<_> = rows
        .iter()
        .map(|r| {
            let role = r.message.role();
            let content = (role != PersistedRole::Tool).then_some(r.message.content.as_str());
            (role, content, tool_call_id(r))
        })
        .collect();
    assert_eq!(
        shape_ignoring_tool_output,
        [
            (PersistedRole::User, Some("go"), None),
            (PersistedRole::Assistant, Some("Checking both."), None),
            (PersistedRole::Tool, None, Some("call-a")),
            (PersistedRole::Tool, None, Some("call-b")),
            (PersistedRole::Assistant, Some("All done."), None),
        ]
    );

    let ids: Vec<_> = tool_calls(&rows[1])
        .iter()
        .map(|call| call.id.as_str())
        .collect();
    assert_eq!(ids, ["call-a", "call-b"]);
    assert!(tool_calls(&rows[4]).is_empty());
}

#[tokio::test]
async fn a_tool_result_is_saved_with_whether_its_call_read_this_machine() {
    let not_offered = ToolCall {
        id: "call-b".into(),
        name: "gone".into(),
        arguments: serde_json::json!({}),
    };
    let backend = scripted(
        [Step::tool_calls(vec![
            run_call("call-a", "echo a"),
            not_offered,
        ])],
        "All done.",
    );
    let (state, conv, _session, branch) = branch_state_with(backend, echo_tools()).await;
    let (res, _) = dispatch(&state, query("rq", "go")).await;
    res.unwrap();
    state.transcript.drain_inflight().await;

    let rows = conv
        .load_branch(branch, HistoryScope::Whole)
        .await
        .unwrap()
        .rows;
    let read_host: Vec<_> = rows
        .iter()
        .filter_map(|row| match &row.message.kind {
            MessageKind::Tool {
                call_id, read_host, ..
            } => Some((call_id.as_str(), *read_host)),
            MessageKind::User | MessageKind::Assistant { .. } => None,
        })
        .collect();
    assert_eq!(read_host, [("call-a", true), ("call-b", false)]);
}

fn tool_calls(row: &HistoryRow) -> &[PersistedToolCall] {
    match &row.message.kind {
        MessageKind::Assistant { tool_calls, .. } => tool_calls,
        MessageKind::User | MessageKind::Tool { .. } => &[],
    }
}

/// One thing a client is shown of a relay's sub-agents, streamed or replayed.
#[derive(Debug, PartialEq)]
enum RelayShown {
    Planned(Vec<RelayStepPlan>),
    Started(u32),
    Said(String),
    Ran {
        args: Option<Value>,
        output: String,
        exit_code: i64,
    },
    Ended(u32, RelayStepEnd),
}

/// What `events` show of a relay's sub-agents, with each reply's deltas joined.
fn relay_shown(events: &[Event]) -> Vec<RelayShown> {
    let mut shown = Vec::new();
    let mut call_args = None;
    for event in events {
        match (event.agent, &event.body) {
            (None, EventBody::RelayPlanned { steps }) => {
                shown.push(RelayShown::Planned(steps.clone()));
            }
            (None, EventBody::RelayStepStarted { step, .. }) => {
                shown.push(RelayShown::Started(*step));
            }
            (None, EventBody::RelayStepEnded { step, end, .. }) => {
                shown.push(RelayShown::Ended(*step, end.clone()));
            }
            (Some(_), EventBody::Delta { text }) => match shown.last_mut() {
                Some(RelayShown::Said(said)) => said.push_str(text),
                _ => shown.push(RelayShown::Said(text.clone())),
            },
            (Some(_), EventBody::ToolCall { args, .. }) => call_args = Some(args.clone()),
            (Some(_), EventBody::ToolResult { result, .. }) => shown.push(RelayShown::Ran {
                args: call_args.take(),
                output: result.output.clone(),
                exit_code: result.exit_code.into(),
            }),
            (Some(_), EventBody::HistoryEntry { role, content, .. })
                if *role == Role::Assistant && !content.is_empty() =>
            {
                shown.push(RelayShown::Said(content.clone()));
            }
            (
                Some(_),
                EventBody::HistoryEntry {
                    role: Role::Tool,
                    content,
                    tool_args,
                    exit_code,
                    ..
                },
            ) => shown.push(RelayShown::Ran {
                args: tool_args.clone(),
                output: content.clone(),
                exit_code: exit_code.unwrap_or_default(),
            }),
            _ => {}
        }
    }
    shown
}

#[tokio::test]
async fn a_relay_turn_replays_its_sub_agent_as_streamed_and_undo_leaves_nothing_of_it() {
    let dir = tempfile::tempdir().unwrap();
    let notes = dir.path().join("notes.txt");
    std::fs::write(&notes, "buy milk").unwrap();
    let read_the_notes = run_call("c-1", "cat ../input/notes.txt | write copy.txt");
    let sub_agent =
        ScriptedBackend::new([Step::tool_calls([read_the_notes]).saying("Reading the notes.")])
            .then_replying("The notes say to buy milk.");
    let relay_call = ToolCall {
        id: "c-relay".into(),
        name: "relay".into(),
        arguments: json!({
            "task": "say what the notes ask for",
            "inputs": [notes],
            "agent": "file_review",
        }),
    };
    let backend = Arc::new(
        ScriptedBackend::new([Step::tool_calls([relay_call])])
            .then_replying("Buy milk.")
            .handing_out([Arc::new(sub_agent) as Arc<dyn LlmBackend>]),
    );
    let mut tools = ToolRegistry::default();
    tools.register(relay_tool(&backend, dir.path()));
    let (state, conv, _session, branch) = branch_state_with(backend, Arc::new(tools)).await;

    let (res, streamed) = dispatch(&state, query("rq", "what do the notes ask for?")).await;
    res.unwrap();
    state.transcript.drain_inflight().await;
    let switch = RequestBody::Switch {
        target: "main".into(),
    };
    let (res, replayed) = dispatch(&state, Request::new("rs", switch)).await;
    res.unwrap();

    let shown = relay_shown(&streamed);
    let [
        RelayShown::Planned(plan),
        RelayShown::Started(1),
        RelayShown::Said(narration),
        RelayShown::Ran { .. },
        RelayShown::Said(answer),
        RelayShown::Ended(1, RelayStepEnd::Answered { answer_file }),
    ] = shown.as_slice()
    else {
        panic!("{shown:?}");
    };
    assert_eq!(plan.len(), 1);
    assert_eq!(
        (narration.as_str(), answer.as_str()),
        ("Reading the notes.", "The notes say to buy milk.")
    );
    assert_eq!(relay_shown(&replayed), shown);
    assert!(answer_file.exists());

    let (res, _) = dispatch(&state, Request::new("ru", RequestBody::Undo)).await;
    res.unwrap();
    assert!(conv.load_agent_runs(branch).await.unwrap().is_empty());
    assert!(!answer_file.exists());
}
