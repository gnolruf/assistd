use super::*;

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
        .agent_turn_lock
        .clone()
        .lock_owned()
        .await;
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
    let backend = ToolCallBackend::new(
        "Cut off mid-",
        "All done.",
        vec![StepOutcome::Truncated, StepOutcome::Final],
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
    let backend = ToolCallBackend::new(
        "Checking both.",
        "All done.",
        vec![StepOutcome::ToolCalls(vec![
            run_call("call-a", "echo a"),
            run_call("call-b", "echo b"),
        ])],
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

fn tool_calls(row: &HistoryRow) -> &[PersistedToolCall] {
    match &row.message.kind {
        MessageKind::Assistant { tool_calls, .. } => tool_calls,
        MessageKind::User | MessageKind::Tool { .. } => &[],
    }
}
