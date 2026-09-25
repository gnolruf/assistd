use super::*;
use tokio::sync::watch;

/// The returned guard keeps the temp dir and the writer's shutdown sender alive.
async fn fresh_store() -> (
    SqliteConversationStore,
    (tempfile::TempDir, watch::Sender<bool>),
) {
    let temp = tempfile::tempdir().unwrap();
    let (tx, rx) = watch::channel(false);
    let (handle, _writer) = SqliteHandle::open(&temp.path().join("memory.db"), rx)
        .await
        .unwrap();
    (SqliteConversationStore::new(Arc::new(handle)), (temp, tx))
}

fn contents(history: &[HistoryRow]) -> Vec<(PersistedRole, &str)> {
    history
        .iter()
        .map(|r| (r.role, r.content.as_str()))
        .collect()
}

#[tokio::test]
async fn round_trip_user_and_assistant_messages() {
    let (store, _guard) = fresh_store().await;
    let (session, branch) = store.begin_session_with_main_branch(42).await.unwrap();
    let turn = store.begin_turn(&session, "what is 2+2?").await.unwrap();

    store
        .append_message_to_branch(
            &session,
            branch,
            Some(turn),
            PersistedMessage::user("what is 2+2?"),
        )
        .await
        .unwrap();
    store
        .append_message_to_branch(
            &session,
            branch,
            Some(turn),
            PersistedMessage::assistant_text("four"),
        )
        .await
        .unwrap();
    store.end_turn(turn).await.unwrap();
    store.end_session(&session).await.unwrap();

    let history = store.load_branch_history(branch).await.unwrap();
    assert_eq!(
        contents(&history),
        [
            (PersistedRole::User, "what is 2+2?"),
            (PersistedRole::Assistant, "four")
        ]
    );

    let recent = store.recent_turns(5).await.unwrap();
    let [summary] = recent.as_slice() else {
        panic!("expected one turn, got {recent:?}");
    };
    assert_eq!(summary.turn_id, turn.0);
    assert_eq!(summary.session_id, session.0);
    assert_eq!(summary.user_text, "what is 2+2?");
    assert_eq!(summary.message_count, 2);
    assert!(summary.ended_at.is_some());
}

#[tokio::test]
async fn tool_calls_and_tool_results_round_trip() {
    let (store, _guard) = fresh_store().await;
    let (session, branch) = store.begin_session_with_main_branch(1).await.unwrap();
    let turn = store.begin_turn(&session, "list files").await.unwrap();

    let calls = serde_json::json!([{"id": "c-1", "name": "run", "arguments": {"command": "ls"}}]);
    let call_id = store
        .append_message_to_branch(
            &session,
            branch,
            Some(turn),
            PersistedMessage::assistant_tool_calls("", calls.clone()),
        )
        .await
        .unwrap();
    let result_id = store
        .append_message_to_branch(
            &session,
            branch,
            Some(turn),
            PersistedMessage::tool_result("file1\nfile2", "c-1", "run"),
        )
        .await
        .unwrap();

    let history = store.load_branch_history(branch).await.unwrap();
    assert_eq!(
        history,
        [
            HistoryRow {
                conversation_id: call_id,
                seq: 0,
                role: PersistedRole::Assistant,
                content: String::new(),
                tool_calls: Some(calls),
                tool_call_id: None,
                tool_name: None,
            },
            HistoryRow {
                conversation_id: result_id,
                seq: 1,
                role: PersistedRole::Tool,
                content: "file1\nfile2".into(),
                tool_calls: None,
                tool_call_id: Some("c-1".into()),
                tool_name: Some("run".into()),
            },
        ]
    );
}

#[tokio::test]
async fn begin_session_with_main_branch_inserts_session_and_main_branch() {
    let (store, _guard) = fresh_store().await;
    let (session, branch) = store.begin_session_with_main_branch(123).await.unwrap();
    assert_eq!(
        store.get_current_branch(&session).await.unwrap(),
        Some(branch)
    );
    let branches = store.list_branches().await.unwrap();
    let [main] = branches.as_slice() else {
        panic!("expected one branch, got {branches:?}");
    };
    assert_eq!(main.branch_id, branch);
    assert_eq!(main.session_id, session.0);
    assert_eq!(main.name, "main");
    assert!(main.is_current_in_session);
    assert_eq!(main.parent_branch_id, None);
    assert_eq!(main.message_count, 0);
}

#[tokio::test]
async fn fork_creates_independent_branch_sharing_history() {
    let (store, _guard) = fresh_store().await;
    let (session, main) = store.begin_session_with_main_branch(1).await.unwrap();
    let turn = store.begin_turn(&session, "hello").await.unwrap();
    store
        .append_message_to_branch(&session, main, Some(turn), PersistedMessage::user("hello"))
        .await
        .unwrap();
    store
        .append_message_to_branch(
            &session,
            main,
            Some(turn),
            PersistedMessage::assistant_text("hi"),
        )
        .await
        .unwrap();
    store.end_turn(turn).await.unwrap();

    let fork = store.fork_branch(main, "experiment").await.unwrap();
    let main_history = store.load_branch_history(main).await.unwrap();
    let fork_history = store.load_branch_history(fork).await.unwrap();
    assert_eq!(main_history.len(), 2);
    assert_eq!(
        fork_history, main_history,
        "fork references the same conversation rows"
    );

    let branches = store.list_branches().await.unwrap();
    let fork_info = branches.iter().find(|b| b.name == "experiment").unwrap();
    assert_eq!(fork_info.parent_branch_id, Some(main));
    assert_eq!(fork_info.parent_branch_name.as_deref(), Some("main"));
    assert_eq!(fork_info.fork_point_seq, Some(1));
    assert_eq!(fork_info.message_count, 2);
}

#[tokio::test]
async fn append_to_one_branch_does_not_show_on_the_other() {
    let (store, _guard) = fresh_store().await;
    let (session, main) = store.begin_session_with_main_branch(1).await.unwrap();
    let turn = store.begin_turn(&session, "q").await.unwrap();
    store
        .append_message_to_branch(&session, main, Some(turn), PersistedMessage::user("q"))
        .await
        .unwrap();
    let fork = store.fork_branch(main, "alt").await.unwrap();
    let alt_turn = store.begin_turn(&session, "alt q").await.unwrap();
    store
        .append_message_to_branch(
            &session,
            fork,
            Some(alt_turn),
            PersistedMessage::user("alt q"),
        )
        .await
        .unwrap();
    let main_history = store.load_branch_history(main).await.unwrap();
    let fork_history = store.load_branch_history(fork).await.unwrap();
    assert_eq!(contents(&main_history), [(PersistedRole::User, "q")]);
    assert_eq!(
        contents(&fork_history),
        [(PersistedRole::User, "q"), (PersistedRole::User, "alt q")]
    );
}

#[tokio::test]
async fn undo_removes_only_the_last_turn() {
    let (store, _guard) = fresh_store().await;
    let (session, main) = store.begin_session_with_main_branch(1).await.unwrap();
    let t1 = store.begin_turn(&session, "first").await.unwrap();
    store
        .append_message_to_branch(&session, main, Some(t1), PersistedMessage::user("first"))
        .await
        .unwrap();
    store
        .append_message_to_branch(
            &session,
            main,
            Some(t1),
            PersistedMessage::assistant_text("a"),
        )
        .await
        .unwrap();
    store.end_turn(t1).await.unwrap();

    let t2 = store.begin_turn(&session, "second").await.unwrap();
    store
        .append_message_to_branch(&session, main, Some(t2), PersistedMessage::user("second"))
        .await
        .unwrap();
    store
        .append_message_to_branch(
            &session,
            main,
            Some(t2),
            PersistedMessage::assistant_text("b"),
        )
        .await
        .unwrap();
    store.end_turn(t2).await.unwrap();

    let outcome = store.undo_last_turn(main).await.unwrap();
    assert_eq!(
        outcome,
        UndoOutcome {
            removed_messages: 2,
            last_user_text: Some("second".into()),
            removed_turn_id: Some(t2.0),
        }
    );

    let history = store.load_branch_history(main).await.unwrap();
    assert_eq!(
        contents(&history),
        [
            (PersistedRole::User, "first"),
            (PersistedRole::Assistant, "a")
        ]
    );
}

#[tokio::test]
async fn resolve_branch_prefers_given_session_or_qualified_prefix() {
    let (store, _guard) = fresh_store().await;
    let (s1, b1) = store.begin_session_with_main_branch(1).await.unwrap();
    let (s2, b2) = store.begin_session_with_main_branch(2).await.unwrap();
    let bare = store.resolve_branch("main", Some(&s1)).await.unwrap();
    assert_eq!(bare, Some((s1, b1)));
    let prefix = &s2.0[..8];
    let qualified = store
        .resolve_branch(&format!("{prefix}/main"), None)
        .await
        .unwrap();
    assert_eq!(qualified, Some((s2, b2)));
}

#[tokio::test]
async fn find_resumable_session_returns_unended() {
    let (store, _guard) = fresh_store().await;
    let (s, branch) = store.begin_session_with_main_branch(99).await.unwrap();
    let cand = store
        .find_resumable_session()
        .await
        .unwrap()
        .expect("unended session is resumable");
    assert_eq!(cand.session_id, s);
    assert_eq!(cand.current_branch_id, branch);
    assert_eq!(cand.daemon_pid, 99);

    store.end_session(&s).await.unwrap();
    assert_eq!(store.find_resumable_session().await.unwrap(), None);
}
