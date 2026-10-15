use super::*;
use tokio::sync::watch;

/// The returned guard keeps the temp dir and the writer's shutdown sender alive.
async fn fresh_store() -> (
    SqliteTranscriptStore,
    (tempfile::TempDir, watch::Sender<bool>),
) {
    let temp = tempfile::tempdir().unwrap();
    let (tx, rx) = watch::channel(false);
    let (handle, _writer) = SqliteHandle::open(&temp.path().join("memory.db"), rx)
        .await
        .unwrap();
    (SqliteTranscriptStore::new(handle), (temp, tx))
}

fn contents(history: &[HistoryRow]) -> Vec<(PersistedRole, &str)> {
    history
        .iter()
        .map(|r| (r.message.role(), r.message.content.as_str()))
        .collect()
}

#[tokio::test]
async fn tool_calls_and_tool_results_round_trip() {
    let (store, _guard) = fresh_store().await;
    let session = SessionId::new();
    let branch = store
        .begin_session_with_main_branch(&session, 1)
        .await
        .unwrap();
    let turn = store.begin_turn(&session).await.unwrap();

    let outcome = ToolOutcome {
        exit_code: 2,
        duration_ms: 41,
    };
    let calls = serde_json::json!([{"id": "c-1", "name": "run", "arguments": {"command": "ls"}}]);
    let call_id = store
        .append_message_to_branch(
            &session,
            branch,
            Some(turn),
            PersistedMessage::assistant_tool_calls("", calls.clone(), "look first".into()),
        )
        .await
        .unwrap();
    let result_id = store
        .append_message_to_branch(
            &session,
            branch,
            Some(turn),
            PersistedMessage::tool_result("file1\nfile2", "c-1", "run", outcome),
        )
        .await
        .unwrap();

    let history = store.load_branch_history(branch).await.unwrap();
    assert_eq!(
        history,
        [
            HistoryRow {
                message_id: call_id,
                seq: 0,
                message: PersistedMessage {
                    content: String::new(),
                    kind: MessageKind::Assistant {
                        reasoning: Some("look first".into()),
                        tool_calls: Some(calls),
                    },
                },
            },
            HistoryRow {
                message_id: result_id,
                seq: 1,
                message: PersistedMessage {
                    content: "file1\nfile2".into(),
                    kind: MessageKind::Tool {
                        call_id: "c-1".into(),
                        tool_name: "run".into(),
                        outcome,
                    },
                },
            },
        ]
    );
}

#[tokio::test]
async fn fork_creates_independent_branch_sharing_history() {
    let (store, _guard) = fresh_store().await;
    let session = SessionId::new();
    let main = store
        .begin_session_with_main_branch(&session, 1)
        .await
        .unwrap();
    let turn = store.begin_turn(&session).await.unwrap();
    store
        .append_message_to_branch(&session, main, Some(turn), PersistedMessage::user("hello"))
        .await
        .unwrap();
    store
        .append_message_to_branch(
            &session,
            main,
            Some(turn),
            PersistedMessage::assistant_text("hi", String::new()),
        )
        .await
        .unwrap();

    let fork = store.fork_branch(main, "experiment").await.unwrap();
    let main_history = store.load_branch_history(main).await.unwrap();
    let fork_history = store.load_branch_history(fork).await.unwrap();
    assert_eq!(main_history.len(), 2);
    assert_eq!(
        fork_history, main_history,
        "fork references the same message rows"
    );

    let branches = store.list_branches().await.unwrap();
    let fork_info = branches.iter().find(|b| b.name == "experiment").unwrap();
    assert_eq!(fork_info.parent_branch_id, Some(main));
    assert_eq!(fork_info.parent_branch_name.as_deref(), Some("main"));
    assert_eq!(fork_info.fork_point_seq, Some(1));
    assert_eq!(fork_info.message_count, 2);
    assert_eq!(
        fork_info.last_active_at, fork_info.created_at,
        "inherited messages predate the fork"
    );
}

#[tokio::test]
async fn branches_list_by_newest_message_not_session_start() {
    let (store, _guard) = fresh_store().await;
    let older = SessionId::new();
    let older_main = store
        .begin_session_with_main_branch(&older, 1)
        .await
        .unwrap();
    let newer_main = store
        .begin_session_with_main_branch(&SessionId::new(), 1)
        .await
        .unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(2)).await;
    store
        .append_message_to_branch(&older, older_main, None, PersistedMessage::user("back"))
        .await
        .unwrap();

    let branches = store.list_branches().await.unwrap();
    let order: Vec<_> = branches.iter().map(|b| b.branch_id).collect();
    assert_eq!(order, [older_main, newer_main]);
    assert!(branches[0].last_active_at > branches[1].created_at);
}

#[tokio::test]
async fn append_to_one_branch_does_not_show_on_the_other() {
    let (store, _guard) = fresh_store().await;
    let session = SessionId::new();
    let main = store
        .begin_session_with_main_branch(&session, 1)
        .await
        .unwrap();
    let turn = store.begin_turn(&session).await.unwrap();
    store
        .append_message_to_branch(&session, main, Some(turn), PersistedMessage::user("q"))
        .await
        .unwrap();
    let fork = store.fork_branch(main, "alt").await.unwrap();
    let alt_turn = store.begin_turn(&session).await.unwrap();
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
    let session = SessionId::new();
    let main = store
        .begin_session_with_main_branch(&session, 1)
        .await
        .unwrap();
    let t1 = store.begin_turn(&session).await.unwrap();
    store
        .append_message_to_branch(&session, main, Some(t1), PersistedMessage::user("first"))
        .await
        .unwrap();
    store
        .append_message_to_branch(
            &session,
            main,
            Some(t1),
            PersistedMessage::assistant_text("a", String::new()),
        )
        .await
        .unwrap();

    let t2 = store.begin_turn(&session).await.unwrap();
    store
        .append_message_to_branch(&session, main, Some(t2), PersistedMessage::user("second"))
        .await
        .unwrap();
    store
        .append_message_to_branch(
            &session,
            main,
            Some(t2),
            PersistedMessage::assistant_text("b", String::new()),
        )
        .await
        .unwrap();

    let outcome = store.undo_last_turn(main).await.unwrap();
    assert_eq!(
        outcome,
        UndoOutcome {
            removed_messages: 2,
            last_user_text: Some("second".into()),
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
async fn summary_follows_kept_turns_forks_and_undo() {
    let (store, _guard) = fresh_store().await;
    let session = SessionId::new();
    let branch = store
        .begin_session_with_main_branch(&session, 1)
        .await
        .unwrap();
    for question in ["one", "two", "three"] {
        let turn = store.begin_turn(&session).await.unwrap();
        for msg in [
            PersistedMessage::user(question),
            PersistedMessage::assistant_text("answer", String::new()),
        ] {
            store
                .append_message_to_branch(&session, branch, Some(turn), msg)
                .await
                .unwrap();
        }
    }
    let kept_two = NonZeroUsize::new(2).unwrap();
    store
        .record_branch_summary(branch, "turn one".into(), kept_two)
        .await
        .unwrap();
    let expected = Some(BranchSummary {
        through_seq: 1,
        content: "turn one".into(),
    });
    assert_eq!(store.load_branch_summary(branch).await.unwrap(), expected);

    let fork = store.fork_branch(branch, "alt").await.unwrap();
    assert_eq!(store.load_branch_summary(fork).await.unwrap(), expected);

    store.undo_last_turn(branch).await.unwrap();
    assert_eq!(store.load_branch_summary(branch).await.unwrap(), expected);
    store.undo_last_turn(branch).await.unwrap();
    assert_eq!(store.load_branch_summary(branch).await.unwrap(), expected);
    store.undo_last_turn(branch).await.unwrap();
    assert_eq!(store.load_branch_summary(branch).await.unwrap(), None);
    assert_eq!(store.load_branch_summary(fork).await.unwrap(), expected);
}

#[tokio::test]
async fn resolve_branch_prefers_given_session_or_qualified_prefix() {
    let (store, _guard) = fresh_store().await;
    let s1 = SessionId::new();
    let b1 = store.begin_session_with_main_branch(&s1, 1).await.unwrap();
    let s2 = SessionId::new();
    let b2 = store.begin_session_with_main_branch(&s2, 2).await.unwrap();
    let bare = store.resolve_branch("main", Some(&s1)).await.unwrap();
    assert_eq!(bare, Some((s1, b1)));
    let s2_text = s2.to_string();
    let prefix = &s2_text[..8];
    let qualified = store
        .resolve_branch(&format!("{prefix}/main"), None)
        .await
        .unwrap();
    assert_eq!(qualified, Some((s2, b2)));
    let wildcard = store.resolve_branch("%/main", None).await.unwrap();
    assert_eq!(wildcard, None);
}

#[tokio::test]
async fn find_resumable_session_returns_unended() {
    let (store, _guard) = fresh_store().await;
    let s = SessionId::new();
    let branch = store.begin_session_with_main_branch(&s, 99).await.unwrap();
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
