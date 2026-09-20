use super::*;
use tokio::sync::watch;

async fn fresh_store() -> (SqliteConversationStore, tokio::task::JoinHandle<()>) {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("memory.db");
    std::mem::forget(temp);
    let (_tx, rx) = watch::channel(false);
    let (handle, writer) = SqliteHandle::open(&path, rx).await.unwrap();
    (SqliteConversationStore::new(Arc::new(handle)), writer)
}

#[tokio::test]
async fn round_trip_user_and_assistant_messages() {
    let (store, _w) = fresh_store().await;
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
    let roles: Vec<PersistedRole> = history.iter().map(|r| r.role).collect();
    assert_eq!(roles, [PersistedRole::User, PersistedRole::Assistant]);
    assert_eq!(history[1].content, "four");

    let recent = store.recent_turns(5).await.unwrap();
    assert_eq!(recent.len(), 1);
    assert_eq!(recent[0].user_text, "what is 2+2?");
    assert_eq!(recent[0].message_count, 2);
}

#[tokio::test]
async fn assistant_with_tool_calls_persists_json() {
    let (store, _w) = fresh_store().await;
    let (session, branch) = store.begin_session_with_main_branch(1).await.unwrap();
    let turn = store.begin_turn(&session, "list files").await.unwrap();

    let calls = serde_json::json!([{"id": "c-1", "name": "run", "arguments": {"command": "ls"}}]);
    let id = store
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
    assert_ne!(id, result_id);

    let conn = store.handle.conn();
    let (assistant_calls, tool_call_id, tool_name): (
        Option<String>,
        Option<String>,
        Option<String>,
    ) = conn
        .call(move |c| -> rusqlite::Result<_> {
            c.query_row(
                "SELECT (SELECT tool_calls FROM conversations WHERE id = ?1),
                        (SELECT tool_call_id FROM conversations WHERE id = ?2),
                        (SELECT tool_name FROM conversations WHERE id = ?2)",
                rusqlite::params![id, result_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
        })
        .await
        .unwrap();
    assert!(assistant_calls.unwrap().contains("\"command\":\"ls\""));
    assert_eq!(tool_call_id.as_deref(), Some("c-1"));
    assert_eq!(tool_name.as_deref(), Some("run"));
}

#[tokio::test]
async fn no_conversation_store_returns_empty() {
    let store = NoConversationStore;
    assert!(store.recent_turns(10).await.unwrap().is_empty());
    assert!(store.list_branches().await.unwrap().is_empty());
}

#[tokio::test]
async fn begin_session_with_main_branch_inserts_session_and_main_branch() {
    let (store, _w) = fresh_store().await;
    let (session, branch) = store.begin_session_with_main_branch(123).await.unwrap();
    let current = store.get_current_branch(&session).await.unwrap();
    assert_eq!(current, Some(branch));
    let branches = store.list_branches().await.unwrap();
    assert_eq!(branches.len(), 1);
    assert_eq!(branches[0].name, "main");
    assert!(branches[0].is_current_in_session);
    assert_eq!(branches[0].parent_branch_id, None);
    assert_eq!(branches[0].message_count, 0);
}

#[tokio::test]
async fn fork_creates_independent_branch_sharing_history() {
    let (store, _w) = fresh_store().await;
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
    assert_eq!(fork_history.len(), 2);
    // Same conversation rows are referenced; no row duplication.
    assert_eq!(
        main_history
            .iter()
            .map(|r| r.conversation_id)
            .collect::<Vec<_>>(),
        fork_history
            .iter()
            .map(|r| r.conversation_id)
            .collect::<Vec<_>>()
    );
    let branches = store.list_branches().await.unwrap();
    let fork_info = branches.iter().find(|b| b.name == "experiment").unwrap();
    assert_eq!(fork_info.parent_branch_name.as_deref(), Some("main"));
    assert_eq!(fork_info.fork_point_seq, Some(1));
    assert_eq!(fork_info.message_count, 2);
}

#[tokio::test]
async fn append_to_one_branch_does_not_show_on_the_other() {
    let (store, _w) = fresh_store().await;
    let (session, main) = store.begin_session_with_main_branch(1).await.unwrap();
    let turn = store.begin_turn(&session, "q").await.unwrap();
    store
        .append_message_to_branch(&session, main, Some(turn), PersistedMessage::user("q"))
        .await
        .unwrap();
    let fork = store.fork_branch(main, "alt").await.unwrap();
    // Append to fork only.
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
    assert_eq!(main_history.len(), 1);
    assert_eq!(fork_history.len(), 2);
}

#[tokio::test]
async fn undo_removes_last_turn_from_branch_only() {
    let (store, _w) = fresh_store().await;
    let (session, main) = store.begin_session_with_main_branch(1).await.unwrap();
    // Two turns: "first", "second".
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
    assert_eq!(outcome.removed_messages, 2);
    assert_eq!(outcome.last_user_text.as_deref(), Some("second"));

    let history = store.load_branch_history(main).await.unwrap();
    assert_eq!(history.len(), 2);
    assert_eq!(history[0].content, "first");
    assert_eq!(history[1].content, "a");
}

#[tokio::test]
async fn resolve_branch_qualified_form() {
    let (store, _w) = fresh_store().await;
    let (s1, _) = store.begin_session_with_main_branch(1).await.unwrap();
    let (s2, _) = store.begin_session_with_main_branch(2).await.unwrap();
    // Two sessions, both with a "main" branch.
    let bare = store.resolve_branch("main", Some(&s1)).await.unwrap();
    assert!(bare.is_some());
    assert_eq!(bare.unwrap().0, s1);
    let prefix = &s2.0[..8];
    let qualified = store
        .resolve_branch(&format!("{prefix}/main"), None)
        .await
        .unwrap();
    assert!(qualified.is_some());
    assert_eq!(qualified.unwrap().0, s2);
}

#[tokio::test]
async fn find_resumable_session_returns_unended() {
    let (store, _w) = fresh_store().await;
    let (s, _) = store.begin_session_with_main_branch(99).await.unwrap();
    let resume = store.find_resumable_session().await.unwrap();
    assert!(resume.is_some());
    let cand = resume.unwrap();
    assert_eq!(cand.session_id, s);
    assert_eq!(cand.daemon_pid, 99);
    store.end_session(&s).await.unwrap();
    // Once ended, no longer resumable.
    let resume2 = store.find_resumable_session().await.unwrap();
    assert!(resume2.is_none());
}
