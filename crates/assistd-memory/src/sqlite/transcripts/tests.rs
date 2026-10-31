use super::*;
use crate::{
    PersistedRelayEnd, PersistedRelayEntry, PersistedRelayProfile, PersistedRelayRun, Seq,
};

/// The returned temp dir must outlive the store.
async fn fresh_store() -> (SqliteTranscriptStore, tempfile::TempDir) {
    let temp = tempfile::tempdir().unwrap();
    let handle = SqliteHandle::open(&temp.path().join("memory.db"))
        .await
        .unwrap();
    (SqliteTranscriptStore::new(handle), temp)
}

fn look_first() -> Reasoning {
    Reasoning {
        text: "look first".into(),
        duration_ms: 1200,
    }
}

fn contents(history: &[HistoryRow]) -> Vec<(PersistedRole, &str)> {
    history
        .iter()
        .map(|r| (r.message.role(), r.message.content.as_str()))
        .collect()
}

#[tokio::test]
async fn a_tool_turn_round_trips_with_its_stats_on_its_last_row() {
    let (store, _temp) = fresh_store().await;
    let session = SessionId::new();
    let branch = store
        .begin_session_with_main_branch(session, 1)
        .await
        .unwrap();
    let turn = store
        .begin_turn(branch, PersistedMessage::user("list files"))
        .await
        .unwrap()
        .turn;

    let outcome = ToolOutcome {
        exit_code: 2,
        duration_ms: 41,
    };
    let calls = vec![PersistedToolCall {
        id: "c-1".into(),
        name: "run".into(),
        arguments: serde_json::json!({"command": "ls"}),
    }];
    let call = store
        .append_message_to_branch(
            branch,
            turn,
            PersistedMessage::assistant_tool_calls("", calls.clone(), Some(look_first())),
        )
        .await
        .unwrap();
    let result = store
        .append_message_to_branch(
            branch,
            turn,
            PersistedMessage::tool_result("file1\nfile2", "c-1", "run", outcome),
        )
        .await
        .unwrap();
    assert!(
        result.chunks.is_empty(),
        "tool results are never chunked for embedding"
    );

    let stats = TurnStats {
        duration_ms: 5300,
        generated_tokens: 812,
    };
    store.record_turn_stats(turn, stats).await.unwrap();

    let history = store
        .load_branch(branch, HistoryScope::Whole)
        .await
        .unwrap()
        .rows;
    assert_eq!(
        history[1..],
        [
            HistoryRow {
                message_id: call.id,
                seq: Seq(1),
                message: PersistedMessage {
                    content: String::new(),
                    kind: MessageKind::Assistant {
                        reasoning: Some(look_first()),
                        tool_calls: calls,
                    },
                },
                turn_stats: None,
            },
            HistoryRow {
                message_id: result.id,
                seq: Seq(2),
                message: PersistedMessage {
                    content: "file1\nfile2".into(),
                    kind: MessageKind::Tool {
                        call_id: "c-1".into(),
                        tool_name: "run".into(),
                        outcome,
                    },
                },
                turn_stats: Some(stats),
            },
        ]
    );
}

#[tokio::test]
async fn fork_creates_independent_branch_sharing_history() {
    let (store, _temp) = fresh_store().await;
    let session = SessionId::new();
    let main = store
        .begin_session_with_main_branch(session, 1)
        .await
        .unwrap();
    let turn = store
        .begin_turn(main, PersistedMessage::user("hello"))
        .await
        .unwrap()
        .turn;
    store
        .append_message_to_branch(main, turn, PersistedMessage::assistant_text("hi", None))
        .await
        .unwrap();

    let fork = store.fork_and_switch(main, "experiment").await.unwrap();
    assert_eq!(store.current_branch(session).await.unwrap(), Some(fork));
    assert!(matches!(
        store.fork_and_switch(main, "experiment").await,
        Err(MemoryError::BranchNameTaken(name)) if name == "experiment"
    ));
    let main_history = store
        .load_branch(main, HistoryScope::Whole)
        .await
        .unwrap()
        .rows;
    let fork_history = store
        .load_branch(fork, HistoryScope::Whole)
        .await
        .unwrap()
        .rows;
    assert_eq!(main_history.len(), 2);
    assert_eq!(
        fork_history, main_history,
        "fork references the same message rows"
    );

    let branches = store.list_branches().await.unwrap();
    let fork_info = branches.iter().find(|b| b.name == "experiment").unwrap();
    assert_eq!(fork_info.parent_branch_id, Some(main));
    assert_eq!(fork_info.parent_branch_name.as_deref(), Some("main"));
    assert_eq!(fork_info.fork_point_seq, Some(Seq(1)));
    assert_eq!(fork_info.message_count, 2);
    assert_eq!(
        fork_info.last_active_at, fork_info.created_at,
        "inherited messages predate the fork"
    );
}

#[tokio::test]
async fn branches_list_by_newest_message_not_session_start() {
    let (store, _temp) = fresh_store().await;
    let older = SessionId::new();
    let older_main = store
        .begin_session_with_main_branch(older, 1)
        .await
        .unwrap();
    let newer_main = store
        .begin_session_with_main_branch(SessionId::new(), 1)
        .await
        .unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(2)).await;
    store
        .begin_turn(older_main, PersistedMessage::user("back"))
        .await
        .unwrap();

    let branches = store.list_branches().await.unwrap();
    let order: Vec<_> = branches.iter().map(|b| b.branch_id).collect();
    assert_eq!(order, [older_main, newer_main]);
    assert!(branches[0].last_active_at > branches[1].created_at);
}

#[tokio::test]
async fn append_to_one_branch_does_not_show_on_the_other() {
    let (store, _temp) = fresh_store().await;
    let session = SessionId::new();
    let main = store
        .begin_session_with_main_branch(session, 1)
        .await
        .unwrap();
    store
        .begin_turn(main, PersistedMessage::user("q"))
        .await
        .unwrap();
    let fork = store.fork_and_switch(main, "alt").await.unwrap();
    store
        .begin_turn(fork, PersistedMessage::user("alt q"))
        .await
        .unwrap();
    let main_history = store
        .load_branch(main, HistoryScope::Whole)
        .await
        .unwrap()
        .rows;
    let fork_history = store
        .load_branch(fork, HistoryScope::Whole)
        .await
        .unwrap()
        .rows;
    assert_eq!(contents(&main_history), [(PersistedRole::User, "q")]);
    assert_eq!(
        contents(&fork_history),
        [(PersistedRole::User, "q"), (PersistedRole::User, "alt q")]
    );
}

#[tokio::test]
async fn undo_removes_only_the_last_turn() {
    let (store, _temp) = fresh_store().await;
    let session = SessionId::new();
    let main = store
        .begin_session_with_main_branch(session, 1)
        .await
        .unwrap();
    let t1 = store
        .begin_turn(main, PersistedMessage::user("first"))
        .await
        .unwrap()
        .turn;
    store
        .append_message_to_branch(main, t1, PersistedMessage::assistant_text("a", None))
        .await
        .unwrap();

    let t2 = store
        .begin_turn(main, PersistedMessage::user("second"))
        .await
        .unwrap()
        .turn;
    store
        .append_message_to_branch(main, t2, PersistedMessage::assistant_text("b", None))
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

    let history = store
        .load_branch(main, HistoryScope::Whole)
        .await
        .unwrap()
        .rows;
    assert_eq!(
        contents(&history),
        [
            (PersistedRole::User, "first"),
            (PersistedRole::Assistant, "a")
        ]
    );

    store
        .begin_turn(main, PersistedMessage::user("third"))
        .await
        .unwrap();
    let seqs: Vec<Seq> = store
        .load_branch(main, HistoryScope::Whole)
        .await
        .unwrap()
        .rows
        .iter()
        .map(|row| row.seq)
        .collect();
    assert_eq!(
        seqs,
        [Seq(0), Seq(1), Seq(4)],
        "undo must not free a seq for reuse"
    );
}

#[tokio::test]
async fn summary_follows_kept_turns_forks_and_undo() {
    let (store, _temp) = fresh_store().await;
    let session = SessionId::new();
    let branch = store
        .begin_session_with_main_branch(session, 1)
        .await
        .unwrap();
    for question in ["one", "two", "three"] {
        let turn = store
            .begin_turn(branch, PersistedMessage::user(question))
            .await
            .unwrap()
            .turn;
        store
            .append_message_to_branch(
                branch,
                turn,
                PersistedMessage::assistant_text("answer", None),
            )
            .await
            .unwrap();
    }
    let cut = SummaryCut {
        kept_turns: NonZeroUsize::new(2).unwrap(),
        kept_steps: None,
    };
    store
        .record_branch_summary(branch, "turn one".into(), cut, 4200)
        .await
        .unwrap();
    let expected = Some(BranchSummary {
        kept_from_seq: Seq(2),
        steps_folded_through: None,
        compacted_after_seq: Seq(5),
        content: "turn one".into(),
        duration_ms: 4200,
    });
    assert_eq!(
        store
            .load_branch(branch, HistoryScope::Whole)
            .await
            .unwrap()
            .summary,
        expected
    );
    let unsummarized: Vec<Seq> = store
        .load_branch(branch, HistoryScope::Unsummarized)
        .await
        .unwrap()
        .rows
        .iter()
        .map(|row| row.seq)
        .collect();
    assert_eq!(unsummarized, [Seq(2), Seq(3), Seq(4), Seq(5)]);

    let fork = store.fork_and_switch(branch, "alt").await.unwrap();
    assert_eq!(
        store
            .load_branch(fork, HistoryScope::Whole)
            .await
            .unwrap()
            .summary,
        expected
    );

    store.undo_last_turn(branch).await.unwrap();
    store.undo_last_turn(branch).await.unwrap();
    assert_eq!(
        store
            .load_branch(branch, HistoryScope::Whole)
            .await
            .unwrap()
            .summary,
        expected
    );
    store.undo_last_turn(branch).await.unwrap();
    assert_eq!(
        store
            .load_branch(branch, HistoryScope::Whole)
            .await
            .unwrap()
            .summary,
        None
    );
    assert_eq!(
        store
            .load_branch(fork, HistoryScope::Whole)
            .await
            .unwrap()
            .summary,
        expected
    );
}

#[tokio::test]
async fn resolve_branch_prefers_given_session_or_qualified_prefix() {
    let (store, _temp) = fresh_store().await;
    let s1 = SessionId::new();
    let b1 = store.begin_session_with_main_branch(s1, 1).await.unwrap();
    let s2 = SessionId::new();
    let b2 = store.begin_session_with_main_branch(s2, 2).await.unwrap();
    let bare = store.resolve_branch("main", Some(s1)).await.unwrap();
    assert_eq!(bare, Some((s1, b1)));
    let prefix = s2.to_string()[..8].to_ascii_uppercase();
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
    let (store, _temp) = fresh_store().await;
    let s = SessionId::new();
    let branch = store.begin_session_with_main_branch(s, 99).await.unwrap();
    let cand = store
        .find_resumable_session()
        .await
        .unwrap()
        .expect("unended session is resumable");
    assert_eq!(cand.session_id, s);
    assert_eq!(cand.current_branch_id, branch);
    assert_eq!(cand.daemon_pid, 99);

    store.end_session(s).await.unwrap();
    assert_eq!(store.find_resumable_session().await.unwrap(), None);
}

fn relay_plan() -> Vec<PersistedRelayStep> {
    let finished = ToolOutcome {
        exit_code: 1,
        duration_ms: 12,
    };
    let entries = vec![
        PersistedRelayEntry::Reasoning {
            text: "grep first".into(),
            duration_ms: 900,
        },
        PersistedRelayEntry::Tool {
            name: "run".into(),
            arguments: serde_json::json!({"command": "grep -rn Error /in"}),
            output: "no matches".into(),
            outcome: Some(finished),
        },
        PersistedRelayEntry::Compaction {
            duration_ms: Some(3000),
        },
        PersistedRelayEntry::Text("Nothing matched.".into()),
        PersistedRelayEntry::Tool {
            name: "run".into(),
            arguments: serde_json::json!({"command": "cat /in/x"}),
            output: String::new(),
            outcome: None,
        },
        PersistedRelayEntry::Compaction { duration_ms: None },
    ];
    vec![
        PersistedRelayStep {
            task: "fetch the notes".into(),
            profile: PersistedRelayProfile::Fetch,
            run: Some(PersistedRelayRun {
                end: PersistedRelayEnd::Answered { truncated: true },
                duration_ms: 41_000,
                tokens: Some(9000),
                entries,
            }),
        },
        PersistedRelayStep {
            task: "compare them".into(),
            profile: PersistedRelayProfile::Read,
            run: Some(PersistedRelayRun {
                end: PersistedRelayEnd::Failed {
                    message: "server crashed".into(),
                },
                duration_ms: 500,
                tokens: None,
                entries: Vec::new(),
            }),
        },
        PersistedRelayStep {
            task: "summarise".into(),
            profile: PersistedRelayProfile::Read,
            run: None,
        },
    ]
}

#[tokio::test]
async fn relay_trace_round_trips_follows_forks_and_goes_with_undo() {
    let (store, _temp) = fresh_store().await;
    let branch = store
        .begin_session_with_main_branch(SessionId::new(), 1)
        .await
        .unwrap();
    let turn = store
        .begin_turn(branch, PersistedMessage::user("research it"))
        .await
        .unwrap()
        .turn;
    let outcome = ToolOutcome {
        exit_code: 1,
        duration_ms: 42_000,
    };
    let result = PersistedMessage::tool_result("[error] relay: step 2", "c-1", "relay", outcome);
    let appended = store
        .append_relay_result(branch, turn, result, relay_plan())
        .await
        .unwrap();

    let traces = store.load_relay_traces(branch).await.unwrap();
    assert_eq!(traces.get(&appended.id), Some(&relay_plan()));
    assert_eq!(traces.len(), 1);

    let fork = store.fork_and_switch(branch, "fork").await.unwrap();
    store.undo_last_turn(branch).await.unwrap();
    let forked = store.load_relay_traces(fork).await.unwrap();
    assert_eq!(forked.get(&appended.id), Some(&relay_plan()));

    store.undo_last_turn(fork).await.unwrap();
    let steps: i64 = store
        .handle
        .reader()
        .call(|c| c.query_row("SELECT COUNT(*) FROM relay_steps", [], |r| r.get(0)))
        .await
        .unwrap();
    assert_eq!(steps, 0, "undo on every branch deletes the trace");
}
