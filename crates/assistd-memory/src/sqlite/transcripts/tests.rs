use std::sync::Arc;

use assistd_utils::backlog::{Signals, drain};
use tokio::sync::watch;

use super::*;
use crate::{
    ChunkSource, EmbeddingQueue, MessageChunker, PersistedRelayEnd, PersistedRelayEntry,
    PersistedRelayRun, Seq, SqliteSemanticStore,
};

/// The returned temp dir must outlive the store.
async fn fresh_store() -> (SqliteTranscriptStore, tempfile::TempDir) {
    let temp = tempfile::tempdir().unwrap();
    let handle = SqliteHandle::open(&temp.path().join("memory.db"))
        .await
        .unwrap();
    (SqliteTranscriptStore::new(handle), temp)
}

/// Cut every saved message that has no chunks, as the daemon's chunker does when woken.
async fn chunk_saved_messages(handle: &SqliteHandle) {
    let (_keep_running, shutdown) = watch::channel(false);
    let signals = Signals {
        wake: Arc::default(),
        shutdown,
    };
    drain(&mut MessageChunker::new(handle.clone()), &signals)
        .await
        .unwrap();
}

/// The text of every message chunk that has no vector, in the order they were cut.
async fn message_chunk_texts(handle: &SqliteHandle) -> Vec<String> {
    let pending = SqliteSemanticStore::new(handle.clone())
        .chunks_missing_embedding(ChunkSource::Message, None, 10)
        .await
        .unwrap();
    pending.into_iter().map(|(_, text)| text).collect()
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
    let branch = store.begin_session_with_main_branch(session).await.unwrap();
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
            PersistedMessage::tool_result("file1\nfile2", "c-1", "run", outcome, true),
        )
        .await
        .unwrap();
    chunk_saved_messages(&store.handle).await;
    assert_eq!(
        message_chunk_texts(&store.handle).await,
        ["list files"],
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
                        read_host: true,
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
    let main = store.begin_session_with_main_branch(session).await.unwrap();
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
    let older_main = store.begin_session_with_main_branch(older).await.unwrap();
    let newer_main = store
        .begin_session_with_main_branch(SessionId::new())
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
    let main = store.begin_session_with_main_branch(session).await.unwrap();
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
    let main = store.begin_session_with_main_branch(session).await.unwrap();
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
            released_answer_files: Vec::new(),
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
    let branch = store.begin_session_with_main_branch(session).await.unwrap();
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
    let summary = BranchSummary {
        kept_from_seq: Seq(2),
        steps_folded_through: None,
        compacted_after_seq: Seq(5),
        content: "turn one".into(),
        duration_ms: 4200,
    };
    store
        .record_branch_summary(branch, summary.clone())
        .await
        .unwrap();
    let expected = Some(summary);
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
    let b1 = store.begin_session_with_main_branch(s1).await.unwrap();
    let s2 = SessionId::new();
    let b2 = store.begin_session_with_main_branch(s2).await.unwrap();
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
async fn the_active_session_follows_every_switch_not_session_age() {
    let (store, _temp) = fresh_store().await;
    let active = |session_id, current_branch_id| {
        Some(ResumeCandidate {
            session_id,
            current_branch_id,
        })
    };
    assert_eq!(store.active_session().await.unwrap(), None);

    let older = SessionId::new();
    let older_main = store.begin_active_session(older).await.unwrap();
    let newer = SessionId::new();
    let newer_main = store.begin_active_session(newer).await.unwrap();
    assert_eq!(
        store.active_session().await.unwrap(),
        active(newer, newer_main)
    );

    store.activate_branch(older, older_main).await.unwrap();
    let background = SessionId::new();
    store
        .begin_session_with_main_branch(background)
        .await
        .unwrap();
    assert_eq!(
        store.active_session().await.unwrap(),
        active(older, older_main)
    );

    let fork = store.fork_and_switch(older_main, "alt").await.unwrap();
    assert_eq!(store.active_session().await.unwrap(), active(older, fork));

    store.clear_active_session().await.unwrap();
    assert_eq!(store.active_session().await.unwrap(), None);
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
            agent: "web_research".into(),
            run: Some(PersistedRelayRun {
                end: PersistedRelayEnd::Answered {
                    answer_file: PathBuf::from("/data/relays/s/relay-1.txt"),
                },
                duration_ms: 41_000,
                tokens: Some(9000),
                entries,
            }),
        },
        PersistedRelayStep {
            task: "compare them".into(),
            agent: "file_review".into(),
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
            agent: "file_review".into(),
            run: None,
        },
    ]
}

#[tokio::test]
async fn relay_trace_round_trips_follows_forks_and_goes_with_undo() {
    let (store, _temp) = fresh_store().await;
    let branch = store
        .begin_session_with_main_branch(SessionId::new())
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
    let result =
        PersistedMessage::tool_result("[error] relay: step 2", "c-1", "relay", outcome, true);
    let appended = store
        .append_tool_result(
            branch,
            turn,
            result,
            ToolTrace {
                relay_steps: relay_plan(),
                ..ToolTrace::default()
            },
        )
        .await
        .unwrap();

    let traces = store.load_relay_traces(branch).await.unwrap();
    assert_eq!(traces.get(&appended.id), Some(&relay_plan()));
    assert_eq!(traces.len(), 1);
    assert_eq!(
        store.relay_answer_files().await.unwrap(),
        [PathBuf::from("/data/relays/s/relay-1.txt")]
    );

    let fork = store.fork_and_switch(branch, "fork").await.unwrap();
    let undone = store.undo_last_turn(branch).await.unwrap();
    let forked = store.load_relay_traces(fork).await.unwrap();
    assert_eq!(forked.get(&appended.id), Some(&relay_plan()));
    assert!(
        undone.released_answer_files.is_empty(),
        "the fork still holds it"
    );

    let undone = store.undo_last_turn(fork).await.unwrap();
    assert_eq!(
        undone.released_answer_files,
        [PathBuf::from("/data/relays/s/relay-1.txt")]
    );
    assert!(store.relay_answer_files().await.unwrap().is_empty());
    let steps: i64 = store
        .handle
        .read("count relay_steps", |c| {
            c.query_row("SELECT COUNT(*) FROM relay_steps", [], |r| r.get(0))
        })
        .await
        .unwrap();
    assert_eq!(steps, 0, "undo on every branch deletes the trace");
}

#[tokio::test]
async fn a_message_saved_without_chunks_has_them_after_the_next_start() {
    let (store, temp) = fresh_store().await;
    let branch = store
        .begin_session_with_main_branch(SessionId::new())
        .await
        .unwrap();
    store
        .begin_turn(branch, PersistedMessage::user("where is the spare key"))
        .await
        .unwrap();
    store.handle.close().await.unwrap();

    let handle = SqliteHandle::open(&temp.path().join("memory.db"))
        .await
        .unwrap();
    assert!(message_chunk_texts(&handle).await.is_empty());
    chunk_saved_messages(&handle).await;
    chunk_saved_messages(&handle).await;
    assert_eq!(
        message_chunk_texts(&handle).await,
        ["where is the spare key"]
    );
}

#[tokio::test]
async fn a_message_whose_tool_calls_no_longer_decode_loads_without_them() {
    let (store, _temp) = fresh_store().await;
    let branch = store
        .begin_session_with_main_branch(SessionId::new())
        .await
        .unwrap();
    let turn = store
        .begin_turn(branch, PersistedMessage::user("list files"))
        .await
        .unwrap()
        .turn;
    let calls = vec![PersistedToolCall {
        id: "c-1".into(),
        name: "run".into(),
        arguments: serde_json::json!({"command": "ls"}),
    }];
    let call = store
        .append_message_to_branch(
            branch,
            turn,
            PersistedMessage::assistant_tool_calls("looking", calls, None),
        )
        .await
        .unwrap();
    store
        .append_message_to_branch(branch, turn, PersistedMessage::assistant_text("done", None))
        .await
        .unwrap();

    store
        .handle
        .write("store tool calls this build rejects", move |tx| {
            tx.execute(
                "UPDATE messages SET tool_calls = '[{\"renamed\":1}]' WHERE id = ?1",
                [call.id],
            )
        })
        .await
        .unwrap();

    let history = store
        .load_branch(branch, HistoryScope::Whole)
        .await
        .unwrap()
        .rows;
    assert_eq!(
        contents(&history),
        [
            (PersistedRole::User, "list files"),
            (PersistedRole::Assistant, "looking"),
            (PersistedRole::Assistant, "done"),
        ]
    );
    assert_eq!(
        history[1].message,
        PersistedMessage::assistant_text("looking", None)
    );
}

/// Every `accesses` row as `(message, action, target, label)`, in the order stored.
async fn stored_accesses(
    store: &SqliteTranscriptStore,
) -> Vec<(MessageId, i64, Vec<u8>, Option<String>)> {
    store
        .handle
        .read("stored accesses", |c| {
            c.prepare("SELECT message_id, action, target, label FROM accesses ORDER BY id")?
                .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?
                .collect()
        })
        .await
        .unwrap()
}

#[tokio::test]
async fn what_a_tool_call_touched_is_stored_with_its_result_until_no_branch_holds_it() {
    let (store, _temp) = fresh_store().await;
    let branch = store
        .begin_session_with_main_branch(SessionId::new())
        .await
        .unwrap();
    let turn = store
        .begin_turn(
            branch,
            PersistedMessage::user("read the lease, then the page"),
        )
        .await
        .unwrap()
        .turn;
    let outcome = ToolOutcome {
        exit_code: 0,
        duration_ms: 7,
    };
    let trace = ToolTrace {
        accesses: vec![
            Access::Read(PathBuf::from("/home/u/lease.pdf")),
            Access::Fetched {
                url: "https://example.com/tenancy".into(),
                title: Some("Tenancy law".into()),
            },
        ],
        relay_steps: Vec::new(),
    };
    let result = PersistedMessage::tool_result("both read", "c-1", "run", outcome, true);
    let appended = store
        .append_tool_result(branch, turn, result, trace)
        .await
        .unwrap();

    let stored = [
        (appended.id, 0, b"/home/u/lease.pdf".to_vec(), None),
        (
            appended.id,
            2,
            b"https://example.com/tenancy".to_vec(),
            Some("Tenancy law".to_string()),
        ),
    ];
    assert_eq!(stored_accesses(&store).await, stored);

    let fork = store.fork_and_switch(branch, "fork").await.unwrap();
    store.undo_last_turn(branch).await.unwrap();
    assert_eq!(stored_accesses(&store).await, stored, "the fork holds it");

    store.undo_last_turn(fork).await.unwrap();
    assert!(stored_accesses(&store).await.is_empty());
}
