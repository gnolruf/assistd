use std::path::PathBuf;

use assistd_ipc::{PermissionMode, SuspendReason, TaskRunSummary, TaskTrigger};

use super::*;
use crate::{
    PersistedMessage, PersistedRelayEnd, PersistedRelayRun, PersistedRelayStep,
    SqliteTranscriptStore, ToolOutcome, TranscriptStore,
};

/// The returned temp dir must outlive the store.
async fn fresh() -> (SqliteTaskStore, tempfile::TempDir) {
    let temp = tempfile::tempdir().unwrap();
    let handle = SqliteHandle::open(&temp.path().join("memory.db"))
        .await
        .unwrap();
    (SqliteTaskStore::new(handle), temp)
}

fn sample_spec(name: &str, prompt: &str) -> TaskSpec {
    TaskSpec {
        name: name.to_string().try_into().unwrap(),
        prompt: prompt.to_string().try_into().unwrap(),
        triggers: Vec::new(),
        enabled: true,
        mode: PermissionMode::AcceptEdits,
        allowed_hosts: vec!["example.org".to_string().try_into().unwrap()],
        announce: true,
        kept_runs: None,
    }
}

#[tokio::test]
async fn save_refuses_a_taken_name_and_an_unknown_id() {
    let (store, _temp) = fresh().await;
    store.save(None, sample_spec("digest", "a")).await.unwrap();
    let other = store.save(None, sample_spec("other", "b")).await.unwrap();

    assert!(matches!(
        store.save(None, sample_spec("digest", "c")).await,
        Err(MemoryError::TaskNameTaken(name)) if name == "digest"
    ));
    assert!(matches!(
        store.save(Some(other.id), sample_spec("digest", "b")).await,
        Err(MemoryError::TaskNameTaken(_))
    ));
    assert!(matches!(
        store.save(Some(TaskId(999)), sample_spec("new", "d")).await,
        Err(MemoryError::TaskNotFound(TaskId(999)))
    ));
}

#[tokio::test]
async fn a_run_session_is_listed_with_its_outcome_and_never_resumed() {
    let temp = tempfile::tempdir().unwrap();
    let handle = SqliteHandle::open(&temp.path().join("memory.db"))
        .await
        .unwrap();
    let tasks = SqliteTaskStore::new(handle.clone());
    let transcripts = SqliteTranscriptStore::new(handle);
    let task = tasks.save(None, sample_spec("digest", "a")).await.unwrap();
    let user_session = SessionId::new();
    transcripts
        .begin_session_with_main_branch(user_session)
        .await
        .unwrap();

    let run = tasks.queue_run(task.id, TaskTrigger::Manual).await.unwrap();
    let run_session = SessionId::new();
    transcripts
        .begin_session_with_main_branch(run_session)
        .await
        .unwrap();
    tasks.start_run(run, run_session).await.unwrap();
    tasks
        .finish_run(run, TaskRunEnd::Succeeded { reply: "ok".into() })
        .await
        .unwrap();

    assert!(matches!(
        tasks.finish_run(run, TaskRunEnd::Cancelled).await,
        Err(MemoryError::TaskRunNotFound(id)) if id == run
    ));
    let resumable = transcripts.find_resumable_session().await.unwrap().unwrap();
    assert_eq!(resumable.session_id, user_session);
    let tags: Vec<_> = transcripts
        .list_branches()
        .await
        .unwrap()
        .into_iter()
        .map(|branch| (branch.session_id, branch.task_run))
        .collect();
    assert!(tags.contains(&(user_session, None)));
    assert!(tags.contains(&(
        run_session,
        Some(TaskRunSummary {
            task: task.id,
            run,
            task_name: task.spec.name.clone(),
            state: TaskRunState::Succeeded,
        })
    )));
}

#[tokio::test]
async fn queue_order_pause_and_interrupted_runs_survive_a_reopen() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("memory.db");
    let tasks = SqliteTaskStore::new(SqliteHandle::open(&path).await.unwrap());
    let task = tasks
        .save(None, sample_spec("digest", "a"))
        .await
        .unwrap()
        .id;
    let mut runs = Vec::new();
    for _ in 0..5 {
        runs.push(tasks.queue_run(task, TaskTrigger::Manual).await.unwrap());
    }
    let [running, first, second, third, cancelled] = runs[..] else {
        unreachable!()
    };
    tasks.start_run(running, SessionId::new()).await.unwrap();
    tasks
        .set_run_waiting(first, Some(SuspendReason::GpuBusy))
        .await
        .unwrap();
    tasks
        .reorder_queue(vec![third, first, second])
        .await
        .unwrap();
    tasks
        .finish_run(cancelled, TaskRunEnd::Cancelled)
        .await
        .unwrap();
    let late = tasks.queue_run(task, TaskTrigger::Manual).await.unwrap();
    tasks.set_queue_paused(true).await.unwrap();
    drop(tasks);

    let reopened = SqliteTaskStore::new(SqliteHandle::open(&path).await.unwrap());
    let recovered = reopened
        .recover_queue("daemon stopped".into())
        .await
        .unwrap();
    let waiting: Vec<_> = recovered
        .waiting
        .iter()
        .map(|waiting| waiting.run)
        .collect();
    assert_eq!(waiting, [third, first, second, late]);
    assert!(recovered.paused);
    assert!(matches!(
        reopened.finish_run(running, TaskRunEnd::Cancelled).await,
        Err(MemoryError::TaskRunNotFound(_))
    ));
    reopened.start_run(first, SessionId::new()).await.unwrap();
}

#[tokio::test]
async fn a_fire_coalesces_into_an_unfinished_run_and_records_the_next_fire() {
    let (store, _temp) = fresh().await;
    let task = store
        .save(None, sample_spec("digest", "summarize example.org"))
        .await
        .unwrap()
        .id;
    let next = DateTime::from_timestamp(1_800_000_000, 0);

    let first = store.fire(task, TaskTrigger::Manual, next).await.unwrap();
    assert!(first.is_some());
    assert_eq!(store.get(task).await.unwrap().next_fire, next);
    let coalesced = store.fire(task, TaskTrigger::Manual, None).await.unwrap();
    assert_eq!(coalesced, None);
    assert_eq!(store.get(task).await.unwrap().next_fire, None);

    store
        .finish_run(first.unwrap(), TaskRunEnd::Cancelled)
        .await
        .unwrap();
    assert!(
        store
            .fire(task, TaskTrigger::Manual, next)
            .await
            .unwrap()
            .is_some()
    );
    store.set_enabled(task, true).await.unwrap();
    assert_eq!(store.get(task).await.unwrap().next_fire, None);
}

/// Run task `task` to success in a new session holding one relay answer saved to `answer`.
async fn finished_run_with_relay(
    tasks: &SqliteTaskStore,
    transcripts: &SqliteTranscriptStore,
    task: TaskId,
    answer: &str,
) -> SessionId {
    let run = tasks.queue_run(task, TaskTrigger::Manual).await.unwrap();
    let session = SessionId::new();
    let branch = transcripts
        .begin_session_with_main_branch(session)
        .await
        .unwrap();
    tasks.start_run(run, session).await.unwrap();
    let turn = transcripts
        .begin_turn(branch, PersistedMessage::user("go"))
        .await
        .unwrap()
        .turn;
    let outcome = ToolOutcome {
        exit_code: 0,
        duration_ms: 1,
    };
    let step = PersistedRelayStep {
        task: "look".into(),
        agent: "web_research".into(),
        run: Some(PersistedRelayRun {
            end: PersistedRelayEnd::Answered {
                answer_file: PathBuf::from(answer),
            },
            duration_ms: 1,
            tokens: None,
            entries: Vec::new(),
        }),
    };
    let result = PersistedMessage::tool_result("done", "c-1", "relay", outcome);
    transcripts
        .append_relay_result(branch, turn, result, vec![step])
        .await
        .unwrap();
    tasks
        .finish_run(run, TaskRunEnd::Succeeded { reply: "ok".into() })
        .await
        .unwrap();
    session
}

#[tokio::test]
async fn prune_keeps_each_tasks_newest_finished_runs_and_the_spared_session() {
    let temp = tempfile::tempdir().unwrap();
    let handle = SqliteHandle::open(&temp.path().join("memory.db"))
        .await
        .unwrap();
    let tasks = SqliteTaskStore::new(handle.clone());
    let transcripts = SqliteTranscriptStore::new(handle.clone());
    let task = tasks
        .save(None, sample_spec("digest", "a"))
        .await
        .unwrap()
        .id;
    let mut sessions = Vec::new();
    for answer in ["/r/1", "/r/2", "/r/3", "/r/4"] {
        sessions.push(finished_run_with_relay(&tasks, &transcripts, task, answer).await);
    }
    tasks.queue_run(task, TaskTrigger::Manual).await.unwrap();
    let weekly_spec = TaskSpec {
        kept_runs: Some(KeptRuns::try_from(2).unwrap()),
        ..sample_spec("weekly", "b")
    };
    let weekly = tasks.save(None, weekly_spec.clone()).await.unwrap().id;
    assert_eq!(tasks.get(weekly).await.unwrap().spec, weekly_spec);
    for answer in ["/w/1", "/w/2", "/w/3"] {
        finished_run_with_relay(&tasks, &transcripts, weekly, answer).await;
    }

    let mut pruned = tasks.prune_runs(1, sessions[1]).await.unwrap();

    assert_eq!(pruned.runs, 3);
    pruned.released_answer_files.sort();
    assert_eq!(
        pruned.released_answer_files,
        [
            PathBuf::from("/r/1"),
            PathBuf::from("/r/3"),
            PathBuf::from("/w/1")
        ]
    );
    let kept: Vec<_> = tasks
        .runs(task, 10)
        .await
        .unwrap()
        .into_iter()
        .map(|record| record.session_id)
        .collect();
    assert_eq!(
        kept,
        [
            None,
            Some(sessions[3].to_string()),
            Some(sessions[1].to_string())
        ]
    );
    let mut remaining = transcripts.relay_answer_files().await.unwrap();
    remaining.sort();
    assert_eq!(
        remaining,
        ["/r/2", "/r/4", "/w/2", "/w/3"].map(PathBuf::from)
    );
    let counts: (i64, i64) = handle
        .reader()
        .call(|c| {
            c.query_row(
                "SELECT (SELECT COUNT(*) FROM sessions), (SELECT COUNT(*) FROM messages)",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
        })
        .await
        .unwrap();
    assert_eq!(counts, (4, 8));
}
