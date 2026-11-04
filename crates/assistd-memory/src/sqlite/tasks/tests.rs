use assistd_ipc::{PlanStep, SuspendReason, TaskGrant, TaskPlan, TaskRunSummary, TaskTrigger};

use super::*;
use crate::{SqliteTranscriptStore, TranscriptStore};

/// The returned temp dir must outlive the store.
async fn fresh() -> (SqliteTaskStore, tempfile::TempDir) {
    let temp = tempfile::tempdir().unwrap();
    let handle = SqliteHandle::open(&temp.path().join("memory.db"))
        .await
        .unwrap();
    (SqliteTaskStore::new(handle), temp)
}

fn approved_spec(name: &str, prompt: &str) -> TaskSpec {
    TaskSpec {
        name: name.to_string().try_into().unwrap(),
        prompt: prompt.to_string().try_into().unwrap(),
        triggers: Vec::new(),
        enabled: true,
        grants: vec![TaskGrant::WebHost {
            host: "example.org".into(),
        }],
        plan: Some(TaskPlan {
            summary: "fetch it".into(),
            steps: vec![PlanStep {
                description: "fetch the page".into(),
                actions: Vec::new(),
            }],
        }),
    }
}

#[tokio::test]
async fn only_a_changed_prompt_drops_the_plan_and_grants() {
    let (store, _temp) = fresh().await;
    let saved = store
        .save(None, approved_spec("digest", "summarize example.org"))
        .await
        .unwrap();

    let renamed = TaskSpec {
        triggers: vec![TaskTrigger::Manual],
        ..approved_spec("morning digest", "summarize example.org")
    };
    let kept = store.save(Some(saved.id), renamed.clone()).await.unwrap();
    assert_eq!(kept.spec, renamed);
    assert_eq!(store.get(saved.id).await.unwrap().spec, renamed);

    let reworded = approved_spec("morning digest", "summarize example.com");
    let cleared = store.save(Some(saved.id), reworded).await.unwrap();
    let stored = store.get(saved.id).await.unwrap();
    assert_eq!(stored, cleared);
    assert!(stored.spec.grants.is_empty());
    assert_eq!(stored.spec.plan, None);
    assert_eq!(stored.spec.prompt.as_str(), "summarize example.com");
}

#[tokio::test]
async fn save_refuses_a_taken_name_and_an_unknown_id() {
    let (store, _temp) = fresh().await;
    store
        .save(None, approved_spec("digest", "a"))
        .await
        .unwrap();
    let other = store.save(None, approved_spec("other", "b")).await.unwrap();

    assert!(matches!(
        store.save(None, approved_spec("digest", "c")).await,
        Err(MemoryError::TaskNameTaken(name)) if name == "digest"
    ));
    assert!(matches!(
        store
            .save(Some(other.id), approved_spec("digest", "b"))
            .await,
        Err(MemoryError::TaskNameTaken(_))
    ));
    assert!(matches!(
        store
            .save(Some(TaskId(999)), approved_spec("new", "d"))
            .await,
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
    let task = tasks
        .save(None, approved_spec("digest", "a"))
        .await
        .unwrap();
    let user_session = SessionId::new();
    transcripts
        .begin_session_with_main_branch(user_session, 1)
        .await
        .unwrap();

    let run = tasks.queue_run(task.id, TaskTrigger::Manual).await.unwrap();
    let run_session = SessionId::new();
    transcripts
        .begin_session_with_main_branch(run_session, 1)
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
        .save(None, approved_spec("digest", "a"))
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
