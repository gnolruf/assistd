use assistd_ipc::{PlanStep, TaskGrant, TaskPlan, TaskTrigger};

use super::*;

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
