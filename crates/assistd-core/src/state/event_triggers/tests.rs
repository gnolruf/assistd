use assistd_ipc::{TaskName, TaskPrompt, TaskSpec};

use super::*;

fn after(task: i64) -> TaskTrigger {
    TaskTrigger::Event {
        event: TaskEvent::TaskFinished {
            task: TaskId(task),
            outcome: FinishOutcome::Either,
        },
        cooldown_secs: 60,
    }
}

fn task(id: i64, triggers: Vec<TaskTrigger>) -> TaskRecord {
    TaskRecord {
        id: TaskId(id),
        spec: TaskSpec {
            name: TaskName::try_from(format!("task {id}")).unwrap(),
            prompt: TaskPrompt::try_from("work".to_string()).unwrap(),
            triggers,
            enabled: true,
            mode: assistd_ipc::PermissionMode::Manual,
            allowed_hosts: Vec::new(),
            announce: false,
        },
        next_fire: None,
    }
}

#[test]
fn a_task_finished_chain_that_would_loop_is_refused() {
    let tasks = [
        task(1, vec![after(3)]),
        task(2, vec![after(1)]),
        task(3, Vec::new()),
    ];
    assert!(matches!(
        check_chain(Some(TaskId(3)), &[after(2)], &tasks),
        Err(TriggerError::ChainCycle(TaskId(2)))
    ));
    assert!(matches!(
        check_chain(Some(TaskId(3)), &[after(3)], &tasks),
        Err(TriggerError::ChainCycle(TaskId(3)))
    ));
    assert!(check_chain(Some(TaskId(1)), &[after(3)], &tasks).is_ok());
    assert!(check_chain(None, &[after(2)], &tasks).is_ok());
}
