//! Handlers for the `Task*` variants of `Request`.

use std::sync::Arc;

use tokio::sync::mpsc;

use assistd_ipc::{Event, TaskId, TaskSpec};
use assistd_memory::{Result as MemoryResult, TaskRecord};

use super::{AppState, DispatchError, send_error};

impl AppState {
    pub(super) async fn handle_task_list(
        self: Arc<Self>,
        id: String,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        let tasks = self.memory.tasks.list().await;
        send_tasks(id, "task list", tasks, &tx).await
    }

    pub(super) async fn handle_task_get(
        self: Arc<Self>,
        id: String,
        task: TaskId,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        let record = self.memory.tasks.get(task).await;
        send_tasks(id, "task get", record.map(|r| vec![r]), &tx).await
    }

    pub(super) async fn handle_task_save(
        self: Arc<Self>,
        id: String,
        task: Option<TaskId>,
        spec: TaskSpec,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        let record = self.memory.tasks.save(task, spec).await;
        send_tasks(id, "task save", record.map(|r| vec![r]), &tx).await
    }

    pub(super) async fn handle_task_delete(
        self: Arc<Self>,
        id: String,
        task: TaskId,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        let deleted = self.memory.tasks.delete(task).await;
        if deleted.is_ok() {
            self.runtime.task_queue.forget_task(task);
        }
        send_tasks(id, "task delete", deleted.map(|()| Vec::new()), &tx).await
    }

    pub(super) async fn handle_task_set_enabled(
        self: Arc<Self>,
        id: String,
        task: TaskId,
        enabled: bool,
        tx: mpsc::Sender<Event>,
    ) -> Result<(), DispatchError> {
        let record = self.memory.tasks.set_enabled(task, enabled).await;
        send_tasks(id, "task set_enabled", record.map(|r| vec![r]), &tx).await
    }
}

/// Send each record as an `Event::Task` then `Done`, or the store's failure as `Error`.
async fn send_tasks(
    id: String,
    op: &str,
    records: MemoryResult<Vec<TaskRecord>>,
    tx: &mpsc::Sender<Event>,
) -> Result<(), DispatchError> {
    match records {
        Ok(records) => {
            for TaskRecord { id: task_id, spec } in records {
                let _ = tx
                    .send(Event::Task {
                        id: id.clone(),
                        task_id,
                        spec,
                    })
                    .await;
            }
            let _ = tx.send(Event::Done { id }).await;
            Ok(())
        }
        Err(e) => {
            send_error(tx, id, format!("{op} failed: {e}")).await;
            Err(e.into())
        }
    }
}
