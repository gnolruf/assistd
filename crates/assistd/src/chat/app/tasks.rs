//! `/tasks`: the tasks panel drawn in place of the transcript, its requests sent on
//! connections of their own, and the answers to the prompts paused task runs wait on.

use assistd_ipc::{Event, TaskQueueSnapshot, TaskRunId};
use assistd_tools::Approval;
use crossterm::event::KeyEvent;
use uuid::Uuid;

use super::{AnswerTo, App, WireStream};
use crate::chat::tasks::{PanelAction, TasksPanel, paused_prompts, task_confirm, waits_on};

impl App {
    pub(super) fn open_tasks_panel(&mut self) {
        self.tasks_panel = Some(TasksPanel::new(
            self.auto_mode_offered,
            self.task_queue.clone(),
        ));
        self.send_tasks_requests();
    }

    /// The panel when it is on screen; an open confirmation prompt hides it.
    pub(in crate::chat) fn shown_tasks_panel(&self) -> Option<&TasksPanel> {
        self.tasks_panel
            .as_ref()
            .filter(|_| self.confirmation.is_none())
    }

    pub(super) fn on_tasks_key(&mut self, ev: KeyEvent) {
        let Some(panel) = self.tasks_panel.as_mut() else {
            return;
        };
        let action = panel.on_key(ev);
        self.send_tasks_requests();
        match action {
            PanelAction::Stay => {}
            PanelAction::Close => self.tasks_panel = None,
            PanelAction::Resume(target) => {
                self.tasks_panel = None;
                self.switch_branch(target);
            }
            PanelAction::Answer(run, prompt) => self.open_task_run_confirmation(run, prompt),
        }
    }

    pub(super) fn on_tasks_event(&mut self, ev: Event) {
        if let Some(panel) = self.tasks_panel.as_mut() {
            panel.on_event(ev);
            self.send_tasks_requests();
        }
    }

    pub(super) fn fail_tasks_stream(&mut self, message: &str) {
        if let Some(panel) = self.tasks_panel.as_mut() {
            panel.on_connection_lost(message);
        }
    }

    /// Keep `snapshot` for the panel, saying when a run newly waits on a prompt and closing
    /// an open task-run prompt the run no longer waits on.
    pub(super) fn on_task_queue(&mut self, snapshot: TaskQueueSnapshot) {
        let newly_paused = paused_prompts(&snapshot)
            .find(|(run, prompt)| {
                self.task_queue
                    .as_ref()
                    .is_none_or(|known| !waits_on(known, *run, &prompt.confirm_id))
            })
            .map(|(run, _)| run);
        if let Some(run) = newly_paused {
            self.set_notice(&format!("task run {run} needs input · answer it in /tasks"));
        }
        let stale = self
            .confirmation
            .as_ref()
            .is_some_and(|prompt| match prompt.answer_to {
                AnswerTo::TaskRun(run) => !waits_on(&snapshot, run, &prompt.confirm_id),
                AnswerTo::Reply => false,
            });
        if stale {
            self.confirmation = None;
            self.set_notice("the task run no longer waits on that prompt");
        }
        if let Some(panel) = self.tasks_panel.as_mut() {
            panel.set_queue(snapshot.clone());
            self.send_tasks_requests();
        }
        self.task_queue = Some(snapshot);
    }

    /// Send `approval` for confirmation `confirm_id` of paused run `run`, through the panel
    /// when it is open so its outcome shows there.
    pub(super) fn answer_task_run(
        &mut self,
        run: TaskRunId,
        confirm_id: String,
        approval: Approval,
    ) {
        if let Some(panel) = self.tasks_panel.as_mut() {
            panel.answer_paused(run, confirm_id, approval);
            self.send_tasks_requests();
            return;
        }
        let req = task_confirm(Uuid::new_v4().to_string(), run, confirm_id, approval);
        self.spawn_one_shot(req, WireStream::Status, "task confirm");
    }

    fn send_tasks_requests(&mut self) {
        let requests = self
            .tasks_panel
            .as_mut()
            .map(TasksPanel::take_requests)
            .unwrap_or_default();
        for request in requests {
            self.spawn_one_shot(request, WireStream::Tasks, "tasks");
        }
    }
}
