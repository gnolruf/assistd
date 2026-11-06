//! `/tasks`: the tasks panel drawn in place of the transcript, its requests sent on
//! connections of their own.

use assistd_ipc::Event;
use crossterm::event::KeyEvent;

use super::{App, WireStream};
use crate::chat::tasks::TasksPanel;

impl App {
    pub(super) fn open_tasks_panel(&mut self) {
        self.tasks_panel = Some(TasksPanel::new(self.auto_mode_offered));
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
        let closed = panel.on_key(ev).is_break();
        self.send_tasks_requests();
        if closed {
            self.tasks_panel = None;
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
