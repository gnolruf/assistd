//! The form one task trigger is written in: its kind, then the fields that kind takes.

use std::path::PathBuf;

use assistd_ipc::{
    DEFAULT_EVENT_COOLDOWN_SECS, DEFAULT_PATH_DEBOUNCE_SECS, FinishOutcome, MissedFire, TaskEvent,
    TaskId, TaskTrigger,
};
use chrono::{Local, NaiveDateTime, TimeDelta, Timelike};
use crossterm::event::{KeyCode, KeyEvent};

use crate::chat::settings::{EditorShape, TextEditor};

/// How an `At` time is typed and shown.
pub(in crate::chat) const AT_FORMAT: &str = "%Y-%m-%d %H:%M";
const DEFAULT_CRON: &str = "0 9 * * *";

/// What starts a run, as the form's first field offers it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::chat) enum TriggerKind {
    Manual,
    At,
    Schedule,
    DaemonStarted,
    SessionUnlocked,
    PathChanged,
    TaskFinished,
}

impl TriggerKind {
    const ALL: [Self; 7] = [
        Self::Manual,
        Self::At,
        Self::Schedule,
        Self::DaemonStarted,
        Self::SessionUnlocked,
        Self::PathChanged,
        Self::TaskFinished,
    ];

    pub(in crate::chat) fn label(self) -> &'static str {
        match self {
            Self::Manual => "on request",
            Self::At => "once at a time",
            Self::Schedule => "on a schedule",
            Self::DaemonStarted => "when the daemon starts",
            Self::SessionUnlocked => "when the session unlocks",
            Self::PathChanged => "when a path changes",
            Self::TaskFinished => "after another task",
        }
    }

    /// The fields after the kind this kind takes, in form order.
    fn fields(self) -> &'static [FormField] {
        match self {
            Self::Manual => &[],
            Self::At => &[FormField::At],
            Self::Schedule => &[FormField::Cron, FormField::Missed],
            Self::DaemonStarted | Self::SessionUnlocked => &[FormField::Cooldown],
            Self::PathChanged => &[FormField::Path, FormField::Debounce, FormField::Cooldown],
            Self::TaskFinished => &[FormField::Upstream, FormField::Outcome, FormField::Cooldown],
        }
    }
}

/// One row of the form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::chat) enum FormField {
    Kind,
    At,
    Cron,
    Missed,
    Cooldown,
    Path,
    Debounce,
    Upstream,
    Outcome,
}

impl FormField {
    pub(in crate::chat) fn label(self) -> &'static str {
        match self {
            Self::Kind => "Starts",
            Self::At => "Time",
            Self::Cron => "Cron",
            Self::Missed => "Missed fires",
            Self::Cooldown => "Cooldown (s)",
            Self::Path => "Path",
            Self::Debounce => "Debounce (s)",
            Self::Upstream => "Task",
            Self::Outcome => "When it",
        }
    }
}

/// A task another task's trigger may wait on.
#[derive(Debug, Clone)]
pub(in crate::chat) struct UpstreamChoice {
    pub id: TaskId,
    pub name: String,
}

/// What a key in the form asks of the editor holding it.
pub(super) enum FormAction {
    Pending,
    Apply(TaskTrigger),
    Cancel,
}

/// A trigger being written: every kind's fields are kept, so switching kinds loses nothing.
pub(in crate::chat) struct TriggerForm {
    /// The trigger row this replaces; `None` adds one.
    pub replaces: Option<usize>,
    kind: TriggerKind,
    focus: FormField,
    at: TextEditor,
    cron: TextEditor,
    missed: MissedFire,
    cooldown: TextEditor,
    path: TextEditor,
    debounce: TextEditor,
    upstream: Option<TaskId>,
    outcome: FinishOutcome,
    /// Why the last Enter could not make a trigger.
    error: Option<String>,
}

impl TriggerForm {
    /// A form for a new trigger, on a schedule.
    pub(super) fn new() -> Self {
        let next_hour = Local::now().naive_local() + TimeDelta::hours(1);
        let at = next_hour
            .with_minute(0)
            .unwrap_or(next_hour)
            .format(AT_FORMAT)
            .to_string();
        Self {
            replaces: None,
            kind: TriggerKind::Schedule,
            focus: FormField::Kind,
            at: line(at),
            cron: line(DEFAULT_CRON.to_string()),
            missed: MissedFire::default(),
            cooldown: line(DEFAULT_EVENT_COOLDOWN_SECS.to_string()),
            path: line(String::new()),
            debounce: line(DEFAULT_PATH_DEBOUNCE_SECS.to_string()),
            upstream: None,
            outcome: FinishOutcome::default(),
            error: None,
        }
    }

    /// A form holding `trigger`, row `index` of the editor's triggers.
    pub(super) fn editing(index: usize, trigger: &TaskTrigger) -> Self {
        let mut form = Self {
            replaces: Some(index),
            ..Self::new()
        };
        form.load(trigger);
        form
    }

    fn load(&mut self, trigger: &TaskTrigger) {
        match trigger {
            TaskTrigger::Manual => self.kind = TriggerKind::Manual,
            TaskTrigger::At { at } => {
                self.kind = TriggerKind::At;
                self.at = line(at.format(AT_FORMAT).to_string());
            }
            TaskTrigger::Schedule { cron, missed } => {
                self.kind = TriggerKind::Schedule;
                self.cron = line(cron.clone());
                self.missed = *missed;
            }
            TaskTrigger::Event {
                event,
                cooldown_secs,
            } => {
                self.cooldown = line(cooldown_secs.to_string());
                self.load_event(event);
            }
        }
    }

    fn load_event(&mut self, event: &TaskEvent) {
        match event {
            TaskEvent::DaemonStarted => self.kind = TriggerKind::DaemonStarted,
            TaskEvent::SessionUnlocked => self.kind = TriggerKind::SessionUnlocked,
            TaskEvent::PathChanged {
                path,
                debounce_secs,
            } => {
                self.kind = TriggerKind::PathChanged;
                self.path = line(path.display().to_string());
                self.debounce = line(debounce_secs.to_string());
            }
            TaskEvent::TaskFinished { task, outcome } => {
                self.kind = TriggerKind::TaskFinished;
                self.upstream = Some(*task);
                self.outcome = *outcome;
            }
        }
    }

    pub(in crate::chat) fn focus(&self) -> FormField {
        self.focus
    }

    pub(in crate::chat) fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    /// The kind, then the fields the kind takes.
    pub(in crate::chat) fn fields(&self) -> Vec<FormField> {
        std::iter::once(FormField::Kind)
            .chain(self.kind.fields().iter().copied())
            .collect()
    }

    /// The text editor behind `field`; `None` for a field picked from choices.
    pub(in crate::chat) fn text_field(&self, field: FormField) -> Option<&TextEditor> {
        match field {
            FormField::At => Some(&self.at),
            FormField::Cron => Some(&self.cron),
            FormField::Cooldown => Some(&self.cooldown),
            FormField::Path => Some(&self.path),
            FormField::Debounce => Some(&self.debounce),
            FormField::Kind | FormField::Missed | FormField::Upstream | FormField::Outcome => None,
        }
    }

    fn text_field_mut(&mut self, field: FormField) -> Option<&mut TextEditor> {
        match field {
            FormField::At => Some(&mut self.at),
            FormField::Cron => Some(&mut self.cron),
            FormField::Cooldown => Some(&mut self.cooldown),
            FormField::Path => Some(&mut self.path),
            FormField::Debounce => Some(&mut self.debounce),
            FormField::Kind | FormField::Missed | FormField::Upstream | FormField::Outcome => None,
        }
    }

    /// The value a choice field shows.
    pub(in crate::chat) fn choice_text(
        &self,
        field: FormField,
        tasks: &[UpstreamChoice],
    ) -> String {
        match field {
            FormField::Kind => self.kind.label().to_string(),
            FormField::Missed => match self.missed {
                MissedFire::RunOnce => "run once on return".into(),
                MissedFire::Skip => "skip".into(),
            },
            FormField::Upstream => self
                .upstream
                .and_then(|id| tasks.iter().find(|task| task.id == id))
                .map_or_else(|| "(none)".to_string(), |task| task.name.clone()),
            FormField::Outcome => match self.outcome {
                FinishOutcome::Succeeded => "succeeds".into(),
                FinishOutcome::Failed => "fails".into(),
                FinishOutcome::Either => "succeeds or fails".into(),
            },
            FormField::At
            | FormField::Cron
            | FormField::Cooldown
            | FormField::Path
            | FormField::Debounce => String::new(),
        }
    }

    /// ↑/↓ and Tab move between fields, ←/→ change a choice, typing edits a text field,
    /// Enter makes the trigger and Esc drops the form. `tasks` are the tasks it may wait on.
    pub(super) fn on_key(&mut self, ev: KeyEvent, tasks: &[UpstreamChoice]) -> FormAction {
        match ev.code {
            KeyCode::Up | KeyCode::BackTab => self.move_focus(false),
            KeyCode::Down | KeyCode::Tab => self.move_focus(true),
            KeyCode::Left if self.text_field(self.focus).is_none() => self.cycle(false, tasks),
            KeyCode::Right if self.text_field(self.focus).is_none() => self.cycle(true, tasks),
            KeyCode::Enter => return self.apply(),
            KeyCode::Esc => return FormAction::Cancel,
            _ => {
                if let Some(editor) = self.text_field_mut(self.focus) {
                    editor.on_key(ev);
                }
            }
        }
        FormAction::Pending
    }

    fn move_focus(&mut self, forward: bool) {
        let fields = self.fields();
        let index = fields
            .iter()
            .position(|field| *field == self.focus)
            .unwrap_or(0);
        let next = if forward {
            (index + 1) % fields.len()
        } else {
            (index + fields.len() - 1) % fields.len()
        };
        self.focus = fields[next];
    }

    fn cycle(&mut self, forward: bool, tasks: &[UpstreamChoice]) {
        match self.focus {
            FormField::Kind => self.kind = cycled(&TriggerKind::ALL, self.kind, forward),
            FormField::Missed => {
                self.missed = cycled(
                    &[MissedFire::RunOnce, MissedFire::Skip],
                    self.missed,
                    forward,
                );
            }
            FormField::Outcome => {
                let outcomes = [
                    FinishOutcome::Either,
                    FinishOutcome::Succeeded,
                    FinishOutcome::Failed,
                ];
                self.outcome = cycled(&outcomes, self.outcome, forward);
            }
            FormField::Upstream => {
                let ids: Vec<TaskId> = tasks.iter().map(|task| task.id).collect();
                self.upstream = match self.upstream {
                    Some(id) if ids.contains(&id) => Some(cycled(&ids, id, forward)),
                    _ => ids.first().copied(),
                };
            }
            FormField::At
            | FormField::Cron
            | FormField::Cooldown
            | FormField::Path
            | FormField::Debounce => {}
        }
    }

    fn apply(&mut self) -> FormAction {
        match self.trigger() {
            Ok(trigger) => FormAction::Apply(trigger),
            Err(message) => {
                self.error = Some(message);
                FormAction::Pending
            }
        }
    }

    /// The trigger the fields describe, or why they describe none.
    fn trigger(&self) -> Result<TaskTrigger, String> {
        Ok(match self.kind {
            TriggerKind::Manual => TaskTrigger::Manual,
            TriggerKind::At => TaskTrigger::At {
                at: NaiveDateTime::parse_from_str(self.at.text().trim(), AT_FORMAT)
                    .map_err(|_| "time must be written YYYY-MM-DD HH:MM".to_string())?,
            },
            TriggerKind::Schedule => TaskTrigger::Schedule {
                cron: self.cron.text().trim().to_string(),
                missed: self.missed,
            },
            TriggerKind::DaemonStarted => self.event(TaskEvent::DaemonStarted)?,
            TriggerKind::SessionUnlocked => self.event(TaskEvent::SessionUnlocked)?,
            TriggerKind::PathChanged => {
                let path = self.path.text().trim();
                if path.is_empty() {
                    return Err("path is empty".into());
                }
                self.event(TaskEvent::PathChanged {
                    path: PathBuf::from(path),
                    debounce_secs: seconds(&self.debounce, "debounce")?,
                })?
            }
            TriggerKind::TaskFinished => self.event(TaskEvent::TaskFinished {
                task: self.upstream.ok_or("pick the task to wait on")?,
                outcome: self.outcome,
            })?,
        })
    }

    fn event(&self, event: TaskEvent) -> Result<TaskTrigger, String> {
        Ok(TaskTrigger::Event {
            event,
            cooldown_secs: seconds(&self.cooldown, "cooldown")?,
        })
    }
}

fn line(text: String) -> TextEditor {
    TextEditor::new(text, EditorShape::Line)
}

/// The whole seconds typed in `editor`, the field named `name`.
fn seconds(editor: &TextEditor, name: &str) -> Result<u32, String> {
    editor
        .text()
        .trim()
        .parse()
        .map_err(|_| format!("{name} must be a whole number of seconds"))
}

/// The entry after or before `current` in `all`, wrapping; the first when it is not there.
fn cycled<T: Copy + PartialEq>(all: &[T], current: T, forward: bool) -> T {
    let index = all.iter().position(|item| *item == current).unwrap_or(0);
    let next = if forward {
        (index + 1) % all.len()
    } else {
        (index + all.len() - 1) % all.len()
    };
    all[next]
}

#[cfg(test)]
mod tests;
