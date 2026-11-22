//! The form one task trigger is written in: its kind, then the fields that kind takes.

use std::path::PathBuf;

use assistd_ipc::{
    DEFAULT_EVENT_COOLDOWN_SECS, DEFAULT_PATH_DEBOUNCE_SECS, FinishOutcome, MissedFire, TaskEvent,
    TaskId, TaskTrigger,
};
use chrono::{Local, NaiveDateTime, TimeDelta, Timelike};

use crate::selection::cycled;
use crate::settings::{FieldValue, Form, FormFields};
use crate::text_editor::{EditorShape, TextEditor};

/// How an `At` time is typed and shown.
pub(crate) const AT_FORMAT: &str = "%Y-%m-%d %H:%M";
const DEFAULT_CRON: &str = "0 9 * * *";

/// The form over one trigger's fields.
pub(crate) type TriggerForm = Form<TriggerFields>;

/// What starts a run, as the form's first field offers it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TriggerKind {
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

    pub(crate) fn label(self) -> &'static str {
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
pub(crate) enum FormField {
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
    pub(crate) fn label(self) -> &'static str {
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
pub(crate) struct UpstreamChoice {
    pub id: TaskId,
    pub name: String,
}

/// A trigger being written: every kind's fields are kept, so switching kinds loses nothing.
pub(crate) struct TriggerFields {
    /// The trigger row this replaces; `None` adds one.
    pub replaces: Option<usize>,
    kind: TriggerKind,
    at: TextEditor,
    cron: TextEditor,
    missed: MissedFire,
    cooldown: TextEditor,
    path: TextEditor,
    debounce: TextEditor,
    upstream: Option<TaskId>,
    outcome: FinishOutcome,
    /// The tasks the trigger may wait on.
    upstreams: Vec<UpstreamChoice>,
}

impl TriggerFields {
    /// A new trigger, on a schedule, that may wait on one of `upstreams`.
    pub(super) fn new(upstreams: Vec<UpstreamChoice>) -> Self {
        let next_hour = Local::now().naive_local() + TimeDelta::hours(1);
        let at = next_hour
            .with_minute(0)
            .unwrap_or(next_hour)
            .format(AT_FORMAT)
            .to_string();
        Self {
            replaces: None,
            kind: TriggerKind::Schedule,
            at: line(at),
            cron: line(DEFAULT_CRON.to_string()),
            missed: MissedFire::default(),
            cooldown: line(DEFAULT_EVENT_COOLDOWN_SECS.to_string()),
            path: line(String::new()),
            debounce: line(DEFAULT_PATH_DEBOUNCE_SECS.to_string()),
            upstream: None,
            outcome: FinishOutcome::default(),
            upstreams,
        }
    }

    /// `trigger`, row `index` of the editor's triggers.
    pub(super) fn editing(
        index: usize,
        trigger: &TaskTrigger,
        upstreams: Vec<UpstreamChoice>,
    ) -> Self {
        let mut fields = Self {
            replaces: Some(index),
            ..Self::new(upstreams)
        };
        fields.load(trigger);
        fields
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

    fn event(&self, event: TaskEvent) -> Result<TaskTrigger, String> {
        Ok(TaskTrigger::Event {
            event,
            cooldown_secs: seconds(&self.cooldown, "cooldown")?,
        })
    }
}

impl FormFields for TriggerFields {
    type Field = FormField;
    type Output = TaskTrigger;

    /// The kind, then the fields the kind takes.
    fn fields(&self) -> Vec<FormField> {
        std::iter::once(FormField::Kind)
            .chain(self.kind.fields().iter().copied())
            .collect()
    }

    fn label(field: FormField) -> &'static str {
        field.label()
    }

    fn value(&self, field: FormField) -> FieldValue<'_> {
        match field {
            FormField::Kind => FieldValue::Choice(self.kind.label()),
            FormField::At => FieldValue::Text(&self.at),
            FormField::Cron => FieldValue::Text(&self.cron),
            FormField::Missed => FieldValue::Choice(match self.missed {
                MissedFire::RunOnce => "run once on return",
                MissedFire::Skip => "skip",
            }),
            FormField::Cooldown => FieldValue::Text(&self.cooldown),
            FormField::Path => FieldValue::Text(&self.path),
            FormField::Debounce => FieldValue::Text(&self.debounce),
            FormField::Upstream => FieldValue::Choice(
                self.upstreams
                    .iter()
                    .find(|task| Some(task.id) == self.upstream)
                    .map_or("(none)", |task| &task.name),
            ),
            FormField::Outcome => FieldValue::Choice(match self.outcome {
                FinishOutcome::Succeeded => "succeeds",
                FinishOutcome::Failed => "fails",
                FinishOutcome::Either => "succeeds or fails",
            }),
        }
    }

    fn text_mut(&mut self, field: FormField) -> Option<&mut TextEditor> {
        match field {
            FormField::At => Some(&mut self.at),
            FormField::Cron => Some(&mut self.cron),
            FormField::Cooldown => Some(&mut self.cooldown),
            FormField::Path => Some(&mut self.path),
            FormField::Debounce => Some(&mut self.debounce),
            FormField::Kind | FormField::Missed | FormField::Upstream | FormField::Outcome => None,
        }
    }

    fn cycle(&mut self, field: FormField, forward: bool) {
        match field {
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
                let ids: Vec<TaskId> = self.upstreams.iter().map(|task| task.id).collect();
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

    fn output(&self) -> Result<TaskTrigger, String> {
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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use assistd_ipc::{FinishOutcome, MissedFire, TaskEvent, TaskId, TaskTrigger};
    use chrono::NaiveDate;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    use super::{TriggerFields, UpstreamChoice};
    use crate::settings::{Form, FormAction};

    fn enter() -> KeyEvent {
        KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)
    }

    #[test]
    fn every_trigger_kind_survives_the_form_unchanged() {
        let at = NaiveDate::from_ymd_opt(2027, 1, 2)
            .and_then(|day| day.and_hms_opt(9, 30, 0))
            .expect("valid time");
        let triggers = [
            TaskTrigger::Manual,
            TaskTrigger::At { at },
            TaskTrigger::Schedule {
                cron: "*/5 * * * 1-5".into(),
                missed: MissedFire::Skip,
            },
            TaskTrigger::Event {
                event: TaskEvent::SessionUnlocked,
                cooldown_secs: 300,
            },
            TaskTrigger::Event {
                event: TaskEvent::PathChanged {
                    path: PathBuf::from("/srv/inbox"),
                    debounce_secs: 7,
                },
                cooldown_secs: 0,
            },
            TaskTrigger::Event {
                event: TaskEvent::TaskFinished {
                    task: TaskId(3),
                    outcome: FinishOutcome::Failed,
                },
                cooldown_secs: 60,
            },
        ];
        let upstreams = [UpstreamChoice {
            id: TaskId(3),
            name: "backup".into(),
        }];
        for trigger in triggers {
            let mut form = Form::new(TriggerFields::editing(0, &trigger, upstreams.to_vec()));
            match form.on_key(enter()) {
                FormAction::Apply(applied) => assert_eq!(applied, trigger),
                FormAction::Refused(reason) => panic!("{trigger:?} was refused: {reason}"),
                FormAction::Pending | FormAction::Cancel => panic!("{trigger:?} did not apply"),
            }
        }
    }
}
