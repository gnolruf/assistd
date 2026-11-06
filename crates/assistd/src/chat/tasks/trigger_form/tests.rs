use std::path::PathBuf;

use assistd_ipc::{FinishOutcome, MissedFire, TaskEvent, TaskId, TaskTrigger};
use chrono::NaiveDate;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::{FormAction, TriggerForm, UpstreamChoice};

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
        let mut form = TriggerForm::editing(0, &trigger);
        match form.on_key(enter(), &upstreams) {
            FormAction::Apply(applied) => assert_eq!(applied, trigger),
            FormAction::Pending | FormAction::Cancel => {
                panic!("{trigger:?} did not apply: {:?}", form.error());
            }
        }
    }
}
