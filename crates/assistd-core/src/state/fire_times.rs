//! When a task's time triggers next fire, which of them fires once a recorded
//! fire time has passed, and why `TaskSave` refuses a trigger.

use std::io;
use std::path::PathBuf;

use chrono::{DateTime, LocalResult, NaiveDateTime, TimeDelta, TimeZone, Utc};
use croner::Cron;
use croner::errors::CronError;
use croner::parser::{CronParser, Seconds, Year};
use thiserror::Error;

use assistd_ipc::{MissedFire, TaskId, TaskTrigger};

/// How long after a fire time it may be noticed and still count as on time, not missed.
const ON_TIME_GRACE: TimeDelta = TimeDelta::minutes(2);

/// Why `TaskSave` refused a trigger.
#[derive(Debug, Error)]
pub enum TriggerError {
    #[error("cron expression {cron:?} is invalid: {source}")]
    InvalidCron {
        cron: String,
        #[source]
        source: CronError,
    },

    #[error("trigger time {0} has already passed")]
    PastAt(NaiveDateTime),

    #[error("watched path {} is not absolute", .0.display())]
    RelativePath(PathBuf),

    #[error("watched path {} cannot be read: {source}", path.display())]
    MissingPath {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("trigger waits on task {0}, which does not exist")]
    UnknownUpstream(TaskId),

    #[error("waiting on task {0} would make this task wait on itself")]
    ChainCycle(TaskId),
}

/// Refuse a trigger in `triggers` with an invalid cron expression, or an `At` time not
/// after `now` unless `stored`, the task's saved triggers, already holds it.
pub(in crate::state) fn check_triggers<Tz: TimeZone>(
    triggers: &[TaskTrigger],
    stored: &[TaskTrigger],
    now: &DateTime<Tz>,
) -> Result<(), TriggerError> {
    for trigger in triggers {
        match trigger {
            TaskTrigger::Manual | TaskTrigger::Event { .. } => {}
            TaskTrigger::At { at } => {
                if !stored.contains(trigger) && at_instant(at, &now.timezone()) <= *now {
                    return Err(TriggerError::PastAt(*at));
                }
            }
            TaskTrigger::Schedule { cron, .. } => {
                parse_cron(cron).map_err(|source| TriggerError::InvalidCron {
                    cron: cron.clone(),
                    source,
                })?;
            }
        }
    }
    Ok(())
}

/// The earliest time any of `triggers` fires after `after`, in `after`'s time zone.
pub(in crate::state) fn next_fire<Tz: TimeZone>(
    triggers: &[TaskTrigger],
    after: &DateTime<Tz>,
) -> Option<DateTime<Utc>> {
    triggers
        .iter()
        .filter_map(|trigger| occurrence(trigger, after, false))
        .min()
        .map(|at| at.with_timezone(&Utc))
}

/// The trigger that fires now that `due`, the recorded next fire time, has passed at
/// `now`: one whose fire time is no more than [`ON_TIME_GRACE`] old, else one that
/// missed a fire time since `due` and runs missed fires once. `None` skips them all.
pub(in crate::state) fn fired_trigger<'a, Tz: TimeZone>(
    triggers: &'a [TaskTrigger],
    due: &DateTime<Tz>,
    now: &DateTime<Tz>,
) -> Option<&'a TaskTrigger> {
    let on_time_since = due.clone().max(now.clone() - ON_TIME_GRACE);
    let fires_between = |trigger: &TaskTrigger, since: &DateTime<Tz>| {
        occurrence(trigger, since, true).is_some_and(|at| at <= *now)
    };
    triggers
        .iter()
        .find(|trigger| fires_between(trigger, &on_time_since))
        .or_else(|| {
            triggers.iter().find(|trigger| {
                missed_fire_policy(trigger) == MissedFire::RunOnce && fires_between(trigger, due)
            })
        })
}

/// A five-field cron expression, without seconds or years.
fn parse_cron(cron: &str) -> Result<Cron, CronError> {
    CronParser::builder()
        .seconds(Seconds::Disallowed)
        .year(Year::Disallowed)
        .build()
        .parse(cron)
}

/// `trigger`'s first fire time after `from`, or at it too when `inclusive`.
fn occurrence<Tz: TimeZone>(
    trigger: &TaskTrigger,
    from: &DateTime<Tz>,
    inclusive: bool,
) -> Option<DateTime<Tz>> {
    match trigger {
        TaskTrigger::Manual | TaskTrigger::Event { .. } => None,
        TaskTrigger::At { at } => {
            let at = at_instant(at, &from.timezone());
            (at > *from || (inclusive && at == *from)).then_some(at)
        }
        TaskTrigger::Schedule { cron, .. } => parse_cron(cron)
            .ok()?
            .find_next_occurrence(from, inclusive)
            .ok(),
    }
}

fn missed_fire_policy(trigger: &TaskTrigger) -> MissedFire {
    match trigger {
        TaskTrigger::Schedule { missed, .. } => *missed,
        TaskTrigger::Manual | TaskTrigger::At { .. } | TaskTrigger::Event { .. } => {
            MissedFire::RunOnce
        }
    }
}

/// The instant wall-clock time `at` names in `zone`: the earlier of a repeated time, and
/// for a time a DST gap skips, the time read at the offset in force just before the gap.
fn at_instant<Tz: TimeZone>(at: &NaiveDateTime, zone: &Tz) -> DateTime<Tz> {
    match zone.from_local_datetime(at) {
        LocalResult::Single(instant) | LocalResult::Ambiguous(instant, _) => instant,
        LocalResult::None => {
            let before_gap = TimeDelta::hours(1);
            zone.from_local_datetime(&(*at - before_gap))
                .earliest()
                .map_or_else(
                    || zone.from_utc_datetime(at),
                    |instant| instant + before_gap,
                )
        }
    }
}

#[cfg(test)]
mod tests;
