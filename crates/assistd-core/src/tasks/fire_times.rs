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
pub(super) fn check_triggers<Tz: TimeZone>(
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
pub(super) fn next_fire<Tz: TimeZone>(
    triggers: &[TaskTrigger],
    after: &DateTime<Tz>,
) -> Option<DateTime<Utc>> {
    triggers
        .iter()
        .filter_map(|trigger| occurrence(trigger, after, false))
        .min()
        .map(|at| at.with_timezone(&Utc))
}

/// The first `count` times `trigger` fires after `after`, soonest first.
pub(super) fn upcoming_fires<Tz: TimeZone>(
    trigger: &TaskTrigger,
    after: &DateTime<Tz>,
    count: usize,
) -> Vec<DateTime<Utc>> {
    std::iter::successors(occurrence(trigger, after, false), |previous| {
        occurrence(trigger, previous, false)
    })
    .take(count)
    .map(|at| at.with_timezone(&Utc))
    .collect()
}

/// The trigger that fires now that `due`, the recorded next fire time, has passed at
/// `now`: one whose fire time is no more than [`ON_TIME_GRACE`] old, else one that
/// missed a fire time since `due` and runs missed fires once. `None` skips them all.
pub(super) fn fired_trigger<'a, Tz: TimeZone>(
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
mod tests {
    use chrono::{NaiveDate, TimeZone};
    use chrono_tz::America::New_York;
    use chrono_tz::Tz;

    use super::*;

    fn new_york(year: i32, month: u32, day: u32, hour: u32, minute: u32) -> DateTime<Tz> {
        New_York
            .with_ymd_and_hms(year, month, day, hour, minute, 0)
            .earliest()
            .unwrap()
    }

    fn schedule(cron: &str, missed: MissedFire) -> TaskTrigger {
        TaskTrigger::Schedule {
            cron: cron.into(),
            missed,
        }
    }

    fn utc(at: DateTime<Tz>) -> Option<DateTime<Utc>> {
        Some(at.with_timezone(&Utc))
    }

    #[test]
    fn next_fire_keeps_local_time_across_dst_changes() {
        let daily = [schedule("30 9 * * *", MissedFire::RunOnce)];
        let before_spring = new_york(2026, 3, 7, 12, 0);
        assert_eq!(
            next_fire(&daily, &before_spring),
            utc(new_york(2026, 3, 8, 9, 30))
        );
        let before_fall = new_york(2026, 10, 31, 12, 0);
        assert_eq!(
            next_fire(&daily, &before_fall),
            utc(new_york(2026, 11, 1, 9, 30))
        );

        let in_repeated_hour = [schedule("30 1 * * *", MissedFire::RunOnce)];
        let first_pass = next_fire(&in_repeated_hour, &before_fall).unwrap();
        assert_eq!(Some(first_pass), utc(new_york(2026, 11, 1, 1, 30)));
        let after_first = first_pass.with_timezone(&New_York);
        assert_eq!(
            next_fire(&in_repeated_hour, &after_first),
            utc(new_york(2026, 11, 2, 1, 30))
        );

        let skipped_by_gap = NaiveDate::from_ymd_opt(2026, 3, 8)
            .and_then(|day| day.and_hms_opt(2, 30, 0))
            .unwrap();
        let at = [TaskTrigger::At { at: skipped_by_gap }];
        assert_eq!(
            next_fire(&at, &before_spring),
            utc(new_york(2026, 3, 8, 3, 30))
        );
    }

    #[test]
    fn missed_fires_after_a_gap_run_once_or_are_skipped() {
        let due = new_york(2026, 11, 4, 9, 0);
        let back = new_york(2026, 11, 4, 11, 10);
        let run_once = [schedule("*/30 * * * *", MissedFire::RunOnce)];
        let skip = [schedule("*/30 * * * *", MissedFire::Skip)];

        assert_eq!(fired_trigger(&run_once, &due, &back), Some(&run_once[0]));
        assert_eq!(fired_trigger(&skip, &due, &back), None);
        assert_eq!(
            next_fire(&run_once, &back),
            utc(new_york(2026, 11, 4, 11, 30))
        );

        let on_time = new_york(2026, 11, 4, 9, 1);
        assert_eq!(fired_trigger(&skip, &due, &on_time), Some(&skip[0]));
    }
}
