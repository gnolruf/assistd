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
