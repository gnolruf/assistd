use super::*;

const CROWDED: VramSnapshot = VramSnapshot {
    free_mb: 150,
    model_mb: 13000,
    foreign_mb: 900,
};

fn poll(state: PresenceState, contended: bool, room_to_wake: bool) -> Poll {
    Poll {
        state,
        snapshot: CROWDED,
        contended,
        room_to_wake,
    }
}

#[test]
fn decide_sleeps_on_contention_and_auto_wakes_only_its_own_sleep_once_it_fits() {
    let mut cfg = assistd_core::Config::default().sleep;
    cfg.gpu_auto_wake = true;
    let cases = [
        (
            "drowsy, crowded",
            poll(PresenceState::Drowsy, true, false),
            SleepCause::None,
            Action::Sleep(CROWDED),
        ),
        (
            "active, not crowded",
            poll(PresenceState::Active, false, true),
            SleepCause::None,
            Action::None,
        ),
        (
            "own sleep, still no room",
            poll(PresenceState::Sleeping, false, false),
            SleepCause::Contention,
            Action::None,
        ),
        (
            "own sleep, fits again",
            poll(PresenceState::Sleeping, false, true),
            SleepCause::Contention,
            Action::Wake,
        ),
        (
            "user sleep, fits",
            poll(PresenceState::Sleeping, false, true),
            SleepCause::None,
            Action::None,
        ),
    ];
    for (label, poll, cause, expected) in cases {
        assert_eq!(decide(poll, &cfg, cause), expected, "{label}");
    }
}
