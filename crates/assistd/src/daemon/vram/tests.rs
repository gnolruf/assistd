use std::collections::HashMap;

use super::*;

const HEADROOM_MB: u64 = 500;

fn snapshot(free_mb: u64, model_mb: u64, foreign_mb: u64) -> VramSnapshot {
    VramSnapshot {
        free_mb,
        model_mb,
        foreign_mb,
    }
}

/// `Learned` after an `Active` poll with the desktop holding 600 MiB.
fn loaded() -> Learned {
    let mut learned = Learned::default();
    learned.observe(
        snapshot(1200, 13000, 600),
        PresenceState::Active,
        HEADROOM_MB,
    );
    learned
}

#[test]
fn only_another_process_growing_into_the_headroom_is_contention() {
    let cases = [
        (
            "game claims VRAM on a full card",
            snapshot(150, 13000, 900),
            true,
        ),
        (
            "our own Whisper or image turn fills the card",
            snapshot(150, 13600, 600),
            false,
        ),
        (
            "game starts with room to spare",
            snapshot(800, 13000, 900),
            false,
        ),
        (
            "desktop drifts on a full card",
            snapshot(150, 13000, 650),
            false,
        ),
    ];
    for (label, poll, contended) in cases {
        let mut learned = loaded();
        learned.observe(poll, PresenceState::Active, HEADROOM_MB);
        assert_eq!(learned.contended, contended, "{label}");
    }
}

#[test]
fn the_foreign_baseline_follows_the_least_seen_and_resets_while_down() {
    let mut learned = loaded();
    learned.observe(
        snapshot(1500, 13000, 300),
        PresenceState::Active,
        HEADROOM_MB,
    );
    learned.observe(
        snapshot(150, 13000, 450),
        PresenceState::Active,
        HEADROOM_MB,
    );
    assert!(
        learned.contended,
        "growth is measured from the least seen, after the desktop shrank"
    );

    learned.observe(
        snapshot(14000, 0, 450),
        PresenceState::Sleeping,
        HEADROOM_MB,
    );
    assert_eq!(learned.foreign_baseline_mb, None);
    assert!(!learned.contended);

    learned.observe(
        snapshot(400, 13000, 2000),
        PresenceState::Active,
        HEADROOM_MB,
    );
    assert!(
        !learned.contended,
        "a process already running when the model came back is the new baseline"
    );
}

#[test]
fn a_wake_needs_the_model_footprint_left_to_load_plus_headroom() {
    let learned = loaded();
    let cases = [
        ("asleep with room", snapshot(13600, 0, 600), Ok(())),
        (
            "asleep beside a game",
            snapshot(9000, 0, 5000),
            Err(VramShortfall {
                free_mb: 9000,
                needed_mb: 13500,
            }),
        ),
        (
            "drowsy server already holds part",
            snapshot(13200, 400, 600),
            Ok(()),
        ),
    ];
    for (label, poll, expected) in cases {
        assert_eq!(learned.room_to_wake(poll, HEADROOM_MB), expected, "{label}");
    }
    assert_eq!(
        Learned::default().room_to_wake(snapshot(10, 0, 0), HEADROOM_MB),
        Ok(()),
        "a model never measured is let through to try"
    );
}

#[test]
fn descends_from_follows_the_parent_chain_to_the_router() {
    let parents: HashMap<u32, u32> = [
        (100, 1),
        (200, 100),
        (300, 200),
        (400, 1),
        (500, 501),
        (501, 500),
    ]
    .into_iter()
    .collect();
    let parent_of = |pid: u32| parents.get(&pid).copied();
    let cases = [
        ("router itself", 100, true),
        ("grandchild", 300, true),
        ("sibling of router", 400, false),
        ("parent cycle", 500, false),
    ];
    for (label, pid, expected) in cases {
        assert_eq!(descends_from(pid, 100, parent_of), expected, "{label}");
    }
}
