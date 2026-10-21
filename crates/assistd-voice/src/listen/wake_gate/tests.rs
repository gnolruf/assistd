use assistd_config::UnitInterval;

use super::*;

const FOLLOW_UP: Duration = Duration::from_secs(8);

fn gate(follow_up: Option<Duration>) -> WakeGate {
    WakeGate::new(
        WakePhrase::builtin(UnitInterval::new(0.8).unwrap()),
        Duration::from_secs(4),
        follow_up,
    )
}

#[tokio::test(start_paused = true)]
async fn follow_up_admits_one_utterance_inside_the_window_only() {
    let mut gate = gate(Some(FOLLOW_UP));
    let ended = Instant::now();
    gate.open_follow_up(ended);
    assert_eq!(
        gate.admit("and the gold one?", ended + Duration::from_secs(3)),
        Admission::Command("and the gold one?".into())
    );
    assert_eq!(
        gate.admit("what about silver", ended + Duration::from_secs(5)),
        Admission::Dropped,
        "one utterance closes the window"
    );

    gate.open_follow_up(ended);
    assert_eq!(
        gate.admit(
            "talking to someone else",
            ended + FOLLOW_UP + Duration::from_millis(1)
        ),
        Admission::Dropped,
        "speech that began after the window is not a command"
    );

    gate.open_follow_up(ended);
    assert_eq!(
        gate.admit("said during the reply", ended - Duration::from_millis(1)),
        Admission::Dropped,
        "speech that began before the reply ended is not a reply to it"
    );

    gate.open_follow_up(ended);
    assert_eq!(
        gate.admit("Hey computer, open firefox", ended),
        Admission::WakePhraseAndCommand("open firefox".into()),
        "the phrase is stripped inside a window too"
    );
}

#[tokio::test(start_paused = true)]
async fn disabled_follow_up_opens_no_window() {
    let mut gate = gate(None);
    let ended = Instant::now();
    gate.open_follow_up(ended);
    assert_eq!(gate.admit("and the gold one?", ended), Admission::Dropped);
}
