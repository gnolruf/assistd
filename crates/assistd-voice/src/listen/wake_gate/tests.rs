use super::*;

const FOLLOW_UP: Duration = Duration::from_secs(8);

#[tokio::test(start_paused = true)]
async fn follow_up_admits_one_utterance_inside_the_window_only() {
    let mut gate = WakeGate::new(Duration::from_secs(4), Some(FOLLOW_UP));
    let ended = Instant::now();
    gate.open_follow_up(ended);
    assert_eq!(
        gate.admit(false, ended + Duration::from_secs(3)),
        Admission::Command
    );
    assert_eq!(
        gate.admit(false, ended + Duration::from_secs(5)),
        Admission::Dropped,
        "one utterance closes the window"
    );

    gate.open_follow_up(ended);
    assert_eq!(
        gate.admit(false, ended + FOLLOW_UP + Duration::from_millis(1)),
        Admission::Dropped,
        "speech that began after the window is not a command"
    );

    gate.open_follow_up(ended);
    assert_eq!(
        gate.admit(false, ended - Duration::from_millis(1)),
        Admission::Dropped,
        "speech that began before the reply ended is not a reply to it"
    );

    gate.open_follow_up(ended);
    assert_eq!(
        gate.admit(true, ended),
        Admission::WakePhrase,
        "a spotted phrase inside a window still has its phrase cut off"
    );
}
