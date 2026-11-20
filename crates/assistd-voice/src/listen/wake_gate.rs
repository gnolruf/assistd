//! The state machine that decides which utterances are addressed to
//! assistd: those holding the wake phrase, and those inside a command window.

use std::time::Duration;

use assistd_ipc::ListenPhase;
use tokio::time::Instant;

/// What one utterance amounts to once gated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Admission {
    /// Not addressed to assistd; never transcribed.
    Dropped,
    /// The wake phrase was spotted; any speech after it is the command.
    WakePhrase,
    /// An utterance inside a command window, taken whole as the command.
    Command,
    /// An utterance inside the answer window: what the user said to a question asked aloud.
    Answer,
}

/// Admits an utterance when the wake phrase was spotted in it, or when it
/// began inside a command window: `command_timeout` after a bare phrase,
/// or `follow_up` after a reply ended. The first utterance that began
/// inside or after the window closes it. While a question asked aloud awaits
/// its answer, every utterance that began inside the answer window is one.
#[derive(Debug)]
pub struct WakeGate {
    command_timeout: Duration,
    /// `None` when follow-ups are disabled.
    follow_up: Option<Duration>,
    window: Option<CommandWindow>,
    answer_window: Option<CommandWindow>,
}

/// Speech that began between `opens` and `closes` is a command.
#[derive(Debug, Clone, Copy)]
struct CommandWindow {
    opens: Instant,
    closes: Instant,
}

impl CommandWindow {
    fn contains(self, started: Instant) -> bool {
        self.opens <= started && started <= self.closes
    }
}

impl WakeGate {
    pub fn new(command_timeout: Duration, follow_up: Option<Duration>) -> Self {
        Self {
            command_timeout,
            follow_up,
            window: None,
            answer_window: None,
        }
    }

    /// Take speech that begins within `wait` of `asked`, when a question finished being
    /// asked aloud, as its answer, until [`Self::settle_answer`].
    pub fn await_answer(&mut self, asked: Instant, wait: Duration) {
        self.answer_window = Some(CommandWindow {
            opens: asked,
            closes: asked + wait,
        });
    }

    /// Stop taking speech as an answer.
    pub fn settle_answer(&mut self) {
        self.answer_window = None;
    }

    /// Open the follow-up window for a reply that ended at `reply_ended`,
    /// unless an open window already closes later.
    pub fn open_follow_up(&mut self, reply_ended: Instant) {
        let Some(follow_up) = self.follow_up else {
            return;
        };
        let closes = reply_ended + follow_up;
        if self.window.is_none_or(|open| open.closes < closes) {
            self.window = Some(CommandWindow {
                opens: reply_ended,
                closes,
            });
        }
    }

    /// When the last of the open command and answer windows closes, if one is open.
    pub fn command_deadline(&self) -> Option<Instant> {
        self.windows().map(|window| window.closes).max()
    }

    /// `Capturing` while a command or answer window is open at `now`, else `Armed`.
    pub fn phase(&self, now: Instant) -> ListenPhase {
        if self.windows().any(|window| window.contains(now)) {
            ListenPhase::Capturing
        } else {
            ListenPhase::Armed
        }
    }

    fn windows(&self) -> impl Iterator<Item = CommandWindow> {
        self.window.into_iter().chain(self.answer_window)
    }

    /// Gate an utterance that began at `started`, `spotted` when the wake
    /// phrase is in it. The answer window wins over everything and stays open; a
    /// spotted phrase wins over an open command window, and an utterance that
    /// began before that window opened leaves it open.
    pub fn admit(&mut self, spotted: bool, started: Instant) -> Admission {
        if self
            .answer_window
            .is_some_and(|window| window.contains(started))
        {
            return Admission::Answer;
        }
        let in_window = self
            .window
            .take_if(|window| window.opens <= started)
            .is_some_and(|window| window.contains(started));
        match (spotted, in_window) {
            (true, _) => Admission::WakePhrase,
            (false, true) => Admission::Command,
            (false, false) => Admission::Dropped,
        }
    }

    /// The phrase in the utterance that began at `started` had nothing after
    /// it: take the next utterance that begins within `command_timeout` as
    /// the command.
    pub fn await_command(&mut self, started: Instant) {
        self.window = Some(CommandWindow {
            opens: started,
            closes: Instant::now() + self.command_timeout,
        });
    }
}

#[cfg(test)]
mod tests {
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
        assert_eq!(
            gate.admit(false, ended + Duration::from_secs(1)),
            Admission::Command,
            "speech that began before the window leaves it open"
        );

        gate.open_follow_up(ended);
        assert_eq!(
            gate.admit(true, ended),
            Admission::WakePhrase,
            "a spotted phrase wins over an open window"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn answer_window_takes_only_speech_begun_after_the_question_until_settled() {
        const WAIT: Duration = Duration::from_secs(10);
        let mut gate = WakeGate::new(Duration::from_secs(4), None);
        let asked = Instant::now();
        gate.await_answer(asked, WAIT);
        assert_eq!(
            gate.admit(false, asked - Duration::from_millis(1)),
            Admission::Dropped,
            "speech over the question is not an answer to it"
        );
        for spotted in [false, true] {
            assert_eq!(
                gate.admit(spotted, asked + Duration::from_secs(2)),
                Admission::Answer,
                "the window stays open until settled"
            );
        }
        assert_eq!(
            gate.admit(false, asked + WAIT + Duration::from_millis(1)),
            Admission::Dropped
        );

        gate.settle_answer();
        assert_eq!(
            gate.admit(false, asked + Duration::from_secs(2)),
            Admission::Dropped
        );
    }
}
