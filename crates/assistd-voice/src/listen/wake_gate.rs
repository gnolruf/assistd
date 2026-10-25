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
}

/// Admits an utterance when the wake phrase was spotted in it, or when it
/// began inside a command window: `command_timeout` after a bare phrase,
/// or `follow_up` after a reply ended. One utterance closes the window.
#[derive(Debug)]
pub struct WakeGate {
    command_timeout: Duration,
    /// `None` when follow-ups are disabled.
    follow_up: Option<Duration>,
    window: Option<CommandWindow>,
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
        }
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

    /// When the open command window closes, if one is open.
    pub fn command_deadline(&self) -> Option<Instant> {
        self.window.map(|window| window.closes)
    }

    /// `Capturing` while a command window is open at `now`, else `Armed`.
    pub fn phase(&self, now: Instant) -> ListenPhase {
        match self.window {
            Some(window) if window.contains(now) => ListenPhase::Capturing,
            _ => ListenPhase::Armed,
        }
    }

    /// Gate an utterance that began at `started`, `spotted` when the wake
    /// phrase is in it. A spotted phrase wins over an open window.
    pub fn admit(&mut self, spotted: bool, started: Instant) -> Admission {
        let in_window = self
            .window
            .take()
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
mod tests;
