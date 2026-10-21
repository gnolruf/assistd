//! The state machine that turns transcripts into wake-phrase detections
//! and commands.

use std::time::Duration;

use assistd_ipc::ListenPhase;
use tokio::time::Instant;

use super::wake_phrase::WakePhrase;

/// What one transcript amounts to once gated on the wake phrase.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Admission {
    /// Not addressed to assistd.
    Dropped,
    /// The phrase alone; the next utterance is the command.
    WakePhrase,
    /// The phrase followed by this command.
    WakePhraseAndCommand(String),
    /// An utterance inside a command window, taken whole as the command.
    Command(String),
}

/// Admits a transcript when it opens with the wake phrase, or when it
/// began inside a command window: `command_timeout` after a bare phrase,
/// or `follow_up` after a reply ended. One utterance closes the window.
#[derive(Debug)]
pub struct WakeGate {
    phrase: WakePhrase,
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
    pub fn new(phrase: WakePhrase, command_timeout: Duration, follow_up: Option<Duration>) -> Self {
        Self {
            phrase,
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

    /// The phrase to bias transcription toward.
    pub fn phrase(&self) -> &str {
        self.phrase.phrase()
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

    /// Gate `transcript`, whose speech began at `started`. Inside a window,
    /// a transcript that opens with the phrase is still stripped of it.
    pub fn admit(&mut self, transcript: &str, started: Instant) -> Admission {
        let in_window = self
            .window
            .take()
            .is_some_and(|window| window.contains(started));
        match self.phrase.match_prefix(transcript) {
            Some("") => {
                self.window = Some(CommandWindow {
                    opens: started,
                    closes: Instant::now() + self.command_timeout,
                });
                Admission::WakePhrase
            }
            Some(command) => Admission::WakePhraseAndCommand(command.to_string()),
            None if in_window => Admission::Command(transcript.to_string()),
            None => Admission::Dropped,
        }
    }
}

#[cfg(test)]
mod tests;
