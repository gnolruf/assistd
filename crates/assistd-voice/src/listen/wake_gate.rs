//! The `Armed` / `AwaitingCommand` state machine that turns transcripts
//! into wake-phrase detections and commands.

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
    /// The utterance after a bare phrase, taken whole as the command.
    Command(String),
}

/// Admits a transcript when it opens with the wake phrase, or when it
/// began within `command_timeout` of a bare phrase.
#[derive(Debug)]
pub struct WakeGate {
    phrase: WakePhrase,
    command_timeout: Duration,
    command_deadline: Option<Instant>,
}

impl WakeGate {
    pub fn new(phrase: WakePhrase, command_timeout: Duration) -> Self {
        Self {
            phrase,
            command_timeout,
            command_deadline: None,
        }
    }

    /// The phrase to bias transcription toward.
    pub fn phrase(&self) -> &str {
        self.phrase.phrase()
    }

    /// When the command window after a bare phrase closes, if one is open.
    pub fn command_deadline(&self) -> Option<Instant> {
        self.command_deadline
    }

    /// `Capturing` while the command window is open at `now`, else `Armed`.
    pub fn phase(&self, now: Instant) -> ListenPhase {
        match self.command_deadline {
            Some(deadline) if now < deadline => ListenPhase::Capturing,
            _ => ListenPhase::Armed,
        }
    }

    /// Gate `transcript`, whose speech began at `started`.
    pub fn admit(&mut self, transcript: &str, started: Instant) -> Admission {
        if let Some(deadline) = self.command_deadline.take()
            && started <= deadline
        {
            return Admission::Command(transcript.to_string());
        }
        match self.phrase.match_prefix(transcript) {
            None => Admission::Dropped,
            Some("") => {
                self.command_deadline = Some(Instant::now() + self.command_timeout);
                Admission::WakePhrase
            }
            Some(command) => Admission::WakePhraseAndCommand(command.to_string()),
        }
    }
}
