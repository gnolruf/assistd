//! Hands-free wake-word listening: VAD-cut utterances, transcribed and sent to clients only
//! once a keyword spotter hears the wake phrase, unlike one-shot [`crate::VoiceInput`] presses.

use std::fmt;

use assistd_ipc::ListenPhase;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::{broadcast, mpsc, watch};
use tokio::time::Instant;

use crate::mic::AudioCaptureError;

pub mod barge_in;
pub mod capture;
pub mod consumer;
pub mod mic;
mod session;
pub mod silero;
pub mod spotter;
pub mod vad;
pub mod wake_gate;

pub use mic::MicContinuousListener;
pub use silero::SileroVad;
pub use spotter::{SpotterError, WakeWordSpotter};

/// Errors surfaced by [`ContinuousListener`] implementations.
#[derive(Debug, Error)]
pub enum ListenError {
    /// Wake-word listening is turned off in config.
    #[error("wake-word listening is disabled in config (voice.wakeword.enabled = false)")]
    Disabled,
    /// The mic is taken by a wake-word test.
    #[error("wake-word test in progress")]
    Testing,
    /// The mic is taken by wake-word listening.
    #[error("wake-word listening is active")]
    Listening,
    #[error("audio capture failed: {0}")]
    Capture(#[from] AudioCaptureError),
    #[error("listen worker panicked: {0}")]
    Worker(#[from] tokio::task::JoinError),
    #[error(transparent)]
    VoiceWorker(#[from] crate::error::VoiceWorkerError),
}

/// What the listener heard that is addressed to assistd.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "text", rename_all = "snake_case")]
pub enum Heard {
    /// The wake phrase, alone or ahead of a command.
    WakePhrase,
    /// A command, without the wake phrase; never empty.
    Command(String),
    /// The user is talking over the reply; their words follow as a
    /// [`Heard::Command`] once they finish.
    BargeIn,
}

/// What a wake-word test reports from the mic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WakeTest {
    /// An utterance started (`true`) or ended (`false`), kept or not.
    Speaking { speaking: bool },
    /// An utterance ended; whether the wake phrase was spotted in it.
    Utterance { matched: bool },
}

/// A long-running, VAD-gated listener that emits what follows the wake
/// phrase.
#[async_trait]
pub trait ContinuousListener: fmt::Debug + Send + Sync + 'static {
    /// Open the mic and start segmenting. A no-op when already active;
    /// [`ListenError::Testing`] while [`Self::test_wake_word`] runs.
    async fn start(&self) -> Result<(), ListenError>;

    /// Close the mic and drain any in-flight utterance. Idempotent.
    async fn stop(&self) -> Result<(), ListenError>;

    /// Hand the mic to push-to-talk: stop without acting on anything already
    /// captured, and record the hand-over for [`Self::last_push_to_talk`].
    async fn stop_for_push_to_talk(&self) -> Result<(), ListenError>;

    /// When push-to-talk last took the mic; commands heard before then
    /// belong to a conversation it superseded.
    fn last_push_to_talk(&self) -> Option<Instant>;

    /// Whether the listener is currently capturing.
    fn is_active(&self) -> bool;

    /// Wake-phrase detections and commands. Slow consumers may miss older ones.
    fn subscribe_heard(&self) -> broadcast::Receiver<Heard>;

    /// Phase transitions. The initial value is the current phase.
    fn subscribe_phase(&self) -> watch::Receiver<ListenPhase>;

    /// Take the next utterance as a command without the wake phrase if it
    /// begins soon enough after now; the reply to the last one has ended.
    /// Ignored while listening is stopped.
    fn open_follow_up(&self);

    /// Whether a reply to a wake-word command is under way. While it is and
    /// barge-in is on, speech is checked for [`Heard::BargeIn`] instead of
    /// for the wake phrase.
    fn set_replying(&self, replying: bool);

    /// Report into `sink` each start and end of speech the mic hears and whether each utterance
    /// held the wake phrase, until `sink` closes. Fails with [`ListenError::Listening`] while
    /// listening is active.
    async fn test_wake_word(&self, sink: mpsc::Sender<WakeTest>) -> Result<(), ListenError>;
}

/// [`ContinuousListener`] for when continuous listening is off in config:
/// refuses to start and never delivers transcripts.
#[derive(Debug)]
pub struct NoContinuousListener {
    phase_tx: watch::Sender<ListenPhase>,
    heard: broadcast::Sender<Heard>,
}

impl Default for NoContinuousListener {
    fn default() -> Self {
        Self::new()
    }
}

impl NoContinuousListener {
    /// A placeholder that stays inactive.
    pub fn new() -> Self {
        let (phase_tx, _) = watch::channel(ListenPhase::Off);
        let (heard, _) = broadcast::channel(16);
        Self { phase_tx, heard }
    }
}

#[async_trait]
impl ContinuousListener for NoContinuousListener {
    async fn start(&self) -> Result<(), ListenError> {
        Err(ListenError::Disabled)
    }

    async fn stop(&self) -> Result<(), ListenError> {
        Ok(())
    }

    async fn stop_for_push_to_talk(&self) -> Result<(), ListenError> {
        Ok(())
    }

    fn last_push_to_talk(&self) -> Option<Instant> {
        None
    }

    fn is_active(&self) -> bool {
        false
    }

    fn subscribe_heard(&self) -> broadcast::Receiver<Heard> {
        self.heard.subscribe()
    }

    fn subscribe_phase(&self) -> watch::Receiver<ListenPhase> {
        self.phase_tx.subscribe()
    }

    fn open_follow_up(&self) {}

    fn set_replying(&self, _replying: bool) {}

    async fn test_wake_word(&self, _sink: mpsc::Sender<WakeTest>) -> Result<(), ListenError> {
        Err(ListenError::Disabled)
    }
}
