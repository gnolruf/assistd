//! Hands-free wake-word listening: VAD-gated segmentation whose transcripts
//! reach clients only after the wake phrase, unlike one-shot [`crate::VoiceInput`] presses.

use std::fmt;

use assistd_ipc::ListenPhase;
use async_trait::async_trait;
use thiserror::Error;
use tokio::sync::{broadcast, watch};

pub mod capture;
pub mod consumer;
pub mod mic;
pub mod playback_gate;
pub mod vad;
pub mod wake_gate;
pub mod wake_phrase;

pub use mic::MicContinuousListener;
pub use wake_phrase::WakePhrase;

/// Errors surfaced by [`ContinuousListener`] implementations.
#[derive(Debug, Error)]
pub enum ListenError {
    /// Wake-word listening is turned off in config.
    #[error("wake-word listening is disabled in config (voice.wakeword.enabled = false)")]
    Disabled,
}

/// What the listener heard that is addressed to assistd.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Heard {
    /// The wake phrase, alone or ahead of a command.
    WakePhrase,
    /// A command, without the wake phrase; never empty.
    Command(String),
}

/// A long-running, VAD-gated listener that emits what follows the wake
/// phrase.
#[async_trait]
pub trait ContinuousListener: fmt::Debug + Send + Sync + 'static {
    /// Open the mic and start segmenting. A no-op when already active.
    async fn start(&self) -> Result<(), ListenError>;

    /// Close the mic and drain any in-flight utterance. Idempotent.
    async fn stop(&self) -> Result<(), ListenError>;

    /// Whether the listener is currently capturing.
    fn is_active(&self) -> bool;

    /// Wake-phrase detections and commands. Slow consumers may miss older ones.
    fn subscribe_heard(&self) -> broadcast::Receiver<Heard>;

    /// Phase transitions. The initial value is the current phase.
    fn subscribe_phase(&self) -> watch::Receiver<ListenPhase>;
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

    fn is_active(&self) -> bool {
        false
    }

    fn subscribe_heard(&self) -> broadcast::Receiver<Heard> {
        self.heard.subscribe()
    }

    fn subscribe_phase(&self) -> watch::Receiver<ListenPhase> {
        self.phase_tx.subscribe()
    }
}
