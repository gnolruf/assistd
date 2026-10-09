//! Hands-free continuous listening: VAD-gated segmentation that emits completed
//! transcripts as a long-running stream, unlike one-shot [`crate::VoiceInput`] presses.

use std::fmt;

use async_trait::async_trait;
use thiserror::Error;
use tokio::sync::{broadcast, watch};

pub mod capture;
pub mod consumer;
pub mod mic;
pub mod playback_gate;
pub mod vad;

pub use mic::MicContinuousListener;

/// Errors surfaced by [`ContinuousListener`] implementations.
#[derive(Debug, Error)]
pub enum ListenError {
    /// Continuous listening is turned off in config.
    #[error("continuous listening is disabled in config (voice.continuous.enabled = false)")]
    Disabled,
}

/// A long-running, VAD-gated listener emitting completed utterance
/// transcripts.
#[async_trait]
pub trait ContinuousListener: fmt::Debug + Send + Sync + 'static {
    /// Open the mic and start segmenting. A no-op when already active.
    async fn start(&self) -> Result<(), ListenError>;

    /// Close the mic and drain any in-flight utterance. Idempotent.
    async fn stop(&self) -> Result<(), ListenError>;

    /// Whether the listener is currently capturing.
    fn is_active(&self) -> bool;

    /// Completed transcripts. Slow consumers may miss older utterances.
    /// Empty transcripts are never delivered.
    fn subscribe_utterances(&self) -> broadcast::Receiver<String>;

    /// On/off transitions. The initial value is the current state.
    fn subscribe_state(&self) -> watch::Receiver<bool>;
}

/// [`ContinuousListener`] for when continuous listening is off in config:
/// refuses to start and never delivers transcripts.
#[derive(Debug)]
pub struct NoContinuousListener {
    state_tx: watch::Sender<bool>,
    utterances: broadcast::Sender<String>,
}

impl Default for NoContinuousListener {
    fn default() -> Self {
        Self::new()
    }
}

impl NoContinuousListener {
    /// A placeholder that stays inactive.
    pub fn new() -> Self {
        let (state_tx, _) = watch::channel(false);
        let (utterances, _) = broadcast::channel(16);
        Self {
            state_tx,
            utterances,
        }
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

    fn subscribe_utterances(&self) -> broadcast::Receiver<String> {
        self.utterances.subscribe()
    }

    fn subscribe_state(&self) -> watch::Receiver<bool> {
        self.state_tx.subscribe()
    }
}
