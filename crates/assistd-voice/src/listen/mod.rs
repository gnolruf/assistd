//! Hands-free continuous listening: VAD-gated utterance segmentation
//! that feeds completed transcripts to the agent loop without a
//! hotkey press.
//!
//! The trait [`ContinuousListener`] is intentionally independent of
//! [`crate::VoiceInput`] because the lifecycles differ: PTT is
//! one-shot (open → buffer → close), continuous is a long-running
//! stream of utterances. A single daemon build can own both, sharing
//! the [`WhisperTranscriber`](crate::WhisperTranscriber) model
//! context but using separate cpal streams — because the caller
//! enforces mutual exclusion (only one can hold the mic at a time),
//! having two objects is simpler than multiplexing state.

use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::{broadcast, watch};

pub mod capture;
pub mod consumer;
pub mod mic;
pub mod vad;

pub use mic::MicContinuousListener;

/// A long-running, VAD-gated listener that emits a stream of
/// completed utterance transcripts.
///
/// Implementations hold a whisper transcriber (or a stub, for
/// disabled builds), open a cpal input stream while active, and
/// re-close it when stopped.
#[async_trait]
pub trait ContinuousListener: Send + Sync + 'static {
    /// Begin listening. Opens the mic, starts the VAD loop, and
    /// returns once the cpal stream is playing. Repeated `start`
    /// while already active is a no-op (returns `Ok`) — callers
    /// that need to error on double-start should check
    /// [`Self::is_active`] first.
    async fn start(&self) -> Result<()>;

    /// Stop listening. Closes the cpal stream and drains any
    /// in-flight utterance. Idempotent.
    async fn stop(&self) -> Result<()>;

    /// Cheap snapshot of the current on/off state.
    fn is_active(&self) -> bool;

    /// Subscribe to the stream of completed transcripts. The
    /// broadcast channel has a finite depth; slow consumers may miss
    /// older utterances but will see every new one after they catch
    /// up. Empty strings (whisper trimming to pure silence) are
    /// filtered out upstream — receivers never see them.
    fn subscribe_utterances(&self) -> broadcast::Receiver<String>;

    /// Subscribe to on/off state transitions. The initial value is
    /// the current state.
    fn subscribe_state(&self) -> watch::Receiver<bool>;
}

/// Placeholder implementation for builds where the feature is
/// disabled or the daemon wants to force-off without building cpal.
/// All methods succeed trivially; transcripts never arrive.
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
    async fn start(&self) -> Result<()> {
        anyhow::bail!("continuous listening is not enabled in this build")
    }

    async fn stop(&self) -> Result<()> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn no_continuous_listener_start_errors() {
        assert!(NoContinuousListener::new().start().await.is_err());
    }

    #[tokio::test]
    async fn no_continuous_listener_stop_ok() {
        NoContinuousListener::new().stop().await.unwrap();
    }

    #[tokio::test]
    async fn no_continuous_listener_is_inactive() {
        assert!(!NoContinuousListener::new().is_active());
    }
}
