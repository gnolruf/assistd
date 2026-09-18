//! Hands-free continuous listening: VAD-gated utterance segmentation
//! that emits completed transcripts without a hotkey press. Separate
//! from [`crate::VoiceInput`] because the lifecycles differ: a press
//! is one-shot, listening is a long-running stream.

use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::{broadcast, watch};

pub mod capture;
pub mod consumer;
pub mod mic;
pub mod vad;

pub use mic::MicContinuousListener;

/// A long-running, VAD-gated listener emitting completed utterance
/// transcripts.
#[async_trait]
pub trait ContinuousListener: Send + Sync + 'static {
    /// Open the mic and start segmenting. A no-op when already active.
    async fn start(&self) -> Result<()>;

    /// Close the mic and drain any in-flight utterance. Idempotent.
    async fn stop(&self) -> Result<()>;

    fn is_active(&self) -> bool;

    /// Completed transcripts. Slow consumers may miss older utterances.
    /// Empty transcripts are never delivered.
    fn subscribe_utterances(&self) -> broadcast::Receiver<String>;

    /// On/off transitions. The initial value is the current state.
    fn subscribe_state(&self) -> watch::Receiver<bool>;
}

/// Placeholder [`ContinuousListener`] that never delivers transcripts.
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
