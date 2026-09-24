//! Runtime control over a [`VoiceOutput`]: a mute switch and a skip
//! epoch. Each speech worker captures the epoch when it starts;
//! [`VoiceOutputController::skip`] advances it so every in-flight
//! worker drops its remaining sentences while later queries are
//! unaffected.

use crate::VoiceOutput;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Mute switch and skip epoch over an `Arc<dyn VoiceOutput>`.
pub struct VoiceOutputController {
    inner: Arc<dyn VoiceOutput>,
    enabled: AtomicBool,
    skip_epoch: AtomicU64,
}

/// Per-sentence decision returned by [`VoiceOutputController::should_speak`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeakDecision {
    /// Pass the sentence to `inner.speak()`.
    Speak,
    /// TTS is disabled; drain the channel without speaking.
    DropSilent,
    /// Skip was triggered since the worker started; drain without speaking.
    DropForSkip,
}

impl VoiceOutputController {
    /// Wraps `inner`, muted unless `initially_enabled`, at epoch zero.
    pub fn new(inner: Arc<dyn VoiceOutput>, initially_enabled: bool) -> Arc<Self> {
        Arc::new(Self {
            inner,
            enabled: AtomicBool::new(initially_enabled),
            skip_epoch: AtomicU64::new(0),
        })
    }

    /// Whether speech is currently unmuted.
    pub fn enabled(&self) -> bool {
        self.enabled.load(Ordering::SeqCst)
    }

    /// The skip epoch a worker captures when it starts and later passes
    /// to [`should_speak`](Self::should_speak).
    pub fn current_epoch(&self) -> u64 {
        self.skip_epoch.load(Ordering::SeqCst)
    }

    /// Turning off cancels queued audio. Turning on does not advance
    /// the epoch, so a running worker resumes speaking.
    pub async fn set_enabled(&self, on: bool) {
        let prev = self.enabled.swap(on, Ordering::SeqCst);
        if prev && !on {
            self.inner.cancel().await;
        }
    }

    /// Advance the epoch and cancel queued audio. Leaves the mute
    /// switch unchanged.
    pub async fn skip(&self) {
        self.skip_epoch.fetch_add(1, Ordering::SeqCst);
        self.inner.cancel().await;
    }

    /// Push-to-talk barge-in; identical to [`skip`](Self::skip).
    pub async fn interrupt(&self) {
        self.skip().await;
    }

    /// What a worker that started at `start_epoch` should do with its
    /// next sentence.
    pub fn should_speak(&self, start_epoch: u64) -> SpeakDecision {
        if self.skip_epoch.load(Ordering::SeqCst) != start_epoch {
            SpeakDecision::DropForSkip
        } else if !self.enabled.load(Ordering::SeqCst) {
            SpeakDecision::DropSilent
        } else {
            SpeakDecision::Speak
        }
    }

    /// The wrapped output.
    pub fn inner(&self) -> &Arc<dyn VoiceOutput> {
        &self.inner
    }
}

#[cfg(test)]
mod tests;
