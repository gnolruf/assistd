//! Runtime control over a [`VoiceOutput`]: a mute switch, a skip epoch that
//! in-flight speech workers compare against to drop stale sentences, and a
//! speaking signal that other subsystems watch to gate the mic.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use tokio::sync::watch;

use crate::VoiceOutput;

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

/// Mute switch, skip epoch, and speaking signal over an `Arc<dyn VoiceOutput>`.
#[derive(Debug)]
pub struct VoiceOutputController {
    inner: Arc<dyn VoiceOutput>,
    enabled: AtomicBool,
    skip_epoch: AtomicU64,
    active_speakers: AtomicUsize,
    speaking_tx: watch::Sender<bool>,
}

impl VoiceOutputController {
    /// Wraps `inner`, muted unless `initially_enabled`, at epoch zero.
    pub fn new(inner: Arc<dyn VoiceOutput>, initially_enabled: bool) -> Arc<Self> {
        let (speaking_tx, _) = watch::channel(false);
        Arc::new(Self {
            inner,
            enabled: AtomicBool::new(initially_enabled),
            skip_epoch: AtomicU64::new(0),
            active_speakers: AtomicUsize::new(0),
            speaking_tx,
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

    /// Mark audio as audible until the returned guard drops. Speaking
    /// stays true while any guard is alive.
    pub fn begin_speaking(self: &Arc<Self>) -> SpeakingGuard {
        if self.active_speakers.fetch_add(1, Ordering::SeqCst) == 0 {
            self.speaking_tx.send_replace(true);
        }
        SpeakingGuard {
            controller: Arc::clone(self),
        }
    }

    /// Whether any speech worker currently holds a [`SpeakingGuard`].
    pub fn is_speaking(&self) -> bool {
        *self.speaking_tx.borrow()
    }

    /// Speaking transitions. The initial value is the current state.
    pub fn subscribe_speaking(&self) -> watch::Receiver<bool> {
        self.speaking_tx.subscribe()
    }

    /// The wrapped output.
    pub fn inner(&self) -> &Arc<dyn VoiceOutput> {
        &self.inner
    }

    fn end_speaking(&self) {
        if self.active_speakers.fetch_sub(1, Ordering::SeqCst) == 1 {
            self.speaking_tx.send_replace(false);
        }
    }
}

/// Keeps the controller's speaking signal raised; dropping the last
/// live guard lowers it.
#[must_use = "speaking ends as soon as the guard is dropped"]
#[derive(Debug)]
pub struct SpeakingGuard {
    controller: Arc<VoiceOutputController>,
}

impl Drop for SpeakingGuard {
    fn drop(&mut self) {
        self.controller.end_speaking();
    }
}

#[cfg(test)]
mod tests;
