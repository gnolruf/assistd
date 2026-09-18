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
    pub fn new(inner: Arc<dyn VoiceOutput>, initially_enabled: bool) -> Arc<Self> {
        Arc::new(Self {
            inner,
            enabled: AtomicBool::new(initially_enabled),
            skip_epoch: AtomicU64::new(0),
        })
    }

    pub fn enabled(&self) -> bool {
        self.enabled.load(Ordering::SeqCst)
    }

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

    pub fn inner(&self) -> &Arc<dyn VoiceOutput> {
        &self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NoVoiceOutput;
    use anyhow::Result;
    use async_trait::async_trait;
    use parking_lot::Mutex;

    #[derive(Default)]
    struct RecordingOutput {
        spoken: Mutex<Vec<String>>,
        cancels: AtomicU64,
        wait_idles: AtomicU64,
    }

    #[async_trait]
    impl VoiceOutput for RecordingOutput {
        async fn speak(&self, text: String) -> Result<()> {
            self.spoken.lock().push(text);
            Ok(())
        }
        async fn wait_idle(&self) -> Result<()> {
            self.wait_idles.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        async fn cancel(&self) {
            self.cancels.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn new_starts_with_given_enabled_flag_and_zero_epoch() {
        let ctrl = VoiceOutputController::new(Arc::new(NoVoiceOutput), true);
        assert!(ctrl.enabled());
        assert_eq!(ctrl.current_epoch(), 0);

        let off = VoiceOutputController::new(Arc::new(NoVoiceOutput), false);
        assert!(!off.enabled());
    }

    #[tokio::test]
    async fn set_enabled_false_cancels_inner() {
        let inner = Arc::new(RecordingOutput::default());
        let ctrl = VoiceOutputController::new(inner.clone(), true);
        ctrl.set_enabled(false).await;
        assert!(!ctrl.enabled());
        assert_eq!(inner.cancels.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn set_enabled_true_does_not_cancel_or_bump_epoch() {
        let inner = Arc::new(RecordingOutput::default());
        let ctrl = VoiceOutputController::new(inner.clone(), false);
        ctrl.set_enabled(true).await;
        assert!(ctrl.enabled());
        assert_eq!(inner.cancels.load(Ordering::SeqCst), 0);
        assert_eq!(ctrl.current_epoch(), 0);
    }

    #[tokio::test]
    async fn set_enabled_idempotent_off_does_not_double_cancel() {
        let inner = Arc::new(RecordingOutput::default());
        let ctrl = VoiceOutputController::new(inner.clone(), false);
        ctrl.set_enabled(false).await;
        assert_eq!(inner.cancels.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn skip_advances_epoch_and_cancels() {
        let inner = Arc::new(RecordingOutput::default());
        let ctrl = VoiceOutputController::new(inner.clone(), true);
        let before = ctrl.current_epoch();
        ctrl.skip().await;
        assert_eq!(ctrl.current_epoch(), before + 1);
        assert_eq!(inner.cancels.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn interrupt_is_alias_of_skip() {
        let inner = Arc::new(RecordingOutput::default());
        let ctrl = VoiceOutputController::new(inner.clone(), true);
        ctrl.interrupt().await;
        assert_eq!(ctrl.current_epoch(), 1);
        assert_eq!(inner.cancels.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn should_speak_returns_speak_when_enabled_and_epoch_matches() {
        let ctrl = VoiceOutputController::new(Arc::new(NoVoiceOutput), true);
        let start = ctrl.current_epoch();
        assert_eq!(ctrl.should_speak(start), SpeakDecision::Speak);
    }

    #[tokio::test]
    async fn should_speak_returns_drop_for_skip_after_epoch_advance() {
        let ctrl = VoiceOutputController::new(Arc::new(NoVoiceOutput), true);
        let start = ctrl.current_epoch();
        ctrl.skip().await;
        assert_eq!(ctrl.should_speak(start), SpeakDecision::DropForSkip);
    }

    #[tokio::test]
    async fn should_speak_returns_drop_silent_when_disabled_same_epoch() {
        let ctrl = VoiceOutputController::new(Arc::new(NoVoiceOutput), false);
        let start = ctrl.current_epoch();
        assert_eq!(ctrl.should_speak(start), SpeakDecision::DropSilent);
    }

    #[tokio::test]
    async fn drop_for_skip_takes_priority_over_drop_silent() {
        let ctrl = VoiceOutputController::new(Arc::new(NoVoiceOutput), false);
        let start = ctrl.current_epoch();
        ctrl.skip().await;
        assert_eq!(ctrl.should_speak(start), SpeakDecision::DropForSkip);
    }

    #[tokio::test]
    async fn future_query_after_skip_speaks_normally() {
        let ctrl = VoiceOutputController::new(Arc::new(NoVoiceOutput), true);
        ctrl.skip().await;
        let start = ctrl.current_epoch();
        assert_eq!(ctrl.should_speak(start), SpeakDecision::Speak);
    }

    #[tokio::test]
    async fn toggle_off_then_on_resumes_speak_decision() {
        let inner = Arc::new(RecordingOutput::default());
        let ctrl = VoiceOutputController::new(inner.clone(), true);
        let start = ctrl.current_epoch();
        ctrl.set_enabled(false).await;
        assert_eq!(ctrl.should_speak(start), SpeakDecision::DropSilent);
        ctrl.set_enabled(true).await;
        assert_eq!(ctrl.should_speak(start), SpeakDecision::Speak);
        assert_eq!(ctrl.current_epoch(), start);
    }
}
