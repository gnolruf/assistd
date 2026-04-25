//! Runtime control over a [`VoiceOutput`] backend: toggle on/off, skip
//! the current response, interrupt for push-to-talk barge-in.
//!
//! The controller composes with — rather than replacing — the existing
//! `Arc<dyn VoiceOutput>`. It owns two pieces of mutable runtime state:
//!
//! - `enabled`: a runtime mute switch. Off means the speech worker
//!   silently discards sentences as it dequeues them. The setting is
//!   *not* persisted across daemon restarts (the initial value is read
//!   from `voice.synthesis.enabled` at construction).
//! - `skip_epoch`: a monotonically increasing counter that lets each
//!   per-query speech worker capture its own epoch at spawn time and
//!   detect later "skip everything in flight" requests by comparing
//!   against the current value. This avoids per-query state in
//!   `AppState` and handles concurrent queries correctly: a `skip()`
//!   advances the global epoch so every active worker independently
//!   sees the change, while a future query captures the new epoch at
//!   spawn and is unaffected.
//!
//! Both `set_enabled(false)` and `skip()` call `inner.cancel()` so
//! audio that's already in the playback queue stops within milliseconds
//! (`PiperVoiceOutput::cancel` clears rodio's FIFO).
//!
//! `set_enabled(true)` does *not* advance the epoch — re-enabling
//! resumes speaking later sentences for the same query (those that
//! arrive after the toggle-back-on; sentences enqueued during the off
//! window are dropped at dequeue and not buffered).

use crate::VoiceOutput;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Shared, reference-counted controller wrapping an `Arc<dyn VoiceOutput>`.
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
    /// TTS is disabled — drain the channel without speaking.
    DropSilent,
    /// Skip was triggered since the worker started — drain without speaking.
    DropForSkip,
}

impl VoiceOutputController {
    /// Create a controller. `initially_enabled` typically comes from
    /// `config.voice.synthesis.enabled`.
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

    /// Flip the runtime mute switch. Turning off cancels currently-queued
    /// audio; turning on is a pure flag flip with no cancel.
    pub async fn set_enabled(&self, on: bool) {
        let prev = self.enabled.swap(on, Ordering::SeqCst);
        if prev && !on {
            self.inner.cancel().await;
        }
    }

    /// Abort the current response: advance the epoch (so active speech
    /// workers drop the rest of their queued sentences) and clear the
    /// audio playback queue. Does not change the enabled flag — TTS
    /// stays armed for the next query.
    pub async fn skip(&self) {
        self.skip_epoch.fetch_add(1, Ordering::SeqCst);
        self.inner.cancel().await;
    }

    /// PTT barge-in. Identical semantics to [`skip`](Self::skip); kept
    /// as a named alias so call sites read intuitively.
    pub async fn interrupt(&self) {
        self.skip().await;
    }

    /// Speech-worker policy: given the epoch the worker captured at
    /// spawn time, return what to do with the next dequeued sentence.
    pub fn should_speak(&self, start_epoch: u64) -> SpeakDecision {
        if self.skip_epoch.load(Ordering::SeqCst) != start_epoch {
            SpeakDecision::DropForSkip
        } else if !self.enabled.load(Ordering::SeqCst) {
            SpeakDecision::DropSilent
        } else {
            SpeakDecision::Speak
        }
    }

    /// Access the underlying backend (used by the speech worker for
    /// `speak()` and `wait_idle()`).
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
    use std::sync::Mutex;

    #[derive(Default)]
    struct RecordingOutput {
        spoken: Mutex<Vec<String>>,
        cancels: AtomicU64,
        wait_idles: AtomicU64,
    }

    #[async_trait]
    impl VoiceOutput for RecordingOutput {
        async fn speak(&self, text: String) -> Result<()> {
            self.spoken.lock().unwrap().push(text);
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
        // Both disabled AND epoch advanced: skip wins so callers can
        // distinguish "user pressed skip" from "user toggled off".
        assert_eq!(ctrl.should_speak(start), SpeakDecision::DropForSkip);
    }

    #[tokio::test]
    async fn future_query_after_skip_speaks_normally() {
        let ctrl = VoiceOutputController::new(Arc::new(NoVoiceOutput), true);
        ctrl.skip().await; // simulates a previous query being skipped
        // New query captures the post-skip epoch:
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
        // Same start_epoch (toggle does NOT bump) so the worker resumes.
        assert_eq!(ctrl.should_speak(start), SpeakDecision::Speak);
        // No epoch advance from either toggle:
        assert_eq!(ctrl.current_epoch(), start);
    }
}
