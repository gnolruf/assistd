//! Runtime control over a [`VoiceOutput`] that may still be starting: mute, a skip epoch that
//! drops stale sentences, a speaking signal that gates the mic, and recent sentences for echo.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use assistd_utils::readiness::{NotReady, Readiness, ReadinessCell};
use tokio::sync::watch;

use crate::VoiceOutput;

/// Sentences of the current reply kept for [`VoiceOutputController::recent_speech`].
const RECENT_SENTENCES: usize = 4;

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

/// Mute switch, skip epoch, and speaking signal over an `Arc<dyn VoiceOutput>`
/// that may still be starting.
#[derive(Debug)]
pub struct VoiceOutputController {
    output: ReadinessCell<Arc<dyn VoiceOutput>>,
    enabled: AtomicBool,
    skip_epoch: AtomicU64,
    active_speakers: AtomicUsize,
    speaking_tx: watch::Sender<bool>,
    recent_speech: parking_lot::Mutex<VecDeque<String>>,
}

impl VoiceOutputController {
    /// A controller whose output is still starting, muted unless
    /// `initially_enabled`, at epoch zero.
    pub fn new(initially_enabled: bool) -> Arc<Self> {
        let (speaking_tx, _) = watch::channel(false);
        Arc::new(Self {
            output: ReadinessCell::starting(),
            enabled: AtomicBool::new(initially_enabled),
            skip_epoch: AtomicU64::new(0),
            active_speakers: AtomicUsize::new(0),
            speaking_tx,
            recent_speech: parking_lot::Mutex::new(VecDeque::with_capacity(RECENT_SENTENCES)),
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

    /// Turning off advances the epoch and cancels queued audio, so a
    /// sentence mid-synthesis is dropped and the running reply stays silent.
    pub async fn set_enabled(&self, on: bool) {
        let prev = self.enabled.swap(on, Ordering::SeqCst);
        if prev && !on {
            self.skip().await;
        }
    }

    /// Advance the epoch and cancel queued audio. Leaves the mute
    /// switch unchanged.
    pub async fn skip(&self) {
        self.skip_epoch.fetch_add(1, Ordering::SeqCst);
        self.cancel_queued().await;
    }

    /// Push-to-talk barge-in; identical to [`skip`](Self::skip).
    pub async fn interrupt(&self) {
        self.skip().await;
    }

    /// Cancel queued audio within the epoch, so the running reply still
    /// speaks its later sentences.
    pub async fn hush(&self) {
        self.cancel_queued().await;
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
            self.recent_speech.lock().clear();
            self.speaking_tx.send_replace(true);
        }
        SpeakingGuard {
            controller: Arc::clone(self),
        }
    }

    /// Record `sentence` as handed to the output for speaking.
    pub fn remember_spoken(&self, sentence: &str) {
        let mut recent = self.recent_speech.lock();
        if recent.len() == RECENT_SENTENCES {
            recent.pop_front();
        }
        recent.push_back(sentence.to_owned());
    }

    /// The last few sentences spoken since speaking last began, oldest first.
    pub fn recent_speech(&self) -> Vec<String> {
        self.recent_speech.lock().iter().cloned().collect()
    }

    /// Whether any speech worker currently holds a [`SpeakingGuard`].
    pub fn is_speaking(&self) -> bool {
        *self.speaking_tx.borrow()
    }

    /// Speaking transitions. The initial value is the current state.
    pub fn subscribe_speaking(&self) -> watch::Receiver<bool> {
        self.speaking_tx.subscribe()
    }

    /// Record how far the output's startup has got.
    pub fn set_output(&self, output: Readiness<Arc<dyn VoiceOutput>>) {
        self.output.set(output);
    }

    /// The output, or why there is none to speak through.
    pub fn output(&self) -> Result<Arc<dyn VoiceOutput>, NotReady> {
        self.output.get()
    }

    async fn cancel_queued(&self) {
        if let Ok(output) = self.output() {
            output.cancel().await;
        }
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
mod tests {
    use super::*;
    use crate::VoiceOutputError;
    use async_trait::async_trait;
    use parking_lot::Mutex;

    #[derive(Debug, Default)]
    struct RecordingOutput {
        spoken: Mutex<Vec<String>>,
        cancels: AtomicU64,
        wait_idles: AtomicU64,
    }

    #[async_trait]
    impl VoiceOutput for RecordingOutput {
        async fn speak(&self, text: String) -> Result<(), VoiceOutputError> {
            self.spoken.lock().push(text);
            Ok(())
        }
        async fn wait_idle(&self) -> Result<(), VoiceOutputError> {
            self.wait_idles.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        async fn cancel(&self) {
            self.cancels.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn ready_controller(output: Arc<RecordingOutput>, enabled: bool) -> Arc<VoiceOutputController> {
        let ctrl = VoiceOutputController::new(enabled);
        ctrl.set_output(Readiness::Ready(output));
        ctrl
    }

    #[tokio::test]
    async fn muting_cancels_once_and_silences_the_running_reply() {
        let inner = Arc::new(RecordingOutput::default());
        let ctrl = ready_controller(inner.clone(), true);
        let start = ctrl.current_epoch();
        ctrl.set_enabled(false).await;
        assert!(!ctrl.enabled());
        assert_eq!(inner.cancels.load(Ordering::SeqCst), 1);
        ctrl.set_enabled(false).await;
        assert_eq!(inner.cancels.load(Ordering::SeqCst), 1);
        ctrl.set_enabled(true).await;
        assert_eq!(ctrl.should_speak(start), SpeakDecision::DropForSkip);
    }

    #[tokio::test]
    async fn should_speak_decision() {
        for (enabled, skip_before_start, skip_after_start, expected) in [
            (true, false, false, SpeakDecision::Speak),
            (true, true, false, SpeakDecision::Speak),
            (true, false, true, SpeakDecision::DropForSkip),
            (false, false, false, SpeakDecision::DropSilent),
            (false, false, true, SpeakDecision::DropForSkip),
        ] {
            let ctrl = ready_controller(Arc::new(RecordingOutput::default()), enabled);
            if skip_before_start {
                ctrl.skip().await;
            }
            let start = ctrl.current_epoch();
            if skip_after_start {
                ctrl.skip().await;
            }
            assert_eq!(
                ctrl.should_speak(start),
                expected,
                "enabled={enabled} skip_before_start={skip_before_start} skip_after_start={skip_after_start}"
            );
        }
    }

    #[tokio::test]
    async fn speaking_is_raised_while_any_guard_is_alive() {
        let ctrl = ready_controller(Arc::new(RecordingOutput::default()), true);
        let mut speaking = ctrl.subscribe_speaking();
        assert!(!ctrl.is_speaking());
        assert!(!*speaking.borrow_and_update());

        let first = ctrl.begin_speaking();
        assert!(ctrl.is_speaking());
        assert!(speaking.has_changed().unwrap());
        assert!(*speaking.borrow_and_update());

        let second = ctrl.begin_speaking();
        drop(first);
        assert!(ctrl.is_speaking(), "one worker still speaking");
        assert!(!speaking.has_changed().unwrap());

        drop(second);
        assert!(!ctrl.is_speaking());
        assert!(!*speaking.borrow_and_update());
    }

    #[tokio::test]
    async fn output_starts_unready_and_skip_still_advances_the_epoch() {
        let ctrl = VoiceOutputController::new(true);
        assert!(ctrl.output().is_err());
        ctrl.skip().await;
        assert_eq!(ctrl.current_epoch(), 1);

        let inner = Arc::new(RecordingOutput::default());
        ctrl.set_output(Readiness::Ready(inner.clone()));
        ctrl.skip().await;
        assert_eq!(inner.cancels.load(Ordering::SeqCst), 1);
    }
}
