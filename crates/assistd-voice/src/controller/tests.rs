use super::*;
use crate::{NoVoiceOutput, VoiceOutputError};
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

#[tokio::test]
async fn new_starts_with_given_enabled_flag_and_zero_epoch() {
    let ctrl = VoiceOutputController::new(Arc::new(NoVoiceOutput), true);
    assert!(ctrl.enabled());
    assert_eq!(ctrl.current_epoch(), 0);

    let off = VoiceOutputController::new(Arc::new(NoVoiceOutput), false);
    assert!(!off.enabled());
}

#[tokio::test]
async fn set_enabled_false_cancels_inner_once() {
    let inner = Arc::new(RecordingOutput::default());
    let ctrl = VoiceOutputController::new(inner.clone(), true);
    ctrl.set_enabled(false).await;
    assert!(!ctrl.enabled());
    assert_eq!(inner.cancels.load(Ordering::SeqCst), 1);
    ctrl.set_enabled(false).await;
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
async fn should_speak_decision() {
    for (enabled, skip_before_start, skip_after_start, expected) in [
        (true, false, false, SpeakDecision::Speak),
        (true, true, false, SpeakDecision::Speak),
        (true, false, true, SpeakDecision::DropForSkip),
        (false, false, false, SpeakDecision::DropSilent),
        (false, false, true, SpeakDecision::DropForSkip),
    ] {
        let ctrl = VoiceOutputController::new(Arc::new(NoVoiceOutput), enabled);
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
