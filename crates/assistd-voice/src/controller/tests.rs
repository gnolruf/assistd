use super::*;
use crate::{NoVoiceOutput, VoiceOutputError};
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

#[tokio::test]
async fn set_enabled_false_cancels_inner_once() {
    let inner = Arc::new(RecordingOutput::default());
    let ctrl = VoiceOutputController::ready(inner.clone(), true);
    ctrl.set_enabled(false).await;
    assert!(!ctrl.enabled());
    assert_eq!(inner.cancels.load(Ordering::SeqCst), 1);
    ctrl.set_enabled(false).await;
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
        let ctrl = VoiceOutputController::ready(Arc::new(NoVoiceOutput), enabled);
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
    let ctrl = VoiceOutputController::ready(Arc::new(NoVoiceOutput), true);
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
