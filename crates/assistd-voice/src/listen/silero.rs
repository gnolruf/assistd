//! Silero speech detection over the live mic stream: each 20 ms frame is
//! labelled speech or not, so music, noise and other non-speech sound never
//! hold an utterance open.

use std::sync::Arc;

use assistd_config::TranscriptionConfig;
use parking_lot::Mutex;

use super::vad::FRAME_SAMPLES;
use crate::transcribe::TranscriptionError;
use crate::vad::{SpeechDetector, WINDOW_SAMPLES, ensure_model};

/// Frames labelled per Silero pass: 160 ms, exactly five windows.
const STRIDE_FRAMES: usize = 8;
const STRIDE_SAMPLES: usize = STRIDE_FRAMES * FRAME_SAMPLES;

const _: () = assert!(
    STRIDE_SAMPLES.is_multiple_of(WINDOW_SAMPLES),
    "strides align to windows"
);

/// The Silero VAD model, streaming one mic session at a time.
#[derive(Debug)]
pub struct SileroVad {
    detector: SpeechDetector,
}

impl SileroVad {
    /// Load `config.vad_model`, downloading it first if it is not cached.
    pub async fn load(config: &TranscriptionConfig) -> Result<Self, TranscriptionError> {
        let model = ensure_model(config).await?;
        let detector =
            tokio::task::spawn_blocking(move || SpeechDetector::load(&model, 0.0, 0.0)).await??;
        Ok(Self { detector })
    }
}

/// Gathers frames into strides and labels each one speech or not.
#[derive(Debug)]
pub(super) struct FrameClassifier {
    silero: Arc<Mutex<SileroVad>>,
    pending: Vec<[i16; FRAME_SAMPLES]>,
}

impl FrameClassifier {
    /// A classifier for a new mic session; Silero forgets the last one.
    pub(super) fn new(silero: Arc<Mutex<SileroVad>>) -> Self {
        silero.lock().detector.reset();
        Self {
            silero,
            pending: Vec::with_capacity(STRIDE_FRAMES),
        }
    }

    /// Add `frame`. Once a stride is complete, returns its frames in order,
    /// each with whether it holds speech; until then, nothing.
    pub(super) fn push(
        &mut self,
        frame: &[i16; FRAME_SAMPLES],
    ) -> Vec<([i16; FRAME_SAMPLES], bool)> {
        self.pending.push(*frame);
        if self.pending.len() < STRIDE_FRAMES {
            return Vec::new();
        }
        let verdicts = self.label_stride();
        std::mem::take(&mut self.pending)
            .into_iter()
            .zip(verdicts)
            .collect()
    }

    fn label_stride(&self) -> [bool; STRIDE_FRAMES] {
        let samples: Vec<f32> = self
            .pending
            .iter()
            .flatten()
            .map(|&sample| f32::from(sample) / 32_768.0)
            .collect();
        let (windows, _) = samples.as_chunks::<WINDOW_SAMPLES>();
        let mut silero = self.silero.lock();
        let window_speech: Vec<bool> = windows
            .iter()
            .map(|window| silero.detector.push_window(window))
            .collect();
        frame_verdicts(&window_speech)
    }
}

/// Each stride frame's verdict from the stride's window verdicts: a frame
/// takes the window holding its midpoint.
fn frame_verdicts(window_speech: &[bool]) -> [bool; STRIDE_FRAMES] {
    std::array::from_fn(|frame| {
        let midpoint = frame * FRAME_SAMPLES + FRAME_SAMPLES / 2;
        window_speech
            .get(midpoint / WINDOW_SAMPLES)
            .copied()
            .unwrap_or(false)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_frame_takes_the_window_under_its_midpoint() {
        let only_third_window_is_speech = [false, false, true, false, false];
        assert_eq!(
            frame_verdicts(&only_third_window_is_speech),
            [false, false, false, true, true, false, false, false],
            "window 2 spans samples 1024..1536, holding the midpoints of frames 3 and 4"
        );
    }
}
