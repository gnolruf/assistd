//! Silero speech detection over the live mic stream: each 20 ms frame is
//! labelled speech or not, so music, noise and other non-speech sound never
//! hold an utterance open.

use std::collections::VecDeque;
use std::fmt;
use std::sync::Arc;

use assistd_config::TranscriptionConfig;
use parking_lot::Mutex;
use tracing::warn;
use whisper_rs::WhisperVadContext;

use super::vad::FRAME_SAMPLES;
use crate::transcribe::TranscriptionError;
use crate::whisper::load_silero;

/// Silero's input chunk: 32 ms at 16 kHz.
const CHUNK_SAMPLES: usize = 512;

/// Frames labelled per Silero call: 160 ms, exactly five chunks.
const STRIDE_FRAMES: usize = 8;
const STRIDE_SAMPLES: usize = STRIDE_FRAMES * FRAME_SAMPLES;
const STRIDE_CHUNKS: usize = STRIDE_SAMPLES / CHUNK_SAMPLES;

/// Audio ahead of each stride that warms Silero's state, which every call
/// resets: 512 ms, a whole number of chunks.
const CONTEXT_SAMPLES: usize = 16 * CHUNK_SAMPLES;

const _: () = assert!(
    STRIDE_SAMPLES.is_multiple_of(CHUNK_SAMPLES),
    "strides align to chunks"
);

/// Speech probability at or above which a chunk counts as speech; Silero's
/// own default.
const SPEECH_THRESHOLD: f32 = 0.5;

/// The Silero VAD model, loaded on the CPU for one stream at a time.
pub struct SileroVad {
    context: WhisperVadContext,
}

impl fmt::Debug for SileroVad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SileroVad").finish_non_exhaustive()
    }
}

impl SileroVad {
    /// Load `config.vad_model`, downloading it first if it is not cached.
    pub async fn load(config: &TranscriptionConfig) -> Result<Self, TranscriptionError> {
        Ok(Self {
            context: load_silero(config).await?,
        })
    }

    /// Speech probability of each chunk of `samples`, which must be a whole
    /// number of chunks.
    fn chunk_probabilities(&mut self, samples: &[f32]) -> Result<&[f32], TranscriptionError> {
        self.context
            .detect_speech(samples)
            .map_err(|err| TranscriptionError::WhisperInference(err.to_string()))?;
        Ok(self.context.probabilities())
    }
}

/// Gathers frames into strides and labels each one speech or not.
#[derive(Debug)]
pub(super) struct FrameClassifier {
    silero: Arc<Mutex<SileroVad>>,
    /// The last [`CONTEXT_SAMPLES`] already labelled, as Silero input.
    context: VecDeque<f32>,
    pending: Vec<[i16; FRAME_SAMPLES]>,
}

impl FrameClassifier {
    pub(super) fn new(silero: Arc<Mutex<SileroVad>>) -> Self {
        Self {
            silero,
            context: VecDeque::with_capacity(CONTEXT_SAMPLES + STRIDE_SAMPLES),
            pending: Vec::with_capacity(STRIDE_FRAMES),
        }
    }

    /// Add `frame`. Once a stride is complete, returns its frames in order,
    /// each with whether it holds speech; until then, nothing.
    pub(super) fn push(
        &mut self,
        frame: &[i16; FRAME_SAMPLES],
    ) -> Vec<([i16; FRAME_SAMPLES], bool)> {
        self.context
            .extend(frame.iter().map(|&sample| f32::from(sample) / 32_768.0));
        self.pending.push(*frame);
        if self.pending.len() < STRIDE_FRAMES {
            return Vec::new();
        }
        let verdicts = self.label_stride();
        let excess = self.context.len().saturating_sub(CONTEXT_SAMPLES);
        self.context.drain(..excess);
        std::mem::take(&mut self.pending)
            .into_iter()
            .zip(verdicts)
            .collect()
    }

    /// Run Silero over the context and the stride that ends it; a failed
    /// run labels the stride silent.
    fn label_stride(&mut self) -> [bool; STRIDE_FRAMES] {
        let window = self.context.make_contiguous();
        let mut silero = self.silero.lock();
        match silero.chunk_probabilities(window) {
            Ok(probs) => frame_verdicts(&probs[probs.len().saturating_sub(STRIDE_CHUNKS)..]),
            Err(err) => {
                warn!(target: "assistd::voice::listen", "silero speech detection failed: {err}");
                [false; STRIDE_FRAMES]
            }
        }
    }
}

/// Each stride frame's verdict from the stride's chunk probabilities: a
/// frame takes the chunk holding its midpoint.
fn frame_verdicts(chunk_probs: &[f32]) -> [bool; STRIDE_FRAMES] {
    std::array::from_fn(|frame| {
        let midpoint = frame * FRAME_SAMPLES + FRAME_SAMPLES / 2;
        chunk_probs
            .get(midpoint / CHUNK_SAMPLES)
            .is_some_and(|&prob| prob >= SPEECH_THRESHOLD)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_frame_takes_the_chunk_under_its_midpoint() {
        let only_third_chunk_is_speech = [0.1, 0.2, 0.9, 0.3, 0.4];
        assert_eq!(
            frame_verdicts(&only_third_chunk_is_speech),
            [false, false, false, true, true, false, false, false],
            "chunk 2 spans samples 1024..1536, holding the midpoints of frames 3 and 4"
        );
    }
}
