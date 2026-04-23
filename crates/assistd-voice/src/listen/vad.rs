//! VAD-driven utterance segmentation.
//!
//! [`UtteranceVad`] consumes fixed-size 16 kHz mono i16 frames (20 ms
//! each by default) from the streaming mic consumer, runs webrtc-vad
//! on each frame, and emits completed utterances back to the listener
//! task. A rolling pre-roll ring prepends a few hundred milliseconds
//! of audio to each utterance so the first syllable isn't clipped
//! between onset confirmation and buffer start.
//!
//! State machine (terse form):
//!
//! ```text
//!                +--- voiced frame ---+
//!                |                    v
//!   Silent ---> PreVoice ---> Voiced ---> Trailing
//!     ^            |            |            |
//!     |            |            |            v
//!     +---- silent frame or cancel ----- Silent
//! ```
//!
//! Onset requires `onset_confirm_frames` consecutive voiced frames
//! (guards against a single keystroke click tripping the whole
//! pipeline). Offset requires `offset_frames` consecutive silent
//! frames (so a natural mid-word pause doesn't cut words off).
//! Utterances shorter than `min_utterance_frames` are dropped;
//! utterances longer than `max_utterance_frames` are force-flushed.

use std::collections::VecDeque;

use webrtc_vad::{SampleRate, Vad, VadMode};

/// Sample rate used throughout the listen pipeline. Must match the
/// resampler output and webrtc-vad's accepted rates.
pub const SAMPLE_RATE_HZ: u32 = 16_000;

/// Frame size in samples at 16 kHz mono = 20 ms. webrtc-vad supports
/// exactly {10, 20, 30} ms at 16 kHz; 20 ms is a good balance between
/// responsiveness and CPU cost.
pub const FRAME_SAMPLES: usize = 320;

/// Knobs for the VAD state machine, derived from `ContinuousListenConfig`
/// at construction time (converted from millisecond/second configs to
/// whole-frame counts so the hot path does no division).
#[derive(Debug, Clone, Copy)]
pub struct VadTuning {
    /// Consecutive voiced frames required to confirm onset.
    pub onset_confirm_frames: u32,
    /// Consecutive silent frames required to confirm end of utterance.
    pub offset_frames: u32,
    /// Drop utterances shorter than this many frames.
    pub min_utterance_frames: u32,
    /// Force-flush after this many frames even if voice continues.
    pub max_utterance_frames: u32,
    /// Pre-roll ring size, in frames.
    pub preroll_frames: u32,
    /// webrtc-vad aggressiveness, mapped to `VadMode` at init.
    pub aggressiveness: u8,
}

impl VadTuning {
    pub fn from_ms(
        silence_ms: u32,
        min_utterance_ms: u32,
        max_utterance_secs: u32,
        preroll_ms: u32,
        onset_confirm_ms: u32,
        aggressiveness: u8,
    ) -> Self {
        let frame_ms = 20u32;
        Self {
            onset_confirm_frames: onset_confirm_ms.div_ceil(frame_ms).max(1),
            offset_frames: silence_ms.div_ceil(frame_ms).max(1),
            min_utterance_frames: min_utterance_ms.div_ceil(frame_ms).max(1),
            max_utterance_frames: max_utterance_secs
                .saturating_mul(1000)
                .div_ceil(frame_ms)
                .max(1),
            preroll_frames: preroll_ms.div_ceil(frame_ms),
            aggressiveness: aggressiveness.min(3),
        }
    }
}

/// Output from [`UtteranceVad::feed`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VadEvent {
    /// A complete utterance bounded by confirmed silence. The buffer
    /// includes the pre-roll plus all voiced + intra-utterance silent
    /// frames up to the trailing silence that terminated it.
    UtteranceComplete(Vec<i16>),
    /// `max_utterance_frames` exceeded; flushing whatever was buffered.
    /// Caller may immediately continue recording into the next
    /// utterance — internal state resets to `Silent` on emit.
    Truncated(Vec<i16>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Silent,
    PreVoice { voiced: u32 },
    Voiced,
    Trailing { silent: u32 },
}

pub struct UtteranceVad {
    vad: Vad,
    tuning: VadTuning,
    state: State,
    preroll: VecDeque<[i16; FRAME_SAMPLES]>,
    utterance: Vec<i16>,
    utterance_frames: u32,
}

impl UtteranceVad {
    pub fn new(tuning: VadTuning) -> Self {
        let mode = match tuning.aggressiveness {
            0 => VadMode::Quality,
            1 => VadMode::LowBitrate,
            2 => VadMode::Aggressive,
            _ => VadMode::VeryAggressive,
        };
        let vad = Vad::new_with_rate_and_mode(SampleRate::Rate16kHz, mode);
        let preroll_cap = tuning.preroll_frames as usize;
        Self {
            vad,
            tuning,
            state: State::Silent,
            preroll: VecDeque::with_capacity(preroll_cap),
            utterance: Vec::with_capacity((tuning.max_utterance_frames as usize) * FRAME_SAMPLES),
            utterance_frames: 0,
        }
    }

    /// Feed one 20-ms frame. Returns `Some(event)` when an utterance
    /// boundary is crossed, `None` otherwise. Uses webrtc-vad to
    /// classify the frame — see [`Self::feed_decided`] for the
    /// state-machine-only variant used in tests.
    pub fn feed(&mut self, frame: &[i16; FRAME_SAMPLES]) -> Option<VadEvent> {
        let is_voiced = self.vad.is_voice_segment(frame).unwrap_or(false);
        self.feed_decided(frame, is_voiced)
    }

    /// Feed one 20-ms frame with an explicit voiced/silent decision.
    /// Bypasses webrtc-vad so unit tests can assert purely on the
    /// state-machine transitions without depending on VAD accuracy on
    /// synthetic audio.
    pub fn feed_decided(
        &mut self,
        frame: &[i16; FRAME_SAMPLES],
        is_voiced: bool,
    ) -> Option<VadEvent> {
        match self.state {
            State::Silent => {
                self.push_preroll(frame);
                if is_voiced {
                    self.state = if self.tuning.onset_confirm_frames <= 1 {
                        self.begin_utterance();
                        self.append_frame(frame);
                        State::Voiced
                    } else {
                        State::PreVoice { voiced: 1 }
                    };
                }
                None
            }
            State::PreVoice { voiced } => {
                if is_voiced {
                    let confirmed = voiced + 1;
                    if confirmed >= self.tuning.onset_confirm_frames {
                        self.begin_utterance();
                        // Include the confirming voiced frames that
                        // were only in preroll until now: begin_utterance
                        // already drained the preroll into the utterance,
                        // so only the current frame still needs appending.
                        self.append_frame(frame);
                        self.state = State::Voiced;
                    } else {
                        self.push_preroll(frame);
                        self.state = State::PreVoice { voiced: confirmed };
                    }
                    None
                } else {
                    // Onset candidate fell through — drop back to silent.
                    self.push_preroll(frame);
                    self.state = State::Silent;
                    None
                }
            }
            State::Voiced => {
                self.append_frame(frame);
                if self.utterance_frames >= self.tuning.max_utterance_frames {
                    return Some(self.flush_truncated());
                }
                if !is_voiced {
                    self.state = State::Trailing { silent: 1 };
                    if self.tuning.offset_frames <= 1 {
                        return self.finish_utterance();
                    }
                }
                None
            }
            State::Trailing { silent } => {
                self.append_frame(frame);
                if self.utterance_frames >= self.tuning.max_utterance_frames {
                    return Some(self.flush_truncated());
                }
                if is_voiced {
                    self.state = State::Voiced;
                    None
                } else {
                    let silent = silent + 1;
                    if silent >= self.tuning.offset_frames {
                        self.finish_utterance()
                    } else {
                        self.state = State::Trailing { silent };
                        None
                    }
                }
            }
        }
    }

    fn push_preroll(&mut self, frame: &[i16; FRAME_SAMPLES]) {
        if self.tuning.preroll_frames == 0 {
            return;
        }
        if self.preroll.len() == self.tuning.preroll_frames as usize {
            self.preroll.pop_front();
        }
        self.preroll.push_back(*frame);
    }

    fn begin_utterance(&mut self) {
        self.utterance.clear();
        self.utterance_frames = 0;
        for frame in self.preroll.drain(..) {
            self.utterance.extend_from_slice(&frame);
            self.utterance_frames = self.utterance_frames.saturating_add(1);
        }
    }

    fn append_frame(&mut self, frame: &[i16; FRAME_SAMPLES]) {
        self.utterance.extend_from_slice(frame);
        self.utterance_frames = self.utterance_frames.saturating_add(1);
    }

    fn finish_utterance(&mut self) -> Option<VadEvent> {
        self.state = State::Silent;
        let frames = self.utterance_frames;
        let pcm = std::mem::take(&mut self.utterance);
        self.utterance_frames = 0;
        if frames < self.tuning.min_utterance_frames {
            return None;
        }
        Some(VadEvent::UtteranceComplete(pcm))
    }

    fn flush_truncated(&mut self) -> VadEvent {
        self.state = State::Silent;
        let pcm = std::mem::take(&mut self.utterance);
        self.utterance_frames = 0;
        VadEvent::Truncated(pcm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn silent_frame() -> [i16; FRAME_SAMPLES] {
        [0; FRAME_SAMPLES]
    }

    /// Dummy non-zero frame; contents don't matter because the
    /// state-machine tests pass the voiced/silent decision explicitly
    /// via [`UtteranceVad::feed_decided`].
    fn voiced_frame() -> [i16; FRAME_SAMPLES] {
        [1000; FRAME_SAMPLES]
    }

    fn tight_tuning() -> VadTuning {
        // 1-frame onset, 2-frame offset, 1-frame min, 50-frame max, 3-frame preroll.
        VadTuning {
            onset_confirm_frames: 1,
            offset_frames: 2,
            min_utterance_frames: 1,
            max_utterance_frames: 50,
            preroll_frames: 3,
            aggressiveness: 3,
        }
    }

    #[test]
    fn silent_input_produces_no_events() {
        let mut v = UtteranceVad::new(tight_tuning());
        let s = silent_frame();
        for _ in 0..100 {
            assert!(v.feed_decided(&s, false).is_none());
        }
    }

    #[test]
    fn voiced_burst_bounded_by_silence_emits_one_utterance() {
        let mut v = UtteranceVad::new(tight_tuning());
        let s = silent_frame();
        let voiced = voiced_frame();

        // Fill preroll with silence.
        for _ in 0..5 {
            assert_eq!(v.feed_decided(&s, false), None);
        }
        // Speak for 10 frames (~200 ms).
        let mut events = Vec::new();
        for _ in 0..10 {
            if let Some(e) = v.feed_decided(&voiced, true) {
                events.push(e);
            }
        }
        // Silence ends the utterance.
        for _ in 0..5 {
            if let Some(e) = v.feed_decided(&s, false) {
                events.push(e);
            }
        }
        assert_eq!(events.len(), 1, "expected exactly one utterance");
        match &events[0] {
            VadEvent::UtteranceComplete(pcm) => {
                // Should include pre-roll (3 frames) + 10 voiced + 2 trailing silent.
                // Each frame is 320 samples at 16 kHz.
                let expected_min = 10 * FRAME_SAMPLES;
                assert!(
                    pcm.len() >= expected_min,
                    "pcm length {} < expected min {expected_min}",
                    pcm.len()
                );
            }
            other => panic!("expected UtteranceComplete, got {other:?}"),
        }
    }

    #[test]
    fn utterance_below_min_is_dropped() {
        let mut cfg = tight_tuning();
        cfg.min_utterance_frames = 20;
        let mut v = UtteranceVad::new(cfg);

        let s = silent_frame();
        let voiced = voiced_frame();

        // Fill preroll; very short burst of 2 frames only.
        for _ in 0..5 {
            v.feed_decided(&s, false);
        }
        for _ in 0..2 {
            v.feed_decided(&voiced, true);
        }
        // End with enough silence to confirm offset.
        let mut events = Vec::new();
        for _ in 0..10 {
            if let Some(e) = v.feed_decided(&s, false) {
                events.push(e);
            }
        }
        assert!(
            events.is_empty(),
            "short burst should be dropped, got {events:?}"
        );
    }

    #[test]
    fn continuous_voiced_input_force_flushes_at_max() {
        let mut cfg = tight_tuning();
        cfg.max_utterance_frames = 10;
        cfg.offset_frames = 100; // never hit via silence in this test
        let mut v = UtteranceVad::new(cfg);

        let voiced = voiced_frame();
        let mut events = Vec::new();
        for _ in 0..25 {
            if let Some(e) = v.feed_decided(&voiced, true) {
                events.push(e);
            }
        }
        assert!(!events.is_empty(), "expected at least one Truncated event");
        assert!(
            matches!(events[0], VadEvent::Truncated(_)),
            "expected Truncated first, got {:?}",
            events[0]
        );
    }

    #[test]
    fn onset_requires_multiple_confirmed_frames() {
        // Single-frame voiced burst followed by silence should not
        // start an utterance when onset_confirm_frames > 1.
        let mut cfg = tight_tuning();
        cfg.onset_confirm_frames = 3;
        let mut v = UtteranceVad::new(cfg);

        let s = silent_frame();
        let voiced = voiced_frame();

        // Preroll + single voiced blip + back to silence.
        for _ in 0..5 {
            v.feed_decided(&s, false);
        }
        v.feed_decided(&voiced, true);
        for _ in 0..10 {
            assert!(v.feed_decided(&s, false).is_none());
        }
        // No utterance should have been emitted.
    }

    #[test]
    fn vad_tuning_from_ms_rounds_up() {
        let t = VadTuning::from_ms(800, 400, 30, 300, 60, 3);
        assert_eq!(t.offset_frames, 40); // 800 / 20
        assert_eq!(t.min_utterance_frames, 20); // 400 / 20
        assert_eq!(t.max_utterance_frames, 1500); // 30000 / 20
        assert_eq!(t.preroll_frames, 15); // 300 / 20
        assert_eq!(t.onset_confirm_frames, 3); // 60 / 20
        assert_eq!(t.aggressiveness, 3);
    }

    #[test]
    fn vad_tuning_clamps_aggressiveness() {
        let t = VadTuning::from_ms(800, 400, 30, 300, 60, 99);
        assert_eq!(t.aggressiveness, 3);
    }
}
