//! VAD-driven utterance segmentation over 20 ms frames.
//!
//! ```text
//!   Silent ---> PreVoice ---> Voiced ---> Trailing ---> Silent
//! ```
//!
//! Onset needs `onset_confirm_frames` consecutive voiced frames so a
//! keystroke click doesn't start an utterance; offset needs
//! `offset_frames` consecutive silent frames so a mid-word pause
//! doesn't end one. A pre-roll ring is prepended to each utterance
//! so the first syllable isn't clipped.

use std::collections::VecDeque;

use webrtc_vad::{SampleRate, Vad, VadMode};

pub const SAMPLE_RATE_HZ: u32 = 16_000;

/// 20 ms at 16 kHz. webrtc-vad accepts exactly 10, 20, or 30 ms frames.
pub const FRAME_SAMPLES: usize = 320;

/// VAD state-machine thresholds, in whole frames.
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

const MIN_UTTERANCE_MS: u32 = 400;
const PREROLL_MS: u32 = 300;
const ONSET_CONFIRM_MS: u32 = 60;
/// webrtc-vad's most selective mode; lower modes admit keyboard and
/// fan noise on a desktop mic.
const AGGRESSIVENESS: u8 = 3;

impl VadTuning {
    /// Convert the two configurable durations to frame counts.
    pub fn from_ms(silence_ms: u32, max_utterance_secs: u32) -> Self {
        let frame_ms = 20u32;
        Self {
            onset_confirm_frames: ONSET_CONFIRM_MS.div_ceil(frame_ms).max(1),
            offset_frames: silence_ms.div_ceil(frame_ms).max(1),
            min_utterance_frames: MIN_UTTERANCE_MS.div_ceil(frame_ms).max(1),
            max_utterance_frames: max_utterance_secs
                .saturating_mul(1000)
                .div_ceil(frame_ms)
                .max(1),
            preroll_frames: PREROLL_MS.div_ceil(frame_ms),
            aggressiveness: AGGRESSIVENESS,
        }
    }
}

/// Output from [`UtteranceVad::feed`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VadEvent {
    /// An utterance bounded by confirmed silence, including pre-roll
    /// and the trailing silence.
    UtteranceComplete(Vec<i16>),
    /// `max_utterance_frames` exceeded; the buffer so far.
    Truncated(Vec<i16>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Silent,
    PreVoice { voiced: u32 },
    Voiced,
    Trailing { silent: u32 },
}

/// Classifies 20 ms frames via webrtc-vad and emits utterances bounded
/// by confirmed silence.
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

    /// Feed one 20 ms frame; `Some` when an utterance boundary is crossed.
    pub fn feed(&mut self, frame: &[i16; FRAME_SAMPLES]) -> Option<VadEvent> {
        let is_voiced = self.vad.is_voice_segment(frame).unwrap_or(false);
        self.feed_decided(frame, is_voiced)
    }

    /// [`feed`](Self::feed) with the voiced/silent decision supplied by
    /// the caller instead of webrtc-vad.
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
                        self.append_frame(frame);
                        self.state = State::Voiced;
                    } else {
                        self.push_preroll(frame);
                        self.state = State::PreVoice { voiced: confirmed };
                    }
                    None
                } else {
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

    fn voiced_frame() -> [i16; FRAME_SAMPLES] {
        [1000; FRAME_SAMPLES]
    }

    fn tight_tuning() -> VadTuning {
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

        for _ in 0..5 {
            assert_eq!(v.feed_decided(&s, false), None);
        }
        let mut events = Vec::new();
        for _ in 0..10 {
            if let Some(e) = v.feed_decided(&voiced, true) {
                events.push(e);
            }
        }
        for _ in 0..5 {
            if let Some(e) = v.feed_decided(&s, false) {
                events.push(e);
            }
        }
        assert_eq!(events.len(), 1, "expected exactly one utterance");
        match &events[0] {
            VadEvent::UtteranceComplete(pcm) => {
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

        for _ in 0..5 {
            v.feed_decided(&s, false);
        }
        for _ in 0..2 {
            v.feed_decided(&voiced, true);
        }
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
        cfg.offset_frames = 100;
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
        let mut cfg = tight_tuning();
        cfg.onset_confirm_frames = 3;
        let mut v = UtteranceVad::new(cfg);

        let s = silent_frame();
        let voiced = voiced_frame();

        for _ in 0..5 {
            v.feed_decided(&s, false);
        }
        v.feed_decided(&voiced, true);
        for _ in 0..10 {
            assert!(v.feed_decided(&s, false).is_none());
        }
    }

    #[test]
    fn vad_tuning_from_ms_rounds_up() {
        let t = VadTuning::from_ms(800, 30);
        assert_eq!(t.offset_frames, 40);
        assert_eq!(t.max_utterance_frames, 1500);
        assert_eq!(t.min_utterance_frames, MIN_UTTERANCE_MS.div_ceil(20));
        assert_eq!(t.preroll_frames, PREROLL_MS.div_ceil(20));
        assert_eq!(t.onset_confirm_frames, ONSET_CONFIRM_MS.div_ceil(20));
        assert_eq!(t.aggressiveness, AGGRESSIVENESS);
    }

    #[test]
    fn vad_tuning_never_yields_a_zero_frame_window() {
        let t = VadTuning::from_ms(1, 0);
        assert_eq!(t.offset_frames, 1);
        assert_eq!(t.max_utterance_frames, 1);
        assert!(t.min_utterance_frames >= 1);
        assert!(t.onset_confirm_frames >= 1);
    }
}
