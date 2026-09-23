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

    const SILENT: i16 = 0;
    const VOICED: i16 = 1000;
    const NO_EVENTS: [VadEvent; 0] = [];

    fn tight_tuning() -> VadTuning {
        VadTuning {
            onset_confirm_frames: 2,
            offset_frames: 2,
            min_utterance_frames: 1,
            max_utterance_frames: 50,
            preroll_frames: 3,
            aggressiveness: 3,
        }
    }

    /// Feed `n` identical frames, collecting any events.
    fn feed(v: &mut UtteranceVad, voiced: bool, n: usize) -> Vec<VadEvent> {
        let frame = [if voiced { VOICED } else { SILENT }; FRAME_SAMPLES];
        (0..n)
            .filter_map(|_| v.feed_decided(&frame, voiced))
            .collect()
    }

    fn frames(sample: i16, n: usize) -> Vec<i16> {
        vec![sample; n * FRAME_SAMPLES]
    }

    #[test]
    fn silent_input_produces_no_events() {
        let mut v = UtteranceVad::new(tight_tuning());
        assert_eq!(feed(&mut v, false, 100), NO_EVENTS);
    }

    #[test]
    fn voiced_burst_bounded_by_silence_emits_one_utterance() {
        let mut v = UtteranceVad::new(tight_tuning());
        assert_eq!(feed(&mut v, false, 5), NO_EVENTS);
        assert_eq!(feed(&mut v, true, 10), NO_EVENTS);
        let events = feed(&mut v, false, 5);

        // Pre-roll holds three frames, the last of which is the first
        // voiced frame; the trailing silence that confirmed the offset
        // is kept.
        let expected = [frames(SILENT, 2), frames(VOICED, 10), frames(SILENT, 2)].concat();
        assert_eq!(events, [VadEvent::UtteranceComplete(expected)]);
    }

    #[test]
    fn utterance_below_min_is_dropped() {
        let mut v = UtteranceVad::new(VadTuning {
            min_utterance_frames: 20,
            ..tight_tuning()
        });
        feed(&mut v, false, 5);
        feed(&mut v, true, 2);
        assert_eq!(feed(&mut v, false, 10), NO_EVENTS);
    }

    #[test]
    fn continuous_voiced_input_force_flushes_at_max() {
        let mut v = UtteranceVad::new(VadTuning {
            max_utterance_frames: 10,
            offset_frames: 100,
            ..tight_tuning()
        });
        let truncated = VadEvent::Truncated(frames(VOICED, 10));
        assert_eq!(feed(&mut v, true, 25), [truncated.clone(), truncated]);
    }

    #[test]
    fn onset_requires_multiple_confirmed_frames() {
        let mut v = UtteranceVad::new(VadTuning {
            onset_confirm_frames: 3,
            ..tight_tuning()
        });
        feed(&mut v, false, 5);
        feed(&mut v, true, 2);
        assert_eq!(feed(&mut v, false, 10), NO_EVENTS);
    }

    #[test]
    fn vad_tuning_from_ms_rounds_up_to_whole_frames() {
        for (silence_ms, max_secs, offset_frames, max_utterance_frames) in
            [(800, 30, 40, 1500), (810, 30, 41, 1500), (1, 0, 1, 1)]
        {
            let t = VadTuning::from_ms(silence_ms, max_secs);
            assert_eq!(
                (t.offset_frames, t.max_utterance_frames),
                (offset_frames, max_utterance_frames),
                "from_ms({silence_ms}, {max_secs})"
            );
        }
    }
}
