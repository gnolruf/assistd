//! VAD-driven utterance segmentation over 20 ms frames.

use std::collections::VecDeque;

use webrtc_vad::{SampleRate, Vad, VadMode};

/// Sample rate of every frame fed to the VAD.
pub const SAMPLE_RATE_HZ: u32 = 16_000;

/// 20 ms at 16 kHz. webrtc-vad accepts exactly 10, 20, or 30 ms frames.
pub const FRAME_SAMPLES: usize = 320;

const FRAME_MS: u32 = 20;
const MIN_UTTERANCE_MS: u32 = 400;
const PREROLL_MS: u32 = 300;
const ONSET_CONFIRM_MS: u32 = 60;
/// webrtc-vad's most selective mode; lower modes admit keyboard and
/// fan noise on a desktop mic.
const AGGRESSIVENESS: u8 = 3;

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

impl VadTuning {
    /// Convert the two configurable durations to frame counts.
    pub fn from_ms(silence_ms: u32, max_utterance_secs: u32) -> Self {
        Self {
            onset_confirm_frames: ONSET_CONFIRM_MS.div_ceil(FRAME_MS).max(1),
            offset_frames: silence_ms.div_ceil(FRAME_MS).max(1),
            min_utterance_frames: MIN_UTTERANCE_MS.div_ceil(FRAME_MS).max(1),
            max_utterance_frames: max_utterance_secs
                .saturating_mul(1000)
                .div_ceil(FRAME_MS)
                .max(1),
            preroll_frames: PREROLL_MS.div_ceil(FRAME_MS),
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
enum Phase {
    Silent,
    PreVoice { voiced: u32 },
    Voiced,
    Trailing { silent: u32 },
}

/// Classifies 20 ms frames via webrtc-vad and emits utterances bounded by
/// confirmed silence. Onset and offset each need consecutive frames per
/// [`VadTuning`]; a pre-roll ring is prepended so the first syllable is not clipped.
pub struct UtteranceVad {
    vad: Vad,
    tuning: VadTuning,
    phase: Phase,
    preroll: VecDeque<[i16; FRAME_SAMPLES]>,
    utterance: Vec<i16>,
    utterance_frames: u32,
}

impl UtteranceVad {
    /// A segmenter in the silent state. An `aggressiveness` above 3 is
    /// treated as 3.
    pub fn new(tuning: VadTuning) -> Self {
        let vad =
            Vad::new_with_rate_and_mode(SampleRate::Rate16kHz, vad_mode(tuning.aggressiveness));
        Self {
            vad,
            tuning,
            phase: Phase::Silent,
            preroll: VecDeque::with_capacity(tuning.preroll_frames as usize),
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
        match self.phase {
            Phase::Silent => {
                self.feed_silent(frame, is_voiced);
                None
            }
            Phase::PreVoice { voiced } => {
                self.feed_pre_voice(frame, is_voiced, voiced);
                None
            }
            Phase::Voiced => self.feed_voiced(frame, is_voiced),
            Phase::Trailing { silent } => self.feed_trailing(frame, is_voiced, silent),
        }
    }

    fn feed_silent(&mut self, frame: &[i16; FRAME_SAMPLES], is_voiced: bool) {
        self.push_preroll(frame);
        if !is_voiced {
            return;
        }
        self.phase = if self.tuning.onset_confirm_frames <= 1 {
            self.begin_utterance();
            self.append_frame(frame);
            Phase::Voiced
        } else {
            Phase::PreVoice { voiced: 1 }
        };
    }

    fn feed_pre_voice(&mut self, frame: &[i16; FRAME_SAMPLES], is_voiced: bool, voiced: u32) {
        if !is_voiced {
            self.push_preroll(frame);
            self.phase = Phase::Silent;
            return;
        }
        let confirmed = voiced + 1;
        if confirmed >= self.tuning.onset_confirm_frames {
            self.begin_utterance();
            self.append_frame(frame);
            self.phase = Phase::Voiced;
        } else {
            self.push_preroll(frame);
            self.phase = Phase::PreVoice { voiced: confirmed };
        }
    }

    fn feed_voiced(&mut self, frame: &[i16; FRAME_SAMPLES], is_voiced: bool) -> Option<VadEvent> {
        self.append_frame(frame);
        if self.utterance_frames >= self.tuning.max_utterance_frames {
            return Some(self.flush_truncated());
        }
        if !is_voiced {
            self.phase = Phase::Trailing { silent: 1 };
            if self.tuning.offset_frames <= 1 {
                return self.finish_utterance();
            }
        }
        None
    }

    fn feed_trailing(
        &mut self,
        frame: &[i16; FRAME_SAMPLES],
        is_voiced: bool,
        silent: u32,
    ) -> Option<VadEvent> {
        self.append_frame(frame);
        if self.utterance_frames >= self.tuning.max_utterance_frames {
            return Some(self.flush_truncated());
        }
        if is_voiced {
            self.phase = Phase::Voiced;
            return None;
        }
        let silent = silent + 1;
        if silent >= self.tuning.offset_frames {
            self.finish_utterance()
        } else {
            self.phase = Phase::Trailing { silent };
            None
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
        self.phase = Phase::Silent;
        let frames = self.utterance_frames;
        let pcm = std::mem::take(&mut self.utterance);
        self.utterance_frames = 0;
        if frames < self.tuning.min_utterance_frames {
            return None;
        }
        Some(VadEvent::UtteranceComplete(pcm))
    }

    fn flush_truncated(&mut self) -> VadEvent {
        self.phase = Phase::Silent;
        let pcm = std::mem::take(&mut self.utterance);
        self.utterance_frames = 0;
        VadEvent::Truncated(pcm)
    }
}

fn vad_mode(aggressiveness: u8) -> VadMode {
    match aggressiveness {
        0 => VadMode::Quality,
        1 => VadMode::LowBitrate,
        2 => VadMode::Aggressive,
        _ => VadMode::VeryAggressive,
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

    /// Feed `count` identical frames, collecting any events.
    fn feed(vad: &mut UtteranceVad, voiced: bool, count: usize) -> Vec<VadEvent> {
        let frame = [if voiced { VOICED } else { SILENT }; FRAME_SAMPLES];
        (0..count)
            .filter_map(|_| vad.feed_decided(&frame, voiced))
            .collect()
    }

    fn frames(sample: i16, count: usize) -> Vec<i16> {
        vec![sample; count * FRAME_SAMPLES]
    }

    #[test]
    fn silent_input_produces_no_events() {
        let mut vad = UtteranceVad::new(tight_tuning());
        assert_eq!(feed(&mut vad, false, 100), NO_EVENTS);
    }

    #[test]
    fn voiced_burst_bounded_by_silence_emits_one_utterance() {
        let mut vad = UtteranceVad::new(tight_tuning());
        assert_eq!(feed(&mut vad, false, 5), NO_EVENTS);
        assert_eq!(feed(&mut vad, true, 10), NO_EVENTS);
        let events = feed(&mut vad, false, 5);

        let expected = [frames(SILENT, 2), frames(VOICED, 10), frames(SILENT, 2)].concat();
        assert_eq!(
            events,
            [VadEvent::UtteranceComplete(expected)],
            "pre-roll ends at the first voiced frame; the confirming silence is kept"
        );
    }

    #[test]
    fn utterance_below_min_is_dropped() {
        let mut vad = UtteranceVad::new(VadTuning {
            min_utterance_frames: 20,
            ..tight_tuning()
        });
        feed(&mut vad, false, 5);
        feed(&mut vad, true, 2);
        assert_eq!(feed(&mut vad, false, 10), NO_EVENTS);
    }

    #[test]
    fn continuous_voiced_input_force_flushes_at_max() {
        let mut vad = UtteranceVad::new(VadTuning {
            max_utterance_frames: 10,
            offset_frames: 100,
            ..tight_tuning()
        });
        let truncated = VadEvent::Truncated(frames(VOICED, 10));
        assert_eq!(feed(&mut vad, true, 25), [truncated.clone(), truncated]);
    }

    #[test]
    fn onset_requires_multiple_confirmed_frames() {
        let mut vad = UtteranceVad::new(VadTuning {
            onset_confirm_frames: 3,
            ..tight_tuning()
        });
        feed(&mut vad, false, 5);
        feed(&mut vad, true, 2);
        assert_eq!(feed(&mut vad, false, 10), NO_EVENTS);
    }

    #[test]
    fn vad_tuning_from_ms_rounds_up_to_whole_frames() {
        for (silence_ms, max_secs, offset_frames, max_utterance_frames) in
            [(800, 30, 40, 1500), (810, 30, 41, 1500), (1, 0, 1, 1)]
        {
            let tuning = VadTuning::from_ms(silence_ms, max_secs);
            assert_eq!(
                (tuning.offset_frames, tuning.max_utterance_frames),
                (offset_frames, max_utterance_frames),
                "from_ms({silence_ms}, {max_secs})"
            );
        }
    }
}
