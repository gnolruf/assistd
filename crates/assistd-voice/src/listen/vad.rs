//! Utterance segmentation over 20 ms frames already labelled speech or not.

use std::collections::VecDeque;
use std::fmt;

/// Sample rate of every frame fed to the VAD.
pub const SAMPLE_RATE_HZ: u32 = 16_000;

/// 20 ms at 16 kHz.
pub const FRAME_SAMPLES: usize = 320;

const FRAME_MS: u32 = 20;
const MIN_UTTERANCE_MS: u32 = 400;
const PREROLL_MS: u32 = 300;
const ONSET_CONFIRM_MS: u32 = 60;

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

/// Emits utterances bounded by confirmed silence from frames labelled
/// speech or not. Onset and offset each need consecutive frames per
/// [`VadTuning`]; a pre-roll ring is prepended so the first syllable is not clipped.
pub struct UtteranceVad {
    tuning: VadTuning,
    phase: Phase,
    preroll: VecDeque<[i16; FRAME_SAMPLES]>,
    utterance: Vec<i16>,
    utterance_frames: u32,
    voiced_frames: u32,
}

impl fmt::Debug for UtteranceVad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UtteranceVad")
            .field("tuning", &self.tuning)
            .field("phase", &self.phase)
            .field("utterance_frames", &self.utterance_frames)
            .field("voiced_frames", &self.voiced_frames)
            .finish_non_exhaustive()
    }
}

impl UtteranceVad {
    /// A segmenter in the silent state.
    pub fn new(tuning: VadTuning) -> Self {
        Self {
            tuning,
            phase: Phase::Silent,
            preroll: VecDeque::with_capacity(tuning.preroll_frames as usize),
            utterance: Vec::new(),
            utterance_frames: 0,
            voiced_frames: 0,
        }
    }

    /// Feed one 20 ms frame and whether it holds speech; `Some` when an
    /// utterance boundary is crossed.
    pub fn feed(&mut self, frame: &[i16; FRAME_SAMPLES], is_voiced: bool) -> Option<VadEvent> {
        let event = self.advance(frame, is_voiced);
        if is_voiced && self.in_utterance() {
            self.voiced_frames = self.voiced_frames.saturating_add(1);
        }
        event
    }

    fn advance(&mut self, frame: &[i16; FRAME_SAMPLES], is_voiced: bool) -> Option<VadEvent> {
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

    /// Whether an utterance is being recorded, its trailing silence included.
    pub fn in_utterance(&self) -> bool {
        matches!(self.phase, Phase::Voiced | Phase::Trailing { .. })
    }

    /// Voiced frames in the utterance being recorded; meaningful only while
    /// [`in_utterance`](Self::in_utterance).
    pub fn voiced_frames(&self) -> u32 {
        self.voiced_frames
    }

    /// The last `max_samples` samples of the utterance being recorded.
    pub fn utterance_tail(&self, max_samples: usize) -> &[i16] {
        let start = self.utterance.len().saturating_sub(max_samples);
        &self.utterance[start..]
    }

    /// Discard any in-progress utterance and pre-roll and return to silence.
    pub fn reset(&mut self) {
        self.phase = Phase::Silent;
        self.preroll.clear();
        self.utterance.clear();
        self.utterance_frames = 0;
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
        self.voiced_frames = 0;
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
        }
    }

    /// Feed `count` identical frames, collecting any events.
    fn feed(vad: &mut UtteranceVad, voiced: bool, count: usize) -> Vec<VadEvent> {
        let frame = [if voiced { VOICED } else { SILENT }; FRAME_SAMPLES];
        (0..count)
            .filter_map(|_| vad.feed(&frame, voiced))
            .collect()
    }

    fn frames(sample: i16, count: usize) -> Vec<i16> {
        vec![sample; count * FRAME_SAMPLES]
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
    fn reset_discards_in_progress_utterance_and_preroll() {
        let mut vad = UtteranceVad::new(tight_tuning());
        feed(&mut vad, false, 5);
        feed(&mut vad, true, 10);
        vad.reset();

        assert_eq!(
            feed(&mut vad, false, 5),
            NO_EVENTS,
            "no utterance survives reset"
        );
        feed(&mut vad, true, 4);
        let events = feed(&mut vad, false, 5);
        let expected = [frames(SILENT, 2), frames(VOICED, 4), frames(SILENT, 2)].concat();
        assert_eq!(
            events,
            [VadEvent::UtteranceComplete(expected)],
            "pre-roll after reset holds only post-reset silence"
        );
    }
}
