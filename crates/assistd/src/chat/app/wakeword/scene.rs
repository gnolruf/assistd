//! The listening computer's animation: audio flows into it from the left
//! while someone speaks, every column it swallows leaves on the right as a
//! bit, and its face wonders while audio flows in, then reacts to the verdict.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Columns in each lane, before and after the computer.
pub(in crate::chat) const LANE_COLUMNS: usize = 14;

const WORRY_HOLD: Duration = Duration::from_secs(3);

const WAVE_GLYPHS: [char; 7] = ['▁', '▂', '▃', '▅', '▆', '▇', '█'];
const QUIET: char = ' ';

/// The computer's face.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::chat) enum Mood {
    Content,
    /// Audio is entering the computer.
    Wondering,
    /// The last speech did not pass.
    Worried,
}

/// Both lanes, leftmost column first, and the computer's mood. A worry is
/// held as `worry_pending` until the audio has finished flowing in.
pub(in crate::chat) struct ComputerScene {
    hearing: bool,
    worried_until: Option<Instant>,
    worry_pending: bool,
    inbound: VecDeque<char>,
    outbound: VecDeque<char>,
    swallowed: usize,
    audio_arrived: bool,
    frame: u32,
}

impl ComputerScene {
    pub(in crate::chat) fn new() -> Self {
        Self {
            hearing: false,
            worried_until: None,
            worry_pending: false,
            inbound: VecDeque::from([QUIET; LANE_COLUMNS]),
            outbound: VecDeque::from([QUIET; LANE_COLUMNS]),
            swallowed: 0,
            audio_arrived: false,
            frame: 0,
        }
    }

    pub(in crate::chat) fn set_hearing(&mut self, hearing: bool) {
        self.hearing = hearing;
    }

    /// Smile at speech that passed; look worried for [`WORRY_HOLD`] at
    /// speech that did not, once its audio has flowed in. A worry already
    /// showing runs its course either way.
    pub(in crate::chat) fn react(&mut self, passed: bool) {
        self.worry_pending = !passed;
        self.start_pending_worry();
    }

    /// A showing worry outranks wondering.
    pub(in crate::chat) fn mood(&self, now: Instant) -> Mood {
        if self.worried_until.is_some_and(|until| now < until) {
            Mood::Worried
        } else if self.audio_arrived {
            Mood::Wondering
        } else {
            Mood::Content
        }
    }

    /// Whether anything is still moving.
    pub(in crate::chat) fn is_live(&self) -> bool {
        self.hearing
            || self.swallowed > 0
            || self
                .inbound
                .iter()
                .chain(&self.outbound)
                .any(|&glyph| glyph != QUIET)
    }

    /// Move both lanes one column to the right.
    pub(in crate::chat) fn advance(&mut self) {
        self.frame = self.frame.wrapping_add(1);
        let noise = scramble(self.frame);
        let entering = if self.hearing {
            WAVE_GLYPHS[noise as usize % WAVE_GLYPHS.len()]
        } else {
            QUIET
        };
        self.inbound.push_front(entering);
        if self.inbound.pop_back().is_some_and(|glyph| glyph != QUIET) {
            self.swallowed += 1;
            self.audio_arrived = true;
        }
        let leaving = if self.swallowed > 0 {
            self.swallowed -= 1;
            if noise & 0x100 == 0 { '0' } else { '1' }
        } else {
            QUIET
        };
        self.outbound.push_front(leaving);
        self.outbound.pop_back();
        if !self.audio_flowing_in() {
            self.audio_arrived = false;
        }
        self.start_pending_worry();
    }

    pub(in crate::chat) fn inbound(&self) -> String {
        self.inbound.iter().collect()
    }

    pub(in crate::chat) fn outbound(&self) -> String {
        self.outbound.iter().collect()
    }

    fn audio_flowing_in(&self) -> bool {
        self.hearing || self.inbound.iter().any(|&glyph| glyph != QUIET)
    }

    fn start_pending_worry(&mut self) {
        if self.worry_pending && !self.audio_flowing_in() {
            self.worry_pending = false;
            self.worried_until = Some(Instant::now() + WORRY_HOLD);
        }
    }
}

/// Cheap, well-mixed bits from a frame counter.
fn scramble(frame: u32) -> u32 {
    let mixed = frame.wrapping_mul(0x9E37_79B9);
    mixed ^ (mixed >> 15)
}
