//! The listening computer's animation: audio flows into it from the left
//! while someone speaks, and every column it swallows leaves on the right as a bit.

use std::collections::VecDeque;

/// Columns in each lane, before and after the computer.
pub(in crate::chat) const LANE_COLUMNS: usize = 14;

const WAVE_GLYPHS: [char; 7] = ['▁', '▂', '▃', '▅', '▆', '▇', '█'];
const QUIET: char = ' ';

/// Both lanes, leftmost column first, and the audio still inside the computer.
pub(in crate::chat) struct ComputerScene {
    hearing: bool,
    inbound: VecDeque<char>,
    outbound: VecDeque<char>,
    swallowed: usize,
    frame: u32,
}

impl ComputerScene {
    pub(in crate::chat) fn new() -> Self {
        Self {
            hearing: false,
            inbound: VecDeque::from([QUIET; LANE_COLUMNS]),
            outbound: VecDeque::from([QUIET; LANE_COLUMNS]),
            swallowed: 0,
            frame: 0,
        }
    }

    pub(in crate::chat) fn set_hearing(&mut self, hearing: bool) {
        self.hearing = hearing;
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
        }
        let leaving = if self.swallowed > 0 {
            self.swallowed -= 1;
            if noise & 0x100 == 0 { '0' } else { '1' }
        } else {
            QUIET
        };
        self.outbound.push_front(leaving);
        self.outbound.pop_back();
    }

    pub(in crate::chat) fn inbound(&self) -> String {
        self.inbound.iter().collect()
    }

    pub(in crate::chat) fn outbound(&self) -> String {
        self.outbound.iter().collect()
    }
}

/// Cheap, well-mixed bits from a frame counter.
fn scramble(frame: u32) -> u32 {
    let mixed = frame.wrapping_mul(0x9E37_79B9);
    mixed ^ (mixed >> 15)
}
