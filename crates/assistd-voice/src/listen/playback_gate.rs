//! Frame gate that mutes the VAD while the daemon's own speech is audible,
//! plus a short hangover for speaker tail and room reverb.

use tokio::sync::watch;

/// Frames still discarded after speaking ends: 400 ms of 20 ms frames.
pub const PLAYBACK_HANGOVER_FRAMES: u32 = 20;

/// Decides per 20 ms frame whether the mic is hearing the daemon's own
/// TTS output. Reads the speaking signal synchronously, so it is safe
/// on a blocking thread.
pub struct PlaybackGate {
    speaking: watch::Receiver<bool>,
    hangover_frames: u32,
    hangover_remaining: u32,
}

impl PlaybackGate {
    /// Gate on `speaking`, holding closed for `hangover_frames` after it drops.
    pub fn new(speaking: watch::Receiver<bool>, hangover_frames: u32) -> Self {
        Self {
            speaking,
            hangover_frames,
            hangover_remaining: 0,
        }
    }

    /// Whether the frame arriving now must be discarded.
    pub fn blocks_frame(&mut self) -> bool {
        if *self.speaking.borrow() {
            self.hangover_remaining = self.hangover_frames;
            return true;
        }
        if self.hangover_remaining > 0 {
            self.hangover_remaining -= 1;
            return true;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn verdicts(gate: &mut PlaybackGate, count: usize) -> Vec<bool> {
        (0..count).map(|_| gate.blocks_frame()).collect()
    }

    #[test]
    fn passes_frames_while_silent() {
        let (_tx, rx) = watch::channel(false);
        let mut gate = PlaybackGate::new(rx, 3);
        assert_eq!(verdicts(&mut gate, 4), [false; 4]);
    }

    #[test]
    fn blocks_while_speaking_and_through_the_hangover() {
        let (tx, rx) = watch::channel(false);
        let mut gate = PlaybackGate::new(rx, 3);

        tx.send_replace(true);
        assert_eq!(verdicts(&mut gate, 2), [true, true]);

        tx.send_replace(false);
        assert_eq!(
            verdicts(&mut gate, 5),
            [true, true, true, false, false],
            "hangover covers exactly three frames after speaking ends"
        );
    }

    #[test]
    fn speaking_again_mid_hangover_rearms_it() {
        let (tx, rx) = watch::channel(false);
        let mut gate = PlaybackGate::new(rx, 2);

        tx.send_replace(true);
        gate.blocks_frame();
        tx.send_replace(false);
        assert!(gate.blocks_frame());

        tx.send_replace(true);
        gate.blocks_frame();
        tx.send_replace(false);
        assert_eq!(verdicts(&mut gate, 3), [true, true, false]);
    }

    #[test]
    fn zero_hangover_reopens_immediately() {
        let (tx, rx) = watch::channel(true);
        let mut gate = PlaybackGate::new(rx, 0);
        assert!(gate.blocks_frame());
        tx.send_replace(false);
        assert!(!gate.blocks_frame());
    }
}
