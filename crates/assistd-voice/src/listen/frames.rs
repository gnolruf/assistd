//! Continuous-listen capture sink: cuts captured audio into 20 ms 16 kHz i16
//! frames for the VAD task.

use std::ops::ControlFlow;

use tokio::sync::mpsc;
use tracing::debug;

use super::vad::FRAME_SAMPLES;
use crate::mic::consumer::ChunkSink;
use crate::mic::resample::f32_to_i16;

/// Sends each complete 20 ms frame on a channel, ending the capture once
/// the receiver is dropped.
#[derive(Debug)]
pub struct FrameSink {
    frame_tx: mpsc::Sender<Box<[i16; FRAME_SAMPLES]>>,
    pending: Vec<i16>,
}

impl FrameSink {
    pub fn new(frame_tx: mpsc::Sender<Box<[i16; FRAME_SAMPLES]>>) -> Self {
        Self {
            frame_tx,
            pending: Vec::with_capacity(FRAME_SAMPLES * 2),
        }
    }
}

impl ChunkSink for FrameSink {
    type Captured = ();

    fn accept(&mut self, chunk: &[f32]) -> ControlFlow<()> {
        self.pending
            .extend(chunk.iter().map(|&sample| f32_to_i16(sample)));
        while self.pending.len() >= FRAME_SAMPLES {
            let mut frame = Box::new([0i16; FRAME_SAMPLES]);
            frame.copy_from_slice(&self.pending[..FRAME_SAMPLES]);
            self.pending.drain(..FRAME_SAMPLES);
            if self.frame_tx.blocking_send(frame).is_err() {
                debug!(
                    target: "assistd::voice::listen",
                    "frame receiver dropped; stopping capture"
                );
                return ControlFlow::Break(());
            }
        }
        ControlFlow::Continue(())
    }

    fn finish(self) {}
}
