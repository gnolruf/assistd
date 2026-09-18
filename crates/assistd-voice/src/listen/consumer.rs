//! Continuous-listen ring consumer: emits 20 ms 16 kHz i16 frames over
//! a channel.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use ringbuf::HeapCons;
use ringbuf::traits::Observer;
use tokio::sync::mpsc;
use tracing::debug;

use super::vad::FRAME_SAMPLES;
use crate::mic::capture::AudioCaptureError;
use crate::mic::resample::{CHUNK_SIZE, ChunkResampler, f32_to_i16};

const IDLE_PARK: Duration = Duration::from_millis(10);

/// Emit 20 ms frames on `frame_tx` until `stop_flag` is set. A dropped
/// receiver is a silent stop.
pub fn drain_to_frames(
    mut consumer: HeapCons<f32>,
    native_rate: u32,
    stop_flag: Arc<AtomicBool>,
    frame_tx: mpsc::Sender<Box<[i16; FRAME_SAMPLES]>>,
) -> Result<(), AudioCaptureError> {
    let mut resampler = ChunkResampler::new(native_rate)?;
    let mut pending: Vec<i16> = Vec::with_capacity(FRAME_SAMPLES * 2);
    let mut frames_emitted: u64 = 0;

    loop {
        if consumer.occupied_len() >= CHUNK_SIZE {
            if let Some(samples) = resampler.pull(&mut consumer)? {
                pending.extend(samples.iter().map(|&s| f32_to_i16(s)));
            }
            while pending.len() >= FRAME_SAMPLES {
                let mut frame = Box::new([0i16; FRAME_SAMPLES]);
                frame.copy_from_slice(&pending[..FRAME_SAMPLES]);
                pending.drain(..FRAME_SAMPLES);
                if frame_tx.blocking_send(frame).is_err() {
                    debug!(
                        target: "assistd::voice::listen",
                        frames_emitted,
                        "frame receiver dropped; stopping capture"
                    );
                    return Ok(());
                }
                frames_emitted += 1;
            }
            continue;
        }

        if stop_flag.load(Ordering::Relaxed) {
            debug!(
                target: "assistd::voice::listen",
                frames_emitted,
                "stop_flag observed; ending stream"
            );
            return Ok(());
        }

        std::thread::park_timeout(IDLE_PARK);
    }
}
