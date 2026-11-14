//! Ring consumer: resamples captured audio to 16 kHz and hands each chunk
//! to a [`ChunkSink`].

use std::ops::ControlFlow;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use ringbuf::HeapCons;
use ringbuf::traits::Observer;

use super::capture::{AudioCaptureError, StreamFailure};
use super::resample::{CHUNK_SIZE, ChunkResampler, f32_to_i16};

const IDLE_PARK: Duration = Duration::from_millis(10);

/// What a capture does with the 16 kHz mono audio it drains.
pub trait ChunkSink: Send + 'static {
    /// What the capture returns once it has ended.
    type Captured: Send + 'static;

    /// Take the next chunk of samples; `Break` ends the capture.
    fn accept(&mut self, chunk: &[f32]) -> ControlFlow<()>;

    /// Hand over the capture's result.
    fn finish(self) -> Self::Captured;
}

/// Push-to-talk sink: keeps the audio as i16 PCM, ending the capture once
/// it holds `max_samples`.
#[derive(Debug)]
pub struct PcmRecording {
    pcm: Vec<i16>,
    max_samples: usize,
}

impl PcmRecording {
    pub fn capped_at(max_samples: usize) -> Self {
        Self {
            pcm: Vec::with_capacity(max_samples.min(16_000 * 10)),
            max_samples,
        }
    }
}

impl ChunkSink for PcmRecording {
    type Captured = Vec<i16>;

    fn accept(&mut self, chunk: &[f32]) -> ControlFlow<()> {
        let room = self.max_samples - self.pcm.len();
        self.pcm
            .extend(chunk.iter().take(room).map(|&sample| f32_to_i16(sample)));
        if self.pcm.len() < self.max_samples {
            ControlFlow::Continue(())
        } else {
            ControlFlow::Break(())
        }
    }

    fn finish(self) -> Vec<i16> {
        self.pcm
    }
}

/// Feed `sink` from the ring until it breaks or `stop_flag` is set, then
/// return what it captured; a stop also feeds it the audio still buffered.
/// Errors once `failure` says the stream feeding the ring has failed, even
/// if `stop_flag` is set too.
pub fn drain<S: ChunkSink>(
    mut consumer: HeapCons<f32>,
    native_rate: u32,
    stop_flag: &AtomicBool,
    failure: &StreamFailure,
    mut sink: S,
) -> Result<S::Captured, AudioCaptureError> {
    let mut resampler = ChunkResampler::new(native_rate)?;
    loop {
        if consumer.occupied_len() >= CHUNK_SIZE {
            let chunk = resampler.pull(&mut consumer)?.unwrap_or_default();
            if sink.accept(chunk).is_break() {
                break;
            }
            continue;
        }
        failure.check()?;
        if stop_flag.load(Ordering::Relaxed) {
            drain_tail(&mut resampler, &mut consumer, &mut sink)?;
            break;
        }
        std::thread::park_timeout(IDLE_PARK);
    }
    Ok(sink.finish())
}

/// Feed `sink` the last partial chunk and what the resampler still holds.
fn drain_tail<S: ChunkSink>(
    resampler: &mut ChunkResampler,
    consumer: &mut HeapCons<f32>,
    sink: &mut S,
) -> Result<(), AudioCaptureError> {
    let partial = resampler.pull(consumer)?.unwrap_or_default();
    if sink.accept(partial).is_break() {
        return Ok(());
    }
    while let Some(tail) = resampler.flush()? {
        if sink.accept(tail).is_break() {
            break;
        }
    }
    Ok(())
}
