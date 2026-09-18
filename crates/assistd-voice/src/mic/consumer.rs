//! Push-to-talk ring consumer: accumulates 16 kHz i16 PCM until stopped.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use ringbuf::HeapCons;
use ringbuf::traits::Observer;
use tracing::debug;

use super::capture::AudioCaptureError;
use super::resample::{CHUNK_SIZE, ChunkResampler, f32_to_i16};

const IDLE_PARK: Duration = Duration::from_millis(10);

/// Drain the ring until `stop_flag` is set or `max_pcm_samples` is
/// reached, returning 16 kHz i16 PCM.
pub fn drain_to_pcm(
    mut consumer: HeapCons<f32>,
    native_rate: u32,
    max_pcm_samples: usize,
    stop_flag: Arc<AtomicBool>,
) -> Result<Vec<i16>, AudioCaptureError> {
    let mut resampler = ChunkResampler::new(native_rate)?;
    let mut pcm: Vec<i16> = Vec::with_capacity(max_pcm_samples.min(16_000 * 10));

    loop {
        if consumer.occupied_len() >= CHUNK_SIZE {
            if append_capped(resampler.pull(&mut consumer)?, &mut pcm, max_pcm_samples) {
                break;
            }
            continue;
        }
        if stop_flag.load(Ordering::Relaxed) {
            append_capped(resampler.pull(&mut consumer)?, &mut pcm, max_pcm_samples);
            break;
        }
        std::thread::park_timeout(IDLE_PARK);
    }

    debug!(
        target: "assistd::voice::mic",
        pcm_samples = pcm.len(),
        "consumer drained"
    );
    Ok(pcm)
}

/// Append `samples` to `pcm` up to `max`. True when the cap is reached.
fn append_capped(samples: Option<&[f32]>, pcm: &mut Vec<i16>, max: usize) -> bool {
    for &s in samples.unwrap_or_default() {
        if pcm.len() >= max {
            return true;
        }
        pcm.push(f32_to_i16(s));
    }
    pcm.len() >= max
}
