//! 48 kHz stereo → 16 kHz mono resampling coverage.
//!
//! Exercises `mic::consumer::drain_loop` end-to-end by feeding a
//! synthesised mono waveform into the same SPSC ring the cpal
//! callback pushes into. Bypasses cpal entirely so the test is
//! deterministic and runs on CI hosts without audio hardware.
//!
//! Stereo → mono downmix is a compile-time feature of the cpal
//! callback (see `mic::capture::CallbackState::push_f32`). By the time
//! a sample hits `drain_loop`, it is already mono f32 at the device's
//! native sample rate; the only remaining work is the rate conversion
//! we verify here.

use std::f32::consts::PI;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use assistd_voice::mic::consumer::drain_loop;
use ringbuf::HeapRb;
#[allow(unused_imports)]
use ringbuf::traits::Observer;
use ringbuf::traits::{Producer, Split};

/// Synthesise `seconds` of a single-channel 440 Hz sine at `rate_hz`,
/// push it into a ring buffer, close the producer, and drain through
/// `drain_loop` with the stop flag already set. Returns the produced
/// 16 kHz i16 PCM.
fn run_drain(rate_hz: u32, seconds: f32) -> Vec<i16> {
    let n_in = (rate_hz as f32 * seconds).round() as usize;
    let rb = HeapRb::<f32>::new(n_in + 16);
    let (mut prod, cons) = rb.split();
    for i in 0..n_in {
        let t = i as f32 / rate_hz as f32;
        let s = (2.0 * PI * 440.0 * t).sin() * 0.5;
        if prod.try_push(s).is_err() {
            break;
        }
    }
    drop(prod); // closes the producer side; consumer.occupied_len stays stable
    let stop = Arc::new(AtomicBool::new(true));
    // max_pcm_samples is generous — enough for a few seconds at 16 kHz.
    drain_loop(cons, rate_hz, 16_000 * 4, stop).expect("drain_loop error")
}

#[test]
fn drain_loop_resamples_48k_to_16k() {
    let pcm = run_drain(48_000, 1.0);
    // 1 s at 48 kHz → ~16 000 samples at 16 kHz. Rubato's
    // FastFixedIn operates in fixed 1024-sample input chunks with a
    // final padded chunk, so we accept a few hundred samples of
    // slack at the tail.
    let expected = 16_000usize;
    assert!(
        (pcm.len() as i64 - expected as i64).unsigned_abs() as usize <= 600,
        "resampled length {} should be within ±600 of {}",
        pcm.len(),
        expected,
    );
    // Energy: a non-zero sine must produce non-zero samples.
    let nonzero = pcm.iter().filter(|s| **s != 0).count();
    assert!(
        nonzero > pcm.len() / 2,
        "expected a majority of non-zero samples, got {nonzero} of {}",
        pcm.len()
    );
}

#[test]
fn drain_loop_noop_when_rate_already_16k() {
    let pcm = run_drain(16_000, 1.0);
    // No resampling: output length equals input length modulo the
    // final-chunk handling. No-op path preserves counts exactly.
    let expected = 16_000usize;
    assert!(
        (pcm.len() as i64 - expected as i64).unsigned_abs() as usize <= 16,
        "no-op path length {} should equal {}",
        pcm.len(),
        expected,
    );
    let nonzero = pcm.iter().filter(|s| **s != 0).count();
    assert!(
        nonzero > pcm.len() / 2,
        "expected a majority of non-zero samples, got {nonzero} of {}",
        pcm.len()
    );
}

#[test]
fn drain_loop_resamples_44100_to_16k() {
    // 44.1 kHz is the other common USB-class rate. Uses a
    // non-integer ratio — a closer stress test of the resampler.
    let pcm = run_drain(44_100, 1.0);
    let expected = 16_000usize;
    assert!(
        (pcm.len() as i64 - expected as i64).unsigned_abs() as usize <= 600,
        "resampled length {} should be within ±600 of {}",
        pcm.len(),
        expected,
    );
}
