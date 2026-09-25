//! `mic::consumer::drain_to_pcm` rate conversion to 16 kHz, fed from a
//! ring buffer directly so no audio hardware is needed.

#![cfg(feature = "mic")]

use std::f32::consts::PI;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use assistd_voice::mic::consumer::drain_to_pcm;
use ringbuf::HeapRb;
use ringbuf::traits::{Producer, Split};

/// Push one second of a 440 Hz mono sine at `rate_hz` into a ring and
/// drain it with the stop flag already set.
fn drain_one_second(rate_hz: u32) -> Vec<i16> {
    let sample_count = rate_hz as usize;
    let (mut producer, consumer) = HeapRb::<f32>::new(sample_count).split();
    for index in 0..sample_count {
        let secs = index as f32 / rate_hz as f32;
        producer
            .try_push((2.0 * PI * 440.0 * secs).sin() * 0.5)
            .expect("ring sized for the whole clip");
    }
    drop(producer);
    let stop = Arc::new(AtomicBool::new(true));
    drain_to_pcm(consumer, rate_hz, 16_000 * 4, stop).expect("drain_to_pcm error")
}

fn assert_mostly_nonzero(pcm: &[i16], label: &str) {
    let nonzero = pcm.iter().filter(|&&sample| sample != 0).count();
    assert!(
        nonzero > pcm.len() / 2,
        "{label}: {nonzero} of {} samples non-zero",
        pcm.len()
    );
}

#[test]
fn drain_resamples_common_device_rates_to_16k() {
    for rate in [48_000, 44_100] {
        let pcm = drain_one_second(rate);
        assert!(
            pcm.len().abs_diff(16_000) <= 600,
            "{rate} Hz: resampled to {} samples (zero-padded last chunk allows slack)",
            pcm.len()
        );
        assert_mostly_nonzero(&pcm, &format!("{rate} Hz"));
    }
}

#[test]
fn drain_passes_16k_through_unchanged_in_length() {
    let pcm = drain_one_second(16_000);
    assert_eq!(pcm.len(), 16_000);
    assert_mostly_nonzero(&pcm, "16000 Hz");
}
