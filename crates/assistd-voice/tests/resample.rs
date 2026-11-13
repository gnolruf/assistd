//! `mic::consumer::drain_to_pcm` rate conversion to 16 kHz, fed from a
//! ring buffer directly so no audio hardware is needed.

#![cfg(feature = "mic")]

use std::f32::consts::PI;
use std::sync::atomic::AtomicBool;

use assistd_voice::mic::capture::StreamFailure;
use assistd_voice::mic::consumer::drain_to_pcm;
use ringbuf::HeapRb;
use ringbuf::traits::{Producer, Split};

const TONE_AMPLITUDE: f32 = 0.5;

/// Push one second of a `tone_hz` mono sine at `rate_hz` into a ring and
/// drain it with the stop flag already set.
fn drain_one_second(rate_hz: u32, tone_hz: f32) -> Vec<i16> {
    let sample_count = rate_hz as usize;
    let (mut producer, consumer) = HeapRb::<f32>::new(sample_count).split();
    for index in 0..sample_count {
        let secs = index as f32 / rate_hz as f32;
        producer
            .try_push((2.0 * PI * tone_hz * secs).sin() * TONE_AMPLITUDE)
            .expect("ring sized for the whole clip");
    }
    drop(producer);
    let stop = AtomicBool::new(true);
    drain_to_pcm(
        consumer,
        rate_hz,
        16_000 * 4,
        &stop,
        &StreamFailure::default(),
    )
    .expect("drain_to_pcm error")
}

fn rms(pcm: &[i16]) -> f32 {
    let energy: f32 = pcm.iter().map(|&sample| f32::from(sample).powi(2)).sum();
    (energy / pcm.len() as f32).sqrt()
}

#[test]
fn drain_resamples_common_device_rates_to_16k() {
    let tone_rms = TONE_AMPLITUDE * f32::from(i16::MAX) / 2f32.sqrt();
    for rate in [48_000, 44_100] {
        let pcm = drain_one_second(rate, 440.0);
        assert_eq!(
            pcm.len(),
            16_000,
            "{rate} Hz: one second in, one second out"
        );
        let level = rms(&pcm) / tone_rms;
        assert!(
            (0.95..1.05).contains(&level),
            "{rate} Hz: 440 Hz tone came out at {level} of its level"
        );
    }
}

#[test]
fn drain_filters_out_what_16k_cannot_carry() {
    let tone_rms = TONE_AMPLITUDE * f32::from(i16::MAX) / 2f32.sqrt();
    for rate in [48_000, 44_100] {
        let level = rms(&drain_one_second(rate, 12_000.0)) / tone_rms;
        assert!(
            level < 0.01,
            "{rate} Hz: 12 kHz tone aliased through at {level} of its level"
        );
    }
}
