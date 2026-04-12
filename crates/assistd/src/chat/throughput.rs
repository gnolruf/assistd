//! Streaming-generation throughput meter.
//!
//! Counts `LlmEvent::Delta` chunks since the first delta arrived and
//! exposes an instantaneous rate snapshot for the status bar. The llama.cpp
//! OpenAI SSE stream emits roughly one chunk per token, so chunks/sec is a
//! serviceable approximation of tokens/sec.

use std::time::{Duration, Instant};

/// How long the final rate stays visible in the status bar after a generation
/// completes.
pub const FINAL_RATE_HOLD: Duration = Duration::from_secs(3);

#[derive(Debug, Default, Clone, Copy)]
pub struct ThroughputMeter {
    first_delta_at: Option<Instant>,
    chunk_count: u64,
    finished_at: Option<Instant>,
    final_rate: Option<f64>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ThroughputSnapshot {
    pub rate: Option<f64>,
}

impl ThroughputMeter {
    pub const fn new() -> Self {
        Self {
            first_delta_at: None,
            chunk_count: 0,
            finished_at: None,
            final_rate: None,
        }
    }

    pub fn on_delta(&mut self, now: Instant) {
        if self.first_delta_at.is_none() {
            self.first_delta_at = Some(now);
        }
        self.chunk_count += 1;
    }

    pub fn on_done(&mut self, now: Instant) {
        self.finished_at = Some(now);
        self.final_rate = self.instant_rate(now);
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    pub fn snapshot(&self, now: Instant) -> ThroughputSnapshot {
        ThroughputSnapshot {
            rate: self.rate_at(now),
        }
    }

    fn rate_at(&self, now: Instant) -> Option<f64> {
        if let Some(finished_at) = self.finished_at {
            if now.duration_since(finished_at) < FINAL_RATE_HOLD {
                self.final_rate
            } else {
                None
            }
        } else {
            self.instant_rate(now)
        }
    }

    fn instant_rate(&self, now: Instant) -> Option<f64> {
        let start = self.first_delta_at?;
        let elapsed = now.duration_since(start).as_secs_f64();
        if elapsed <= 0.0 {
            return None;
        }
        Some(self.chunk_count as f64 / elapsed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> Instant {
        Instant::now()
    }

    #[test]
    fn new_meter_reports_nothing() {
        let m = ThroughputMeter::new();
        assert!(m.snapshot(base()).rate.is_none());
        assert!(m.first_delta_at.is_none());
    }

    #[test]
    fn rate_is_chunks_over_elapsed_seconds() {
        let t0 = base();
        let mut m = ThroughputMeter::new();
        m.on_delta(t0);
        m.on_delta(t0 + Duration::from_millis(500));
        m.on_delta(t0 + Duration::from_millis(1000));
        let rate = m
            .snapshot(t0 + Duration::from_secs(2))
            .rate
            .expect("rate should be set");
        assert!((rate - 1.5).abs() < 1e-9, "rate={rate}");
    }

    #[test]
    fn rate_is_none_at_zero_elapsed() {
        let t0 = base();
        let mut m = ThroughputMeter::new();
        m.on_delta(t0);
        assert!(m.snapshot(t0).rate.is_none());
    }

    #[test]
    fn done_freezes_final_rate_and_expires_after_hold() {
        let t0 = base();
        let mut m = ThroughputMeter::new();
        m.on_delta(t0);
        m.on_delta(t0 + Duration::from_secs(1));
        m.on_done(t0 + Duration::from_secs(2));

        assert_eq!(m.snapshot(t0 + Duration::from_secs(3)).rate, Some(1.0));
        assert!(m.snapshot(t0 + Duration::from_secs(10)).rate.is_none());
    }

    #[test]
    fn reset_clears_state() {
        let t0 = base();
        let mut m = ThroughputMeter::new();
        m.on_delta(t0);
        m.on_delta(t0 + Duration::from_millis(200));
        m.on_done(t0 + Duration::from_millis(500));
        m.reset();
        assert!(m.snapshot(t0 + Duration::from_secs(1)).rate.is_none());
        assert!(m.first_delta_at.is_none());
        assert!(m.finished_at.is_none());
        assert_eq!(m.chunk_count, 0);
    }
}
