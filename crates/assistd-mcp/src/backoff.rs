use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Cap on consecutive failed restart attempts before a server's
/// supervisor backs off to a slow-cadence retry loop in
/// `HealthState::Unhealthy`. Matches the embed-server budget at
/// `assistd_embed::server::backoff::MAX_CONSECUTIVE_FAILURES`.
pub const MAX_CONSECUTIVE_FAILURES: u32 = 5;

/// A transport that ran successfully for at least this many seconds
/// before exiting resets the consecutive-failure counter; i.e. a
/// long-lived server that occasionally crashes does not get parked.
pub const MIN_HEALTHY_SECONDS: u64 = 30;

/// Rolling-window cap on restarts, counted regardless of how long the
/// preceding session lived. Without it a server that stays up for
/// [`MIN_HEALTHY_SECONDS`] before each crash resets the consecutive
/// counter every cycle and restarts forever. Mirrors
/// `assistd_llm::llama_server::backoff::MAX_RESTARTS_PER_WINDOW`.
pub const MAX_RESTARTS_PER_WINDOW: usize = 10;

/// Width of the rolling window used by [`MAX_RESTARTS_PER_WINDOW`].
pub const RESTART_WINDOW: Duration = Duration::from_secs(600);

/// Cap on the SSE reconnection delay (also reused for stdio restarts).
pub const RECONNECT_MAX_SECS: u64 = 60;

/// Once a supervisor has hit [`MAX_CONSECUTIVE_FAILURES`] or
/// [`MAX_RESTARTS_PER_WINDOW`], it transitions to
/// `HealthState::Unhealthy` and sleeps this long between further
/// spawn attempts. Picked so a misconfigured server the user has just
/// fixed (typo'd binary path, etc.) self-heals within minutes without
/// the daemon thrashing if the underlying problem persists.
pub const UNHEALTHY_RETRY_INTERVAL: Duration = Duration::from_secs(300);

/// Exponential backoff: 1s, 2s, 4s, 8s, 16s, 32s, 60s (capped).
pub fn backoff_delay(attempt: u32) -> Duration {
    let secs = 1u64
        .checked_shl(attempt)
        .unwrap_or(RECONNECT_MAX_SECS)
        .min(RECONNECT_MAX_SECS);
    Duration::from_secs(secs)
}

/// How a supervisor should pace the restart it is about to attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartDecision {
    /// Retry after `delay`; `failures` is how many attempts have
    /// failed since the last healthy session.
    Backoff { delay: Duration, failures: u32 },
    /// [`MAX_CONSECUTIVE_FAILURES`] attempts in a row failed.
    ConsecutiveCapReached { failures: u32 },
    /// The rolling window is saturated even though individual sessions
    /// may have been long enough to reset the consecutive counter.
    WindowCapReached { restarts: usize },
}

/// Restart accounting for a single server's supervisor: consecutive
/// failed attempts plus a rolling window of every restart made.
#[derive(Debug, Default)]
pub struct RestartPolicy {
    consecutive_failures: u32,
    restarts: VecDeque<Instant>,
}

impl RestartPolicy {
    /// Account for a transport session that ended after `ran_for`. A
    /// session that lasted [`MIN_HEALTHY_SECONDS`] clears the
    /// consecutive-failure counter; anything shorter counts as a
    /// failed attempt, which is what makes a server that initializes
    /// and immediately dies back off instead of looping at 1s.
    pub fn record_session_end(&mut self, ran_for: Duration) {
        if ran_for >= Duration::from_secs(MIN_HEALTHY_SECONDS) {
            self.consecutive_failures = 0;
        } else {
            self.consecutive_failures += 1;
        }
    }

    /// Account for a transport that never came up at all.
    pub fn record_spawn_failure(&mut self) {
        self.consecutive_failures += 1;
    }

    /// Register the restart attempt about to be made and decide how
    /// long to wait before it.
    pub fn next_restart(&mut self, now: Instant) -> RestartDecision {
        while let Some(&oldest) = self.restarts.front() {
            if now.duration_since(oldest) > RESTART_WINDOW {
                self.restarts.pop_front();
            } else {
                break;
            }
        }
        self.restarts.push_back(now);

        if self.restarts.len() >= MAX_RESTARTS_PER_WINDOW {
            RestartDecision::WindowCapReached {
                restarts: self.restarts.len(),
            }
        } else if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
            RestartDecision::ConsecutiveCapReached {
                failures: self.consecutive_failures,
            }
        } else {
            RestartDecision::Backoff {
                delay: backoff_delay(self.consecutive_failures.saturating_sub(1)),
                failures: self.consecutive_failures,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_spec_sequence() {
        let expected = [1, 2, 4, 8, 16, 32, 60, 60, 60, 60];
        for (i, want) in expected.iter().enumerate() {
            assert_eq!(
                backoff_delay(i as u32),
                Duration::from_secs(*want),
                "attempt {i}"
            );
        }
    }

    #[test]
    fn caps_at_sixty_seconds_for_large_attempts() {
        assert_eq!(backoff_delay(64), Duration::from_secs(60));
        assert_eq!(backoff_delay(u32::MAX), Duration::from_secs(60));
    }

    #[test]
    fn init_then_immediate_death_backs_off_and_parks() {
        let t0 = Instant::now();
        let mut policy = RestartPolicy::default();
        let mut delays = Vec::new();

        for i in 0..MAX_CONSECUTIVE_FAILURES - 1 {
            policy.record_session_end(Duration::from_millis(200));
            match policy.next_restart(t0 + Duration::from_secs(u64::from(i))) {
                RestartDecision::Backoff { delay, .. } => delays.push(delay),
                other => panic!("attempt {i}: unexpected {other:?}"),
            }
        }
        assert_eq!(
            delays,
            [1, 2, 4, 8]
                .into_iter()
                .map(Duration::from_secs)
                .collect::<Vec<_>>()
        );

        policy.record_session_end(Duration::from_millis(200));
        assert_eq!(
            policy.next_restart(t0 + Duration::from_secs(60)),
            RestartDecision::ConsecutiveCapReached {
                failures: MAX_CONSECUTIVE_FAILURES
            }
        );
    }

    #[test]
    fn healthy_session_resets_the_consecutive_counter() {
        let t0 = Instant::now();
        let mut policy = RestartPolicy::default();

        policy.record_spawn_failure();
        policy.record_spawn_failure();
        assert_eq!(
            policy.next_restart(t0),
            RestartDecision::Backoff {
                delay: Duration::from_secs(2),
                failures: 2
            }
        );

        policy.record_session_end(Duration::from_secs(MIN_HEALTHY_SECONDS));
        assert_eq!(
            policy.next_restart(t0 + Duration::from_secs(60)),
            RestartDecision::Backoff {
                delay: Duration::from_secs(1),
                failures: 0
            }
        );
    }

    #[test]
    fn rolling_window_parks_a_server_that_crashes_after_each_healthy_session() {
        let t0 = Instant::now();
        let mut policy = RestartPolicy::default();
        let healthy = Duration::from_secs(MIN_HEALTHY_SECONDS + 5);

        for i in 0..MAX_RESTARTS_PER_WINDOW - 1 {
            policy.record_session_end(healthy);
            let at = t0 + healthy * (i as u32 + 1);
            assert!(matches!(
                policy.next_restart(at),
                RestartDecision::Backoff { .. }
            ));
        }

        policy.record_session_end(healthy);
        let at = t0 + healthy * MAX_RESTARTS_PER_WINDOW as u32;
        assert_eq!(
            policy.next_restart(at),
            RestartDecision::WindowCapReached {
                restarts: MAX_RESTARTS_PER_WINDOW
            }
        );
    }

    #[test]
    fn restarts_older_than_the_window_are_forgotten() {
        let t0 = Instant::now();
        let mut policy = RestartPolicy::default();
        let healthy = Duration::from_secs(MIN_HEALTHY_SECONDS + 5);

        for i in 0..MAX_RESTARTS_PER_WINDOW - 1 {
            policy.record_session_end(healthy);
            policy.next_restart(t0 + healthy * (i as u32 + 1));
        }

        policy.record_session_end(healthy);
        assert!(matches!(
            policy.next_restart(t0 + RESTART_WINDOW * 2),
            RestartDecision::Backoff { .. }
        ));
    }
}
