//! Exponential backoff and rolling-window restart accounting for supervisors.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Consecutive failed restarts before a supervisor stops retrying at the
/// normal cadence.
pub const MAX_CONSECUTIVE_FAILURES: u32 = 5;

/// A session at least this long resets the consecutive-failure counter.
pub const MIN_HEALTHY_SECONDS: u64 = 30;

/// Rolling-window cap on restarts; catches a child that dies just after
/// each [`MIN_HEALTHY_SECONDS`] reset.
pub const MAX_RESTARTS_PER_WINDOW: usize = 10;

/// Width of the rolling window used by [`MAX_RESTARTS_PER_WINDOW`].
pub const RESTART_WINDOW: Duration = Duration::from_secs(600);

/// Upper bound on [`backoff_delay`], in seconds.
pub const BACKOFF_CAP_SECS: u64 = 60;

/// Exponential backoff: `2^attempt` seconds from `attempt` 0, capped at
/// [`BACKOFF_CAP_SECS`].
pub fn backoff_delay(attempt: u32) -> Duration {
    let secs = 1u64
        .checked_shl(attempt)
        .unwrap_or(BACKOFF_CAP_SECS)
        .min(BACKOFF_CAP_SECS);
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

/// Restart accounting for one supervised child: consecutive failed
/// attempts plus a rolling window of every restart made.
#[derive(Debug, Default)]
pub struct RestartPolicy {
    consecutive_failures: u32,
    restarts: VecDeque<Instant>,
}

impl RestartPolicy {
    /// Account for a session that ended after `ran_for`: one lasting
    /// [`MIN_HEALTHY_SECONDS`] clears the consecutive counter, a shorter
    /// one counts as a failure.
    pub fn record_session_end(&mut self, ran_for: Duration) {
        if ran_for >= Duration::from_secs(MIN_HEALTHY_SECONDS) {
            self.consecutive_failures = 0;
        } else {
            self.consecutive_failures += 1;
        }
    }

    /// Account for a child that never came up at all.
    pub fn record_spawn_failure(&mut self) {
        self.consecutive_failures += 1;
    }

    /// Register the restart about to be made and decide its delay.
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
    fn backoff_doubles_from_one_second_and_caps_at_sixty() {
        let cases = [(0, 1), (5, 32), (6, 60), (u32::MAX, 60)];
        for (attempt, secs) in cases {
            assert_eq!(
                backoff_delay(attempt),
                Duration::from_secs(secs),
                "attempt {attempt}"
            );
        }
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
    fn rolling_window_parks_a_child_that_crashes_after_each_healthy_session() {
        let t0 = Instant::now();
        let mut policy = RestartPolicy::default();
        let healthy = Duration::from_secs(MIN_HEALTHY_SECONDS + 5);

        let window = u32::try_from(MAX_RESTARTS_PER_WINDOW).expect("restart cap fits u32");
        for i in 1..window {
            policy.record_session_end(healthy);
            let at = t0 + healthy * i;
            assert!(matches!(
                policy.next_restart(at),
                RestartDecision::Backoff { .. }
            ));
        }

        policy.record_session_end(healthy);
        let at = t0 + healthy * window;
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

        let window = u32::try_from(MAX_RESTARTS_PER_WINDOW).expect("restart cap fits u32");
        for i in 1..window {
            policy.record_session_end(healthy);
            policy.next_restart(t0 + healthy * i);
        }

        policy.record_session_end(healthy);
        assert!(matches!(
            policy.next_restart(t0 + RESTART_WINDOW * 2),
            RestartDecision::Backoff { .. }
        ));
    }
}
