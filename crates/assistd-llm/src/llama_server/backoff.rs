use std::time::Duration;

/// Maximum number of consecutive startup failures before the supervisor gives
/// up and enters [`crate::llama_server::ReadyState::Degraded`].
pub const MAX_CONSECUTIVE_FAILURES: u32 = 5;

/// Rolling-window cap on *any* restart (startup or crash-after-ready). Without
/// it, a child that stays healthy for `MIN_HEALTHY_SECONDS` before each crash
/// would reset the consecutive-failure counter and restart forever, paying the
/// multi-second weight-load cost every cycle. Hitting this cap forces
/// [`crate::llama_server::ReadyState::Degraded`] regardless of how long any
/// individual child lived.
pub const MAX_RESTARTS_PER_WINDOW: usize = 10;

/// Width of the rolling window used by [`MAX_RESTARTS_PER_WINDOW`].
pub const RESTART_WINDOW: Duration = Duration::from_secs(600);

/// Exponential backoff schedule: `2^attempt` seconds, capped at 60s.
///
/// `attempt` starts at 0, yielding the sequence
/// `1s, 2s, 4s, 8s, 16s, 32s, 60s, 60s, …`.
pub fn backoff_delay(attempt: u32) -> Duration {
    const CAP_SECS: u64 = 60;
    let secs = 1u64.checked_shl(attempt).unwrap_or(CAP_SECS).min(CAP_SECS);
    Duration::from_secs(secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doubles_from_one_second_and_caps_at_sixty() {
        let cases = [
            (0, 1),
            (1, 2),
            (2, 4),
            (3, 8),
            (4, 16),
            (5, 32),
            (6, 60),
            (9, 60),
            (63, 60),
            (64, 60),
            (u32::MAX, 60),
        ];
        for (attempt, secs) in cases {
            assert_eq!(
                backoff_delay(attempt),
                Duration::from_secs(secs),
                "attempt {attempt}"
            );
        }
    }
}
