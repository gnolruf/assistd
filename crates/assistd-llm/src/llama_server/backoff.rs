use std::time::Duration;

/// Maximum number of consecutive startup failures before the supervisor gives
/// up and enters [`crate::llama_server::ReadyState::Degraded`].
pub const MAX_CONSECUTIVE_FAILURES: u32 = 5;

/// Rolling-window cap on *any* restart (startup or crash-after-ready). Without
/// this, a child that stays up for at least [`super::supervisor::MIN_HEALTHY_SECONDS`]
/// before each crash would reset the consecutive-failure counter and restart
/// forever — paying the multi-second weight-load cost every cycle. Hitting
/// this cap forces [`crate::llama_server::ReadyState::Degraded`] regardless
/// of how long any individual child lived.
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
}
