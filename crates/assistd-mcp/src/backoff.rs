use std::time::Duration;

/// Cap on consecutive transport-startup failures before a server's
/// supervisor parks itself in `HealthState::Unhealthy` permanently.
/// Matches the embed-server budget at
/// `assistd_embed::server::backoff::MAX_CONSECUTIVE_FAILURES`.
pub const MAX_CONSECUTIVE_FAILURES: u32 = 5;

/// A transport that ran successfully for at least this many seconds
/// before exiting resets the consecutive-failure counter — i.e. a
/// long-lived server that occasionally crashes does not get parked.
pub const MIN_HEALTHY_SECONDS: u64 = 30;

/// Cap on the SSE reconnection delay (also reused for stdio restarts).
pub const RECONNECT_MAX_SECS: u64 = 60;

/// Exponential backoff: 1s, 2s, 4s, 8s, 16s, 32s, 60s (capped).
pub fn backoff_delay(attempt: u32) -> Duration {
    let secs = 1u64
        .checked_shl(attempt)
        .unwrap_or(RECONNECT_MAX_SECS)
        .min(RECONNECT_MAX_SECS);
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
