//! Exponential backoff for the WM-backend reconnection supervisor.
//!
//! The shape is `2^attempt` seconds, capped at 60s, the same schedule as
//! `assistd_embed::server::backoff` so both subsystems hit the same
//! patience budget when their underlying daemon (llama-server / i3 /
//! sway) churns. Duplicated here intentionally: pulling 12 lines into
//! a separate workspace crate would cost more in plumbing than it
//! saves in DRY.

use std::time::Duration;

/// Exponential backoff schedule: `2^attempt` seconds, capped at 60s.
/// `attempt = 0` is the first retry, i.e. 1s after the first failure.
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
