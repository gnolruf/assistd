use std::time::Duration;

/// `2^attempt` seconds, capped at 60s. `attempt = 0` is the first retry.
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
        for (attempt, secs) in [
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
        ] {
            assert_eq!(
                backoff_delay(attempt),
                Duration::from_secs(secs),
                "attempt {attempt}"
            );
        }
    }
}
