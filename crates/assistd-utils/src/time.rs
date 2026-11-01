//! Duration conversions for wire fields and logs.

use std::time::Duration;

/// `duration` in whole milliseconds, saturating at `u64::MAX`.
pub fn saturating_millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}
