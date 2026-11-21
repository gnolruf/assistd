//! `SessionLapse`: when a session talked to by voice is left behind, so that the next spoken
//! prompt begins a new one.

use std::time::Duration;

use parking_lot::Mutex;
use tokio::time::Instant;

/// Times the quiet since the last user turn ended, speech included. A daemon that has run no
/// turn yet counts as lapsed.
#[derive(Debug)]
pub(super) struct SessionLapse {
    /// `None` for a session that never lapses.
    after: Option<Duration>,
    last_turn_ended: Mutex<Option<Instant>>,
}

impl SessionLapse {
    /// A session that lapses `after` its last turn ends, or never when `after` is zero.
    pub(super) fn new(after: Duration) -> Self {
        Self {
            after: (!after.is_zero()).then_some(after),
            last_turn_ended: Mutex::new(None),
        }
    }

    /// A user turn in flight, noted as ended when dropped.
    pub(super) fn turn(&self) -> TurnInFlight<'_> {
        TurnInFlight(self)
    }

    /// Whether the last user turn ended long enough ago that the session is left behind.
    pub(super) fn lapsed(&self) -> bool {
        self.after.is_some_and(|after| {
            self.last_turn_ended
                .lock()
                .is_none_or(|ended| ended.elapsed() >= after)
        })
    }
}

/// A user turn whose end restarts its [`SessionLapse`].
#[derive(Debug)]
pub(super) struct TurnInFlight<'lapse>(&'lapse SessionLapse);

impl Drop for TurnInFlight<'_> {
    fn drop(&mut self) {
        *self.0.last_turn_ended.lock() = Some(Instant::now());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(start_paused = true)]
    async fn session_lapses_only_once_its_time_has_passed_since_the_last_turn_ended() {
        assert!(!SessionLapse::new(Duration::ZERO).lapsed());

        let lapse = SessionLapse::new(Duration::from_secs(16));
        assert!(lapse.lapsed());

        let turn = lapse.turn();
        tokio::time::advance(Duration::from_secs(60)).await;
        drop(turn);
        tokio::time::advance(Duration::from_secs(15)).await;
        assert!(!lapse.lapsed());
        tokio::time::advance(Duration::from_secs(1)).await;
        assert!(lapse.lapsed());
    }
}
