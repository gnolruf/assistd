//! `UserInteraction`: whether the user is talking to the model, which keeps task
//! runs off a lone lane from a turn's start until they have been quiet a while.

use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use tokio::sync::watch;
use tokio::time::Instant;

/// Where the user's interaction stands.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum InteractionState {
    #[default]
    Idle,
    /// `turns` user turns are waiting for the lane or running.
    Talking {
        turns: NonZeroU32,
        /// Ended on request while talking, so it ends with its last turn.
        ends_with_turn: bool,
    },
    /// No turn is in flight; the interaction ends at `until`, wall-clock `resumes_at`.
    Quiet {
        until: Instant,
        resumes_at: DateTime<Utc>,
    },
}

/// The user's interaction, which holds the task queue while it is not
/// [`InteractionState::Idle`].
#[derive(Debug, Default)]
pub(crate) struct UserInteraction {
    state: watch::Sender<InteractionState>,
}

impl UserInteraction {
    pub(super) fn subscribe(&self) -> watch::Receiver<InteractionState> {
        self.state.subscribe()
    }

    /// Open or continue the interaction with a turn; once its last turn ends the
    /// interaction stays open for `quiet`.
    pub(crate) fn begin_turn(self: &Arc<Self>, quiet: Duration) -> InteractionTurn {
        self.state.send_modify(|state| {
            *state = match *state {
                InteractionState::Talking {
                    turns,
                    ends_with_turn,
                } => InteractionState::Talking {
                    turns: turns.saturating_add(1),
                    ends_with_turn,
                },
                InteractionState::Idle | InteractionState::Quiet { .. } => {
                    InteractionState::Talking {
                        turns: NonZeroU32::MIN,
                        ends_with_turn: false,
                    }
                }
            };
        });
        InteractionTurn {
            interaction: Arc::clone(self),
            quiet,
        }
    }

    /// End the interaction now, or with its last turn if one is in flight.
    pub(crate) fn end(&self) {
        self.state.send_if_modified(|state| match state {
            InteractionState::Idle => false,
            InteractionState::Quiet { .. } => {
                *state = InteractionState::Idle;
                true
            }
            InteractionState::Talking { ends_with_turn, .. } => {
                *ends_with_turn = true;
                true
            }
        });
    }

    /// End a quiet interaction whose quiet time ran out by `now`.
    pub(crate) fn end_if_quiet_by(&self, now: Instant) {
        self.state.send_if_modified(|state| match *state {
            InteractionState::Quiet { until, .. } if until <= now => {
                *state = InteractionState::Idle;
                true
            }
            _ => false,
        });
    }

    fn end_turn(&self, quiet: Duration) {
        self.state.send_modify(|state| {
            let InteractionState::Talking {
                turns,
                ends_with_turn,
            } = *state
            else {
                return;
            };
            *state = match NonZeroU32::new(turns.get() - 1) {
                Some(turns) => InteractionState::Talking {
                    turns,
                    ends_with_turn,
                },
                None if ends_with_turn || quiet.is_zero() => InteractionState::Idle,
                None => InteractionState::Quiet {
                    until: Instant::now() + quiet,
                    resumes_at: Utc::now() + quiet,
                },
            };
        });
    }
}

/// One user turn of the interaction, ended when dropped.
#[derive(Debug)]
pub(crate) struct InteractionTurn {
    interaction: Arc<UserInteraction>,
    quiet: Duration,
}

impl Drop for InteractionTurn {
    fn drop(&mut self) {
        self.interaction.end_turn(self.quiet);
    }
}
