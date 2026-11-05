//! `UserConversation`: whether the user is talking to the model, which keeps task
//! runs off a lone lane from a turn's start until the conversation has been quiet a while.

use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use tokio::sync::watch;
use tokio::time::Instant;

/// Where the user's conversation stands.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(in crate::state) enum ConversationState {
    #[default]
    Idle,
    /// `turns` user turns are waiting for the lane or running.
    Talking {
        turns: NonZeroU32,
        /// Ended on request while talking, so it ends with its last turn.
        ends_with_turn: bool,
    },
    /// No turn is in flight; the conversation ends at `until`, wall-clock `resumes_at`.
    Quiet {
        until: Instant,
        resumes_at: DateTime<Utc>,
    },
}

/// The user's conversation, which holds the task queue while it is not
/// [`ConversationState::Idle`].
#[derive(Debug, Default)]
pub(in crate::state) struct UserConversation {
    state: watch::Sender<ConversationState>,
}

impl UserConversation {
    pub(in crate::state) fn subscribe(&self) -> watch::Receiver<ConversationState> {
        self.state.subscribe()
    }

    pub(in crate::state) fn state(&self) -> ConversationState {
        *self.state.borrow()
    }

    pub(in crate::state) fn is_open(&self) -> bool {
        self.state() != ConversationState::Idle
    }

    /// Open or continue the conversation with a turn; once its last turn ends the
    /// conversation stays open for `quiet`.
    pub(in crate::state) fn begin_turn(self: &Arc<Self>, quiet: Duration) -> ConversationTurn {
        self.state.send_modify(|state| {
            *state = match *state {
                ConversationState::Talking {
                    turns,
                    ends_with_turn,
                } => ConversationState::Talking {
                    turns: turns.saturating_add(1),
                    ends_with_turn,
                },
                ConversationState::Idle | ConversationState::Quiet { .. } => {
                    ConversationState::Talking {
                        turns: NonZeroU32::MIN,
                        ends_with_turn: false,
                    }
                }
            };
        });
        ConversationTurn {
            conversation: Arc::clone(self),
            quiet,
        }
    }

    /// End the conversation now, or with its last turn if one is in flight.
    pub(in crate::state) fn end(&self) {
        self.state.send_if_modified(|state| match state {
            ConversationState::Idle => false,
            ConversationState::Quiet { .. } => {
                *state = ConversationState::Idle;
                true
            }
            ConversationState::Talking { ends_with_turn, .. } => {
                *ends_with_turn = true;
                true
            }
        });
    }

    /// End a quiet conversation whose quiet time ran out by `now`.
    pub(in crate::state) fn end_if_quiet_by(&self, now: Instant) {
        self.state.send_if_modified(|state| match *state {
            ConversationState::Quiet { until, .. } if until <= now => {
                *state = ConversationState::Idle;
                true
            }
            _ => false,
        });
    }

    fn end_turn(&self, quiet: Duration) {
        self.state.send_modify(|state| {
            let ConversationState::Talking {
                turns,
                ends_with_turn,
            } = *state
            else {
                return;
            };
            *state = match NonZeroU32::new(turns.get() - 1) {
                Some(turns) => ConversationState::Talking {
                    turns,
                    ends_with_turn,
                },
                None if ends_with_turn || quiet.is_zero() => ConversationState::Idle,
                None => ConversationState::Quiet {
                    until: Instant::now() + quiet,
                    resumes_at: Utc::now() + quiet,
                },
            };
        });
    }
}

/// One user turn of the conversation, ended when dropped.
#[derive(Debug)]
pub(in crate::state) struct ConversationTurn {
    conversation: Arc<UserConversation>,
    quiet: Duration,
}

impl ConversationTurn {
    /// End a turn that never ran, leaving no quiet time behind it.
    pub(in crate::state) fn abandon(mut self) {
        self.quiet = Duration::ZERO;
    }
}

impl Drop for ConversationTurn {
    fn drop(&mut self) {
        self.conversation.end_turn(self.quiet);
    }
}
