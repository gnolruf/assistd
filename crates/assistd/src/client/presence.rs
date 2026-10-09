//! `sleep`, `wake`, `drowse`, and `cycle` subcommands.

use std::io::{self, Write};

use anyhow::Result;
use assistd_ipc::{Event, PresenceState, PresenceTarget, Request};
use uuid::Uuid;

use super::run_one_shot;

#[derive(Debug, Clone, Copy)]
pub(crate) enum PresenceAction {
    Sleep,
    Drowse,
    Wake,
    Cycle,
}

impl PresenceAction {
    fn to_request(self, id: String) -> Request {
        match self {
            PresenceAction::Sleep => Request::SetPresence {
                id,
                target: PresenceTarget::Sleeping,
            },
            PresenceAction::Drowse => Request::SetPresence {
                id,
                target: PresenceTarget::Drowsy,
            },
            PresenceAction::Wake => Request::SetPresence {
                id,
                target: PresenceTarget::Active,
            },
            PresenceAction::Cycle => Request::Cycle { id },
        }
    }
}

pub(crate) async fn run(action: PresenceAction) -> Result<()> {
    let req = action.to_request(Uuid::new_v4().to_string());
    run_one_shot(req, |event| {
        if let Event::Presence { state, .. } = event {
            writeln!(io::stdout(), "presence: {}", presence_label(*state))?;
        }
        Ok(())
    })
    .await
}

fn presence_label(state: PresenceState) -> &'static str {
    match state {
        PresenceState::Active => "active",
        PresenceState::Drowsy => "drowsy",
        PresenceState::Sleeping => "sleeping",
        PresenceState::Waking => "waking",
    }
}
