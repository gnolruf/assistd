//! `sleep`, `wake`, `drowse`, and `cycle` subcommands.

use anyhow::Result;
use assistd_ipc::{Event, PresenceState, Request};
use uuid::Uuid;

use crate::ipc_helper::run_one_shot;

#[derive(Debug, Clone, Copy)]
pub enum PresenceAction {
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
                target: PresenceState::Sleeping,
            },
            PresenceAction::Drowse => Request::SetPresence {
                id,
                target: PresenceState::Drowsy,
            },
            PresenceAction::Wake => Request::SetPresence {
                id,
                target: PresenceState::Active,
            },
            PresenceAction::Cycle => Request::Cycle { id },
        }
    }
}

pub async fn run(action: PresenceAction) -> Result<()> {
    let req = action.to_request(Uuid::new_v4().to_string());
    run_one_shot(req, |event| {
        if let Event::Presence { state, .. } = event {
            println!("presence: {}", presence_label(*state));
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
    }
}
