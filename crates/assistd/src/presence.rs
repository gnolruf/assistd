//! Client for the manual presence-control subcommands: `sleep`, `wake`,
//! `drowse`, `cycle`. Each sends exactly one IPC request over the daemon's
//! Unix socket, prints the resulting presence state, and exits on
//! `Event::Done` or `Event::Error`.

use anyhow::Result;
use assistd_ipc::{Event, IpcClient, PresenceState, Request};
use uuid::Uuid;

/// Which transition the CLI is asking the daemon to run.
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
    let mut stream = IpcClient::new()
        .one_shot(req)
        .await
        .map_err(crate::ipc_helper::map_not_reachable)?;

    loop {
        let event = match stream.next_event().await? {
            Some(ev) => ev,
            None => anyhow::bail!("daemon closed the connection without sending a terminal event"),
        };
        match event {
            Event::Presence { state, .. } => {
                println!("presence: {}", presence_label(state));
            }
            Event::Done { .. } => return Ok(()),
            Event::Error { message, .. } => {
                eprintln!("daemon error: {message}");
                std::process::exit(1);
            }
            // Any other event types (Delta, ToolCall, …) shouldn't appear on a
            // presence request — ignore so a future protocol addition doesn't
            // break this CLI.
            _ => {}
        }
    }
}

fn presence_label(state: PresenceState) -> &'static str {
    match state {
        PresenceState::Active => "active",
        PresenceState::Drowsy => "drowsy",
        PresenceState::Sleeping => "sleeping",
    }
}
