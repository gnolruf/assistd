//! `listen-*` subcommands.

use anyhow::Result;
use assistd_ipc::{Event, IpcClient, Request};
use uuid::Uuid;

#[derive(Debug, Clone, Copy)]
pub enum ListenAction {
    Start,
    Stop,
    Toggle,
    State,
}

impl ListenAction {
    fn to_request(self, id: String) -> Request {
        match self {
            ListenAction::Start => Request::ListenStart { id },
            ListenAction::Stop => Request::ListenStop { id },
            ListenAction::Toggle => Request::ListenToggle { id },
            ListenAction::State => Request::GetListenState { id },
        }
    }
}

pub async fn run(action: ListenAction) -> Result<()> {
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
            Event::ListenState { active, .. } => {
                println!("listen: {}", if active { "on" } else { "off" });
            }
            Event::Done { .. } => return Ok(()),
            Event::Error { message, .. } => {
                eprintln!("daemon error: {message}");
                std::process::exit(1);
            }
            _ => {}
        }
    }
}
