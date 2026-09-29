//! `listen-*` subcommands.

use std::io::{self, Write};

use anyhow::Result;
use assistd_ipc::{Event, Request};
use uuid::Uuid;

use super::run_one_shot;

#[derive(Debug, Clone, Copy)]
pub(crate) enum ListenAction {
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

pub(crate) async fn run(action: ListenAction) -> Result<()> {
    let req = action.to_request(Uuid::new_v4().to_string());
    run_one_shot(req, |event| {
        if let Event::ListenState { active, .. } = event {
            writeln!(
                io::stdout(),
                "listen: {}",
                if *active { "on" } else { "off" }
            )?;
        }
        Ok(())
    })
    .await
}
