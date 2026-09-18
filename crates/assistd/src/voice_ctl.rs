//! `voice-*` subcommands.

use anyhow::Result;
use assistd_ipc::{Event, Request};
use uuid::Uuid;

use crate::ipc_helper::run_one_shot;

#[derive(Debug, Clone, Copy)]
pub enum VoiceCtlAction {
    Toggle,
    Skip,
    State,
}

impl VoiceCtlAction {
    fn to_request(self, id: String) -> Request {
        match self {
            VoiceCtlAction::Toggle => Request::VoiceToggle { id },
            VoiceCtlAction::Skip => Request::VoiceSkip { id },
            VoiceCtlAction::State => Request::GetVoiceState { id },
        }
    }
}

pub async fn run(action: VoiceCtlAction) -> Result<()> {
    let req = action.to_request(Uuid::new_v4().to_string());
    run_one_shot(req, |event| {
        if let Event::VoiceOutputState { enabled, .. } = event {
            println!("voice-output: {}", if *enabled { "on" } else { "off" });
        }
        Ok(())
    })
    .await
}
