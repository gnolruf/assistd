//! `voice-*` subcommands.

use std::io::{self, Write};

use anyhow::Result;
use assistd_ipc::{Event, Request};
use uuid::Uuid;

use super::run_one_shot;

#[derive(Debug, Clone, Copy)]
pub(crate) enum VoiceCtlAction {
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

pub(crate) async fn run(action: VoiceCtlAction) -> Result<()> {
    let req = action.to_request(Uuid::new_v4().to_string());
    run_one_shot(req, |event| {
        match event {
            Event::VoiceOutputState { enabled, .. } => writeln!(
                io::stdout(),
                "voice-output: {}",
                if *enabled { "on" } else { "off" }
            )?,
            Event::Readiness {
                component, state, ..
            } => writeln!(io::stdout(), "{component}: {state}")?,
            _ => {}
        }
        Ok(())
    })
    .await
}
