//! Client for the voice-output CLI subcommands: `voice-toggle`,
//! `voice-skip`, `voice-state`. Each sends one IPC request over the
//! daemon's Unix socket, prints the response, and exits on `Event::Done`
//! or `Event::Error`.

use anyhow::Result;
use assistd_ipc::{Event, IpcClient, Request};
use uuid::Uuid;

/// Which voice-output command the CLI is dispatching.
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

/// Send a voice-output command to the daemon and print the response.
///
/// # Errors
///
/// Returns an error if the IPC connection fails or the daemon sends an
/// unexpected terminal event.
pub async fn run(action: VoiceCtlAction) -> Result<()> {
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
            Event::VoiceOutputState { enabled, .. } => {
                println!("voice-output: {}", if enabled { "on" } else { "off" });
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
