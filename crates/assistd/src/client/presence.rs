//! `sleep`, `wake`, and `drowse` subcommands.

use std::io::{self, Write};

use anyhow::Result;
use assistd_ipc::{Event, PresenceTarget, Request};
use uuid::Uuid;

use super::run_one_shot;

pub(crate) async fn run(target: PresenceTarget) -> Result<()> {
    let req = Request::SetPresence {
        id: Uuid::new_v4().to_string(),
        target,
    };
    run_one_shot(req, |event| {
        if let Event::Presence { state, .. } = event {
            writeln!(io::stdout(), "presence: {}", state.as_str())?;
        }
        Ok(())
    })
    .await
}
