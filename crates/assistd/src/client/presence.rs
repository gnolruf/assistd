//! `sleep`, `wake`, and `drowse` subcommands.

use std::io::{self, Write};

use anyhow::Result;
use assistd_ipc::{EventBody, PresenceTarget, Request, RequestBody};
use uuid::Uuid;

use super::run_one_shot;

pub(crate) async fn run(target: PresenceTarget) -> Result<()> {
    let req = Request::new(
        Uuid::new_v4().to_string(),
        RequestBody::SetPresence { target },
    );
    run_one_shot(req, |event| {
        if let EventBody::Presence { state } = &event.body {
            writeln!(io::stdout(), "presence: {}", state.as_str())?;
        }
        Ok(())
    })
    .await
}
