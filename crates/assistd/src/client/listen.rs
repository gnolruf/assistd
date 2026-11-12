//! `listen-*` subcommands.

use std::io::{self, Write};

use anyhow::Result;
use assistd_ipc::{EventBody, Request};
use uuid::Uuid;

use super::run_one_shot;

/// Sends the request `with_id` builds and prints the daemon's answer.
pub(crate) async fn run(with_id: fn(String) -> Request) -> Result<()> {
    let req = with_id(Uuid::new_v4().to_string());
    run_one_shot(req, |event| {
        if let EventBody::ListenState { phase } = &event.body {
            writeln!(io::stdout(), "listen: {}", phase.as_str())?;
        }
        Ok(())
    })
    .await
}
