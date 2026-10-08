//! Supervises a child process that serves HTTP: spawn it in its own process
//! group, poll `/health`, restart with backoff, and park once restart limits trip.

use std::net::SocketAddr;
use std::time::Duration;

use tokio::process::Command;

mod error;
mod health;
mod llama_env;
mod ownership;
mod process;
mod service;
mod supervisor;

pub use error::ChildServerError;
pub use llama_env::remove_llama_env;
pub use service::{ChildServer, ChildServerStatus, ReadyState};

/// Describes one supervised server: how to spawn it and where it listens.
pub trait ChildServerSpec: Send + Sync + 'static {
    /// Short name used in every log line and error about this server.
    fn name(&self) -> &'static str;

    /// The command to spawn. Stdio, the process group and the parent-death
    /// signal are set by the supervisor.
    fn command(&self) -> Command;

    /// Address whose `/health` endpoint signals readiness.
    fn listen_addr(&self) -> SocketAddr;

    /// Longest a freshly spawned child may take to pass `/health`.
    fn ready_timeout(&self) -> Duration;
}
