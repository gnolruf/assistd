//! Supervises a child process that serves HTTP: spawn it in its own process
//! group, poll its health endpoint, restart with backoff, and park once restart limits trip.

use std::net::SocketAddr;
use std::time::Duration;

use tokio::process::Command;

mod api_key;
mod error;
mod health;
mod llama_env;
mod ownership;
mod process;
mod service;
mod supervisor;

pub use api_key::ApiKey;
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

    /// Address the child serves HTTP on, including its health endpoint.
    fn listen_addr(&self) -> SocketAddr;

    /// Path whose 200 response marks the child ready.
    fn health_path(&self) -> &'static str {
        "/health"
    }

    /// Longest a freshly spawned child may take to pass its health check.
    fn ready_timeout(&self) -> Duration;
}
