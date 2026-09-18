//! Typed error for the [`WindowManager`](crate::WindowManager) trait.

use std::fmt::Display;
use std::time::Duration;

use crate::WindowId;

/// Errors produced by [`WindowManager`](crate::WindowManager) methods.
#[derive(thiserror::Error, Debug)]
pub enum WmError {
    /// No backend is configured, the initial connect failed, or the
    /// reconnection supervisor is mid-backoff.
    #[error("compositor IPC disconnected")]
    Disconnected,

    #[error("window {0:?} not found")]
    NotFound(WindowId),

    /// The compositor accepted the IPC frame but rejected the command.
    #[error("compositor rejected command: {0}")]
    Rejected(String),

    /// Retry once, then assume `Disconnected`.
    #[error("IPC timed out after {0:?}")]
    Timeout(Duration),

    /// The string names the unsupported operation.
    #[error("backend does not support {0}")]
    Unsupported(&'static str),

    /// Catch-all for unexpected backend errors.
    #[error(transparent)]
    Ipc(#[from] anyhow::Error),
}

pub type WmResult<T> = std::result::Result<T, WmError>;

/// Wrap a transport error with context as a [`WmError::Ipc`].
pub fn ipc_ctx<E: Display>(err: E, ctx: &'static str) -> WmError {
    WmError::Ipc(anyhow::anyhow!("{ctx}: {err}"))
}
