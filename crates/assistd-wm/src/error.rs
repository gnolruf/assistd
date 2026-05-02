//! Typed error for the [`WindowManager`](crate::WindowManager) trait.
//!
//! The trait used to return `anyhow::Result<T>` for every method, which
//! collapses every failure mode into one opaque string at the call site.
//! [`crate::commands::wm::WmCommand`]-class consumers can't pick a useful
//! recovery hint from that ("compositor not connected" wants a different
//! message than "window 17 not found"). [`WmError`] keeps the variants
//! distinguishable while still letting backends propagate underlying
//! `anyhow::Error`s through `?` via the [`From<anyhow::Error>`] impl
//! provided by `#[error(transparent)]`.

use std::fmt::Display;
use std::time::Duration;

use crate::WindowId;

/// Errors produced by [`WindowManager`](crate::WindowManager) methods.
///
/// The `Ipc` variant is the catch-all for unexpected backend errors;
/// prefer the typed variants when the caller can act on them differently
/// (a `Disconnected` is recoverable by waiting for reconnect, a
/// `NotFound` means the LLM should re-run `wm list`, …).
#[derive(thiserror::Error, Debug)]
pub enum WmError {
    /// The backend is not currently connected to a compositor. Either no
    /// backend was configured (`NoWindowManager`), the initial connect
    /// failed, or a previously-connected backend's reconnection
    /// supervisor (PR 5) is mid-backoff.
    #[error("compositor IPC disconnected")]
    Disconnected,

    /// The requested window doesn't exist (or no longer exists). Holds
    /// the id we attempted to act on so the caller can include it in
    /// the recovery hint.
    #[error("window {0:?} not found")]
    NotFound(WindowId),

    /// The compositor accepted the IPC frame but rejected the command
    /// itself (e.g. i3's per-result `success: false` reply).
    #[error("compositor rejected command: {0}")]
    Rejected(String),

    /// An IPC call did not complete within the configured timeout.
    /// Consumers should retry once, then assume `Disconnected`.
    #[error("IPC timed out after {0:?}")]
    Timeout(Duration),

    /// The active backend doesn't implement this operation (e.g. i3 has
    /// no rich `GET_OUTPUTS` reply, so [`crate::WindowManager::list_outputs`]
    /// surfaces this variant). The string names the operation so the
    /// caller can render a precise message.
    #[error("backend does not support {0}")]
    Unsupported(&'static str),

    /// Catch-all for unexpected backend errors. Wraps an [`anyhow::Error`]
    /// so backends can keep using `?` and `.context(...)` over their
    /// underlying `io::Result` / library errors.
    #[error(transparent)]
    Ipc(#[from] anyhow::Error),
}

/// Convenience `Result` alias used by every [`WindowManager`](crate::WindowManager)
/// method.
pub type WmResult<T> = std::result::Result<T, WmError>;

/// Wrap a foreign error (typically `io::Error` from `tokio-i3ipc` or a
/// `swayipc::Error` Display impl) with a human-readable context string,
/// producing a [`WmError::Ipc`]. Saves backends from repeating
/// `anyhow::anyhow!("...: {e}")` everywhere.
pub fn ipc_ctx<E: Display>(err: E, ctx: &'static str) -> WmError {
    WmError::Ipc(anyhow::anyhow!("{ctx}: {err}"))
}
