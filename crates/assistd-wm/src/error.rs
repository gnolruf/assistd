//! Typed error for the [`WindowManager`](crate::WindowManager) trait.

use std::time::Duration;

/// Errors produced by [`WindowManager`](crate::WindowManager) methods.
#[derive(thiserror::Error, Debug)]
pub enum WmError {
    /// No backend is configured, the initial connect failed, or the
    /// reconnection supervisor is mid-backoff.
    #[error("compositor IPC disconnected")]
    Disconnected,

    /// The compositor accepted the IPC frame but rejected the command.
    #[error("compositor rejected command: {0}")]
    Rejected(String),

    /// The IPC call exceeded its timeout. Retry once, then treat the
    /// backend as `Disconnected`.
    #[error("IPC timed out after {0:?}")]
    Timeout(Duration),

    /// The string names the unsupported operation.
    #[error("backend does not support {0}")]
    Unsupported(&'static str),

    /// The IPC exchange named by `op` failed in transport.
    #[error("{op}: {source}")]
    Ipc {
        op: &'static str,
        #[source]
        source: TransportError,
    },
}

/// Underlying failure of a compositor IPC exchange.
#[derive(thiserror::Error, Debug)]
pub enum TransportError {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[cfg(feature = "sway")]
    #[error(transparent)]
    Sway(#[from] swayipc_async::Error),
}

#[cfg(any(feature = "i3", feature = "sway"))]
impl WmError {
    pub(crate) fn ipc(op: &'static str, source: impl Into<TransportError>) -> Self {
        Self::Ipc {
            op,
            source: source.into(),
        }
    }
}

pub type WmResult<T> = std::result::Result<T, WmError>;
