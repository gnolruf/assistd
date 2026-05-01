#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::print_stdout,
        clippy::print_stderr
    )
)]

//! Window manager / compositor integration trait.
//!
//! Milestone 8 ships concrete implementations for i3, Sway, and Hyprland
//! that speak their respective IPC protocols. The i3 backend lands first
//! ([`i3::I3Backend`]); Sway and Hyprland follow in later PRs.

use anyhow::Result;
use async_trait::async_trait;

pub mod i3;
pub use i3::{I3Backend, I3Handle};

/// Identifier for a window or client in the compositor's namespace.
/// Represented as a string because i3, Sway, and Hyprland each use
/// different underlying id types — the i3 backend treats this as the
/// X11 window class (e.g. `"Firefox"`).
pub type WindowId = String;

/// Workspace identifier. Same reasoning as [`WindowId`]. The i3 backend
/// emits `workspace number N` when this parses as `u32`, otherwise
/// `workspace "<name>"`.
pub type WorkspaceId = String;

#[async_trait]
pub trait WindowManager: Send + Sync + 'static {
    /// Focus the window with the given id.
    async fn focus(&self, window: &WindowId) -> Result<()>;

    /// Move the named window to the given workspace.
    async fn move_to_workspace(&self, window: &WindowId, workspace: &WorkspaceId) -> Result<()>;

    /// Return the id of the currently focused window, or `None` when no
    /// window holds focus (e.g. an empty workspace).
    async fn focused_window(&self) -> Result<Option<WindowId>>;
}

/// Placeholder [`WindowManager`] that refuses every operation. Used by
/// the daemon when no compositor backend is configured or the configured
/// backend failed to connect at startup.
pub struct NoWindowManager;

#[async_trait]
impl WindowManager for NoWindowManager {
    async fn focus(&self, _window: &WindowId) -> Result<()> {
        anyhow::bail!("no window manager backend is configured")
    }
    async fn move_to_workspace(&self, _window: &WindowId, _workspace: &WorkspaceId) -> Result<()> {
        anyhow::bail!("no window manager backend is configured")
    }
    async fn focused_window(&self) -> Result<Option<WindowId>> {
        Ok(None)
    }
}

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn no_window_manager_reports_no_focused_window() {
        assert!(NoWindowManager.focused_window().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn no_window_manager_refuses_focus() {
        assert!(NoWindowManager.focus(&"win1".into()).await.is_err());
    }

    #[tokio::test]
    async fn no_window_manager_refuses_move() {
        assert!(
            NoWindowManager
                .move_to_workspace(&"win1".into(), &"3".into())
                .await
                .is_err()
        );
    }

    #[test]
    fn version_is_not_empty() {
        assert!(!version().is_empty());
    }
}
