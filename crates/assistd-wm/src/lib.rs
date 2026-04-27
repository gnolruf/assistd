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
//! that speak their respective IPC protocols. Milestone 1 only defines
//! the trait so the daemon can hold `Arc<dyn WindowManager>` in
//! `AppState` when the feature lands.

use anyhow::Result;
use async_trait::async_trait;

/// Identifier for a window or client in the compositor's namespace.
/// Represented as a string because i3, Sway, and Hyprland each use
/// different underlying id types.
pub type WindowId = String;

/// Workspace identifier. Same reasoning as [`WindowId`].
pub type WorkspaceId = String;

#[async_trait]
pub trait WindowManager: Send + Sync + 'static {
    /// Focus the window with the given id.
    async fn focus(&self, window: &WindowId) -> Result<()>;

    /// Move the focused window to the given workspace.
    async fn move_to_workspace(&self, workspace: &WorkspaceId) -> Result<()>;

    /// Return the id of the currently focused window, if any.
    async fn focused_window(&self) -> Result<Option<WindowId>>;
}

/// Placeholder [`WindowManager`] that refuses every operation. Used by
/// the scaffolding until a real compositor backend lands.
pub struct NoWindowManager;

#[async_trait]
impl WindowManager for NoWindowManager {
    async fn focus(&self, _window: &WindowId) -> Result<()> {
        anyhow::bail!("no window manager backend is configured")
    }
    async fn move_to_workspace(&self, _workspace: &WorkspaceId) -> Result<()> {
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

    #[test]
    fn version_is_not_empty() {
        assert!(!version().is_empty());
    }
}
