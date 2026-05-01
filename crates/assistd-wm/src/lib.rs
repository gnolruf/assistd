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

/// One row of [`WindowManager::list_windows`]. The trait's wider methods
/// (`focus`, `move_to_workspace`) take a [`WindowId`] alone; this struct
/// is the richer view returned to surfaces (the `wm list` command) that
/// need title and workspace context to format human-readable output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Window {
    /// Identifier usable with [`WindowManager::focus`] /
    /// [`WindowManager::move_to_workspace`]. For i3 this is the X11 class.
    pub id: WindowId,
    /// `_NET_WM_NAME` / `WM_NAME`. Missing on some transient or just-
    /// mapped windows.
    pub title: Option<String>,
    /// Workspace this window currently lives on. `None` for scratchpad
    /// or otherwise un-anchored windows.
    pub workspace: Option<String>,
}

/// One row of [`WindowManager::list_workspaces`]. Named to avoid colliding
/// with `tokio_i3ipc::reply::Workspace` at use sites; both backends will
/// flatten their compositor's workspace shape into this common view.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceInfo {
    /// Numeric component (i3's `num`). `-1` when the workspace name does
    /// not begin with a number.
    pub num: i32,
    /// User-visible label (e.g. `"1"`, `"1:web"`, `"scratch"`).
    pub name: String,
    /// True when this workspace currently holds the active focus on its
    /// output. (Multi-monitor setups have one focused workspace per
    /// output; this matches i3's `focused` field on `reply::Workspace`.)
    pub focused: bool,
    /// Monitor / output name (e.g. `"DP-1"`).
    pub output: String,
}

#[async_trait]
pub trait WindowManager: Send + Sync + 'static {
    /// Focus the window with the given id.
    async fn focus(&self, window: &WindowId) -> Result<()>;

    /// Move the named window to the given workspace.
    async fn move_to_workspace(&self, window: &WindowId, workspace: &WorkspaceId) -> Result<()>;

    /// Return the id of the currently focused window, or `None` when no
    /// window holds focus (e.g. an empty workspace).
    async fn focused_window(&self) -> Result<Option<WindowId>>;

    /// Enumerate every mapped window the compositor knows about.
    async fn list_windows(&self) -> Result<Vec<Window>>;

    /// Enumerate every workspace the compositor knows about.
    async fn list_workspaces(&self) -> Result<Vec<WorkspaceInfo>>;

    /// Send a raw, backend-specific command payload. Used for operations
    /// that have no typed return value and where adding a per-operation
    /// trait method would be a thin wrapper around a string format
    /// (resize, layout, …). Callers are responsible for the syntax —
    /// keep this behind subcommands that document the expected dialect.
    async fn run_raw(&self, payload: &str) -> Result<()>;

    /// Report whether the backend is actually connected to a compositor.
    /// The default `true` covers concrete backends; [`NoWindowManager`]
    /// overrides to `false` so callers can short-circuit with a single
    /// "compositor not connected" message instead of waiting for the
    /// generic `bail!` from each operation.
    fn is_connected(&self) -> bool {
        true
    }
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
    async fn list_windows(&self) -> Result<Vec<Window>> {
        anyhow::bail!("no window manager backend is configured")
    }
    async fn list_workspaces(&self) -> Result<Vec<WorkspaceInfo>> {
        anyhow::bail!("no window manager backend is configured")
    }
    async fn run_raw(&self, _payload: &str) -> Result<()> {
        anyhow::bail!("no window manager backend is configured")
    }
    fn is_connected(&self) -> bool {
        false
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

    #[tokio::test]
    async fn no_window_manager_refuses_list_windows() {
        assert!(NoWindowManager.list_windows().await.is_err());
    }

    #[tokio::test]
    async fn no_window_manager_refuses_list_workspaces() {
        assert!(NoWindowManager.list_workspaces().await.is_err());
    }

    #[tokio::test]
    async fn no_window_manager_refuses_run_raw() {
        assert!(NoWindowManager.run_raw("focus").await.is_err());
    }

    #[test]
    fn no_window_manager_reports_disconnected() {
        assert!(!NoWindowManager.is_connected());
    }

    #[test]
    fn version_is_not_empty() {
        assert!(!version().is_empty());
    }
}
