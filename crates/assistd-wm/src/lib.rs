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
//! that speak their respective IPC protocols. The i3 ([`i3::I3Backend`])
//! and Sway ([`sway::SwayBackend`]) backends are landed; Hyprland
//! follows in a later PR. The daemon picks one at startup based on
//! `[compositor].type` in `config.toml` (or runtime detection when
//! `type = "auto"` — see `assistd_config::compositor::detect_from_env`).

use async_trait::async_trait;

pub mod criteria;
pub mod error;
pub mod i3;
pub mod sway;
pub use error::{WmError, WmResult};
pub use i3::{I3Backend, I3Handle};
pub use sway::{SwayBackend, SwayHandle};

/// Aggregated shutdown handle for the active WM backend, so the daemon
/// can hold a single `Option<WmHandle>` regardless of which backend
/// was started. Each variant wraps the per-backend handle's own event
/// task; [`WmHandle::shutdown`] dispatches.
pub enum WmHandle {
    I3(I3Handle),
    Sway(SwayHandle),
}

impl WmHandle {
    /// Drains the per-backend event task. The daemon should flip its
    /// shutdown watch first; this just awaits the task's exit.
    pub async fn shutdown(self) {
        match self {
            Self::I3(h) => h.shutdown().await,
            Self::Sway(h) => h.shutdown().await,
        }
    }
}

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

/// One row of [`WindowManager::list_outputs`]. Sway exposes rich output
/// metadata (mode, scale, refresh, focused workspace) over IPC; i3's
/// equivalent reply is much sparser, so the trait method is opt-in:
/// backends without an implementation return `Err` so callers can
/// distinguish "not supported" from "no monitors".
#[derive(Debug, Clone, PartialEq)]
pub struct OutputInfo {
    /// Connector / output name (e.g. `"DP-1"`, `"eDP-1"`, `"HDMI-A-1"`).
    pub name: String,
    /// Whether the output is currently active (powered, has a mode).
    pub active: bool,
    /// X11/i3 sense of "primary"; on Sway this is always `false` because
    /// Wayland has no primary-output concept.
    pub primary: bool,
    /// Current resolution + refresh as `(width, height, refresh_mHz)`.
    /// `None` for disabled outputs or when the compositor doesn't expose
    /// it.
    pub current_mode: Option<(u32, u32, u32)>,
    /// Output scale factor (e.g. `1.0`, `1.5`, `2.0`). `None` when not
    /// reported by the compositor.
    pub scale: Option<f64>,
    /// Name of the workspace currently visible on this output. `None`
    /// for disabled outputs.
    pub focused_workspace: Option<String>,
}

/// Snapshot of what the user is currently looking at. Returned by
/// [`WindowManager::focused_context`]. Each field is independently
/// optional because compositors deliver focus / title / workspace
/// state through separate events; a backend may have any subset.
///
/// Built from the backend's cached event snapshot — no IPC round-trip
/// — so callers can read it cheaply on every LLM query.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FocusedWindowContext {
    /// X11 `WM_CLASS` of the focused window (or compositor-equivalent).
    pub class: Option<String>,
    /// `_NET_WM_NAME` / `WM_NAME` of the focused window.
    pub title: Option<String>,
    /// Name of the workspace currently holding focus.
    pub workspace: Option<String>,
}

/// Best-effort allowlist of X11 `WM_CLASS` values for terminal
/// emulators. Matched case-insensitively because compositors and apps
/// disagree on capitalization (`xterm` vs `XTerm`, `alacritty` vs
/// `Alacritty`). Future work: surface extra classes via config.
const TERMINAL_CLASSES: &[&str] = &[
    "Alacritty",
    "kitty",
    "XTerm",
    "UXTerm",
    "URxvt",
    "rxvt-unicode",
    "st-256color",
    "st",
    "Gnome-terminal",
    "Konsole",
    "Terminator",
    "Tilix",
    "foot",
    "footclient",
    "WezTerm",
    "org.wezfurlong.wezterm",
    "Xfce4-terminal",
    "Termite",
];

/// Returns true when `class` looks like a terminal emulator. The list
/// is intentionally conservative: emitters not in [`TERMINAL_CLASSES`]
/// fall through as non-terminal (no hint emitted, but no wrong hint).
pub fn is_terminal_class(class: &str) -> bool {
    TERMINAL_CLASSES
        .iter()
        .any(|t| t.eq_ignore_ascii_case(class))
}

#[async_trait]
pub trait WindowManager: Send + Sync + 'static {
    /// Focus the window with the given id.
    async fn focus(&self, window: &WindowId) -> WmResult<()>;

    /// Move the named window to the given workspace.
    async fn move_to_workspace(&self, window: &WindowId, workspace: &WorkspaceId)
    -> WmResult<()>;

    /// Return the id of the currently focused window, or `None` when no
    /// window holds focus (e.g. an empty workspace).
    async fn focused_window(&self) -> WmResult<Option<WindowId>>;

    /// Enumerate every mapped window the compositor knows about.
    async fn list_windows(&self) -> WmResult<Vec<Window>>;

    /// Enumerate every workspace the compositor knows about.
    async fn list_workspaces(&self) -> WmResult<Vec<WorkspaceInfo>>;

    /// Return a snapshot of the currently focused window's class,
    /// title, and the active workspace. Used by the daemon to inject
    /// passive desktop context into the LLM's per-turn system prompt.
    ///
    /// Returns `Ok(None)` when nothing is focused or the backend has
    /// no opinion. The default impl returns `Ok(None)` so backends
    /// without event-snapshot support compile unchanged.
    async fn focused_context(&self) -> WmResult<Option<FocusedWindowContext>> {
        Ok(None)
    }

    /// Send a raw, backend-specific command payload. Used for operations
    /// that have no typed return value and where adding a per-operation
    /// trait method would be a thin wrapper around a string format
    /// (resize, layout, …). Callers are responsible for the syntax —
    /// keep this behind subcommands that document the expected dialect.
    async fn run_raw(&self, payload: &str) -> WmResult<()>;

    /// Enumerate the compositor's outputs (monitors). Sway exposes a
    /// detailed reply via `GET_OUTPUTS`; i3's reply is much sparser, so
    /// the default implementation returns [`WmError::Unsupported`] and
    /// i3-class backends inherit it. Callers that surface this through
    /// a tool (`wm outputs`) must translate the error into a "not
    /// supported" message rather than an empty list, so users see the
    /// difference between an unsupported backend and a connected
    /// machine with zero monitors.
    async fn list_outputs(&self) -> WmResult<Vec<OutputInfo>> {
        Err(WmError::Unsupported("output enumeration"))
    }

    /// Report whether the backend is actually connected to a compositor.
    /// The default `true` covers concrete backends; [`NoWindowManager`]
    /// overrides to `false` so callers can short-circuit with a single
    /// "compositor not connected" message instead of waiting for the
    /// generic [`WmError::Disconnected`] from each operation.
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
    async fn focus(&self, _window: &WindowId) -> WmResult<()> {
        Err(WmError::Disconnected)
    }
    async fn move_to_workspace(
        &self,
        _window: &WindowId,
        _workspace: &WorkspaceId,
    ) -> WmResult<()> {
        Err(WmError::Disconnected)
    }
    async fn focused_window(&self) -> WmResult<Option<WindowId>> {
        Ok(None)
    }
    async fn list_windows(&self) -> WmResult<Vec<Window>> {
        Err(WmError::Disconnected)
    }
    async fn list_workspaces(&self) -> WmResult<Vec<WorkspaceInfo>> {
        Err(WmError::Disconnected)
    }
    async fn run_raw(&self, _payload: &str) -> WmResult<()> {
        Err(WmError::Disconnected)
    }
    async fn list_outputs(&self) -> WmResult<Vec<OutputInfo>> {
        Err(WmError::Disconnected)
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

    #[tokio::test]
    async fn no_window_manager_refuses_list_outputs() {
        assert!(NoWindowManager.list_outputs().await.is_err());
    }

    #[tokio::test]
    async fn default_list_outputs_reports_unsupported() {
        // The default trait impl errors so backends that don't override
        // (i.e. I3Backend) propagate "not supported" rather than an
        // empty Vec — see the OutputInfo doc-comment for the rationale.
        struct MinimalWm;

        #[async_trait]
        impl WindowManager for MinimalWm {
            async fn focus(&self, _w: &WindowId) -> WmResult<()> {
                Ok(())
            }
            async fn move_to_workspace(&self, _w: &WindowId, _ws: &WorkspaceId) -> WmResult<()> {
                Ok(())
            }
            async fn focused_window(&self) -> WmResult<Option<WindowId>> {
                Ok(None)
            }
            async fn list_windows(&self) -> WmResult<Vec<Window>> {
                Ok(Vec::new())
            }
            async fn list_workspaces(&self) -> WmResult<Vec<WorkspaceInfo>> {
                Ok(Vec::new())
            }
            async fn run_raw(&self, _payload: &str) -> WmResult<()> {
                Ok(())
            }
        }

        let err = MinimalWm.list_outputs().await.unwrap_err();
        assert!(matches!(err, WmError::Unsupported(op) if op == "output enumeration"));
    }

    #[test]
    fn no_window_manager_reports_disconnected() {
        assert!(!NoWindowManager.is_connected());
    }

    #[test]
    fn version_is_not_empty() {
        assert!(!version().is_empty());
    }

    #[test]
    fn is_terminal_class_matches_known_emulators() {
        assert!(is_terminal_class("Alacritty"));
        assert!(is_terminal_class("kitty"));
        assert!(is_terminal_class("WezTerm"));
        assert!(is_terminal_class("foot"));
    }

    #[test]
    fn is_terminal_class_is_case_insensitive() {
        // X11 WM_CLASS capitalization varies in practice; users on
        // `alacritty` vs `Alacritty` should both get the terminal hint.
        assert!(is_terminal_class("alacritty"));
        assert!(is_terminal_class("XTERM"));
        assert!(is_terminal_class("xterm"));
    }

    #[test]
    fn is_terminal_class_rejects_non_terminals() {
        assert!(!is_terminal_class("firefox"));
        assert!(!is_terminal_class("code"));
        assert!(!is_terminal_class(""));
        assert!(!is_terminal_class("Slack"));
    }

    #[tokio::test]
    async fn no_window_manager_focused_context_is_none() {
        // Default trait impl returns Ok(None); NoWindowManager inherits
        // it because we don't override.
        assert!(NoWindowManager.focused_context().await.unwrap().is_none());
    }
}
