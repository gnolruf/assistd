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

pub(crate) mod backoff;
pub(crate) mod criteria;
pub mod error;
#[cfg(feature = "i3")]
pub mod i3;
pub(crate) mod snapshot;
#[cfg(feature = "sway")]
pub mod sway;

/// Per-call IPC timeout — `run_command`, `get_tree`, `get_workspaces`,
/// `get_outputs`. A wedged compositor must release within one user
/// turn or the daemon's "every-turn-injects-window-context" path
/// wedges with it.
pub(crate) const WM_IPC_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);
pub use error::{WmError, WmResult};
#[cfg(feature = "i3")]
pub use i3::{I3Backend, I3Handle};
#[cfg(feature = "sway")]
pub use sway::{SwayBackend, SwayHandle};

/// Aggregated shutdown handle for the active WM backend, so the daemon
/// can hold a single `Option<WmHandle>` regardless of which backend
/// was started. Each variant wraps the per-backend supervisor task;
/// [`WmHandle::shutdown`] dispatches.
///
/// The variants are feature-gated to match the backend modules — a
/// build with `--no-default-features --features i3` only sees
/// `WmHandle::I3` and the daemon's match becomes exhaustive against
/// the smaller enum.
pub enum WmHandle {
    #[cfg(feature = "i3")]
    I3(I3Handle),
    #[cfg(feature = "sway")]
    Sway(SwayHandle),
}

impl WmHandle {
    /// Drains the per-backend supervisor task. The daemon should flip
    /// its shutdown watch first; this just awaits the task's exit.
    pub async fn shutdown(self) {
        match self {
            #[cfg(feature = "i3")]
            Self::I3(h) => h.shutdown().await,
            #[cfg(feature = "sway")]
            Self::Sway(h) => h.shutdown().await,
        }
    }
}

/// Opaque identifier for a window in the compositor's namespace.
///
/// Wraps the compositor's container id — i3 emits this as
/// `reply::Node.id: usize`, Sway as `Node.id: i64` — so two windows of
/// the same X11 class (e.g. two Firefox windows) can be addressed
/// independently. The criteria string for both compositors is
/// `[con_id="N"]`.
///
/// `NonZeroU64` is chosen so `Option<WindowId>` is the same size as a
/// bare `u64` and so a zero id (which neither compositor emits, but
/// which would be ambiguous if it ever appeared) is unrepresentable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WindowId(pub std::num::NonZeroU64);

impl WindowId {
    /// Construct a [`WindowId`] from a raw `u64`. Returns `None` for 0
    /// — the compositor never emits zero, so the wrapper takes that as
    /// a structural error signal rather than treating it as valid.
    pub fn new(raw: u64) -> Option<Self> {
        std::num::NonZeroU64::new(raw).map(WindowId)
    }

    /// Unwrap to the raw `u64`. Useful for IPC payload formatting
    /// (`format!("{}", id.get())` produces `"42"`).
    pub fn get(self) -> u64 {
        self.0.get()
    }
}

impl std::fmt::Display for WindowId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.get())
    }
}

/// Returned by [`WindowId::from_str`] when the input is not a positive
/// decimal integer. The caller already has the offending input and
/// renders its own message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseWindowIdError;

impl std::fmt::Display for ParseWindowIdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("expected positive decimal con_id")
    }
}

impl std::error::Error for ParseWindowIdError {}

impl std::str::FromStr for WindowId {
    type Err = ParseWindowIdError;

    /// Parse a decimal con_id. Hex / leading-`0x` is rejected — Sway's
    /// id space is large enough that decimal is the only consistent
    /// rendering, and non-numeric ids would just be confusing for the
    /// LLM that's piping `wm list` output back into `wm focus`.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.parse::<u64>()
            .ok()
            .and_then(WindowId::new)
            .ok_or(ParseWindowIdError)
    }
}

/// Workspace identifier — either a number (i3/sway's `workspace number
/// N` form, robust to renames) or a free-form name (`workspace
/// "<name>"`).
///
/// Modeling this as an enum kills the parse-round-trip the previous
/// type alias forced on `format_workspace_target`: the parser ran on
/// every focus/move dispatch even though the caller already knew which
/// form they meant.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum WorkspaceId {
    /// Numeric workspace, addressed by `workspace number N`.
    Num(u32),
    /// Named workspace, addressed by `workspace "<name>"`.
    Name(String),
}

impl WorkspaceId {
    /// Construct a numeric workspace id.
    pub fn num(n: u32) -> Self {
        WorkspaceId::Num(n)
    }

    /// Construct a named workspace id. Use this for non-numeric
    /// workspace names like `"scratch"` or `"1:web"` — note that
    /// `"1:web"` is a *named* workspace because it does not parse as
    /// `u32` even though it visually starts with a digit.
    pub fn name(s: impl Into<String>) -> Self {
        WorkspaceId::Name(s.into())
    }
}

impl std::fmt::Display for WorkspaceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WorkspaceId::Num(n) => write!(f, "{n}"),
            WorkspaceId::Name(s) => f.write_str(s),
        }
    }
}

impl std::str::FromStr for WorkspaceId {
    type Err = std::convert::Infallible;

    /// Parse-or-name: a string that round-trips through `u32::from_str`
    /// becomes [`WorkspaceId::Num`]; everything else is
    /// [`WorkspaceId::Name`]. Mirrors the previous alias-based
    /// `format_workspace_target` behaviour so call sites don't change.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Ok(n) = s.parse::<u32>() {
            Ok(WorkspaceId::Num(n))
        } else {
            Ok(WorkspaceId::Name(s.to_string()))
        }
    }
}

impl From<&str> for WorkspaceId {
    fn from(s: &str) -> Self {
        s.parse().expect("WorkspaceId parser is infallible")
    }
}

impl From<String> for WorkspaceId {
    fn from(s: String) -> Self {
        s.as_str().into()
    }
}

impl From<u32> for WorkspaceId {
    fn from(n: u32) -> Self {
        WorkspaceId::Num(n)
    }
}

/// One row of [`WindowManager::list_windows`]. The trait's wider methods
/// (`focus`, `move_to_workspace`) take a [`WindowId`] (con_id) alone;
/// this struct is the richer view returned to surfaces (the `wm list`
/// command) that need a human-readable label and workspace context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Window {
    /// Compositor-unique id, usable with [`WindowManager::focus`] /
    /// [`WindowManager::move_to_workspace`]. Emitted as the leading
    /// `<id>` column in `wm list` so the LLM can pipe it back.
    pub id: WindowId,
    /// Human-readable application label — X11 `WM_CLASS` on i3, or
    /// `app_id` (Wayland-native) / `class` (XWayland) on Sway. `None`
    /// when the window has neither set, which is rare. Surfaced as the
    /// second column of `wm list` so the LLM can disambiguate ids by
    /// app name.
    pub app: Option<String>,
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

/// Direction for a width-resize operation. Decoded from the
/// `wm resize <class> <grow|shrink> <px>` user argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeDir {
    Grow,
    Shrink,
}

impl ResizeDir {
    /// i3/sway-syntax keyword used inside the `resize` command payload.
    pub fn as_str(self) -> &'static str {
        match self {
            ResizeDir::Grow => "grow",
            ResizeDir::Shrink => "shrink",
        }
    }
}

impl std::fmt::Display for ResizeDir {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for ResizeDir {
    type Err = ParseResizeDirError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "grow" => Ok(ResizeDir::Grow),
            "shrink" => Ok(ResizeDir::Shrink),
            _ => Err(ParseResizeDirError),
        }
    }
}

/// Returned by [`ResizeDir::from_str`] when the input is neither
/// `"grow"` nor `"shrink"`. Carries no data — the caller already has
/// the offending input and renders its own message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseResizeDirError;

/// Layout to apply to the focused container. Decoded from the
/// `wm layout <name>` user argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    Default,
    Tabbed,
    Stacking,
    SplitH,
    SplitV,
}

impl Layout {
    /// i3/sway-syntax keyword used inside the `layout` command payload.
    pub fn as_str(self) -> &'static str {
        match self {
            Layout::Default => "default",
            Layout::Tabbed => "tabbed",
            Layout::Stacking => "stacking",
            Layout::SplitH => "splith",
            Layout::SplitV => "splitv",
        }
    }
}

impl std::fmt::Display for Layout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for Layout {
    type Err = ParseLayoutError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "default" => Ok(Layout::Default),
            "tabbed" => Ok(Layout::Tabbed),
            "stacking" => Ok(Layout::Stacking),
            "splith" => Ok(Layout::SplitH),
            "splitv" => Ok(Layout::SplitV),
            _ => Err(ParseLayoutError),
        }
    }
}

/// Returned by [`Layout::from_str`] when the input is not a known
/// layout name. Carries no data; the caller already has the offending
/// input and renders its own message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseLayoutError;

/// Snapshot of what the user is currently looking at. Returned by
/// [`WindowManager::focused_context`]. Each field is independently
/// optional because compositors deliver focus / title / workspace
/// state through separate events; a backend may have any subset.
///
/// Built from the backend's cached event snapshot — no IPC round-trip
/// — so callers can read it cheaply on every LLM query.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FocusedWindowContext {
    /// Compositor con_id of the focused window. Surfaced for callers
    /// that want to wire `wm focus $(wm active)` flows; the system-
    /// prompt block does not render it (the model picks windows by
    /// human description, so a numeric id is just token waste).
    pub id: Option<WindowId>,
    /// X11 `WM_CLASS` (or `app_id` on Wayland-native sway) of the
    /// focused window. Used for the "Current desktop context" block
    /// the daemon injects into the LLM's per-turn system prompt.
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
    async fn move_to_workspace(&self, window: &WindowId, workspace: &WorkspaceId) -> WmResult<()>;

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

    /// Resize the named window's width by `pixels`. Backends format the
    /// compositor-specific payload internally so consumers don't reach
    /// into IPC syntax.
    async fn resize_width(
        &self,
        window: &WindowId,
        direction: ResizeDir,
        pixels: u32,
    ) -> WmResult<()>;

    /// Set the layout of the currently-focused container. Acts on the
    /// focus state — no window argument — because that's how
    /// `i3-msg layout …` and `swaymsg layout …` behave.
    async fn set_layout(&self, layout: Layout) -> WmResult<()>;

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
    async fn resize_width(
        &self,
        _window: &WindowId,
        _direction: ResizeDir,
        _pixels: u32,
    ) -> WmResult<()> {
        Err(WmError::Disconnected)
    }
    async fn set_layout(&self, _layout: Layout) -> WmResult<()> {
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

    fn id1() -> WindowId {
        WindowId::new(1).expect("1 is non-zero")
    }

    #[tokio::test]
    async fn no_window_manager_refuses_focus() {
        assert!(NoWindowManager.focus(&id1()).await.is_err());
    }

    #[tokio::test]
    async fn no_window_manager_refuses_move() {
        assert!(
            NoWindowManager
                .move_to_workspace(&id1(), &WorkspaceId::Num(3))
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
    async fn no_window_manager_refuses_resize() {
        let err = NoWindowManager
            .resize_width(&id1(), ResizeDir::Grow, 10)
            .await
            .unwrap_err();
        assert!(matches!(err, WmError::Disconnected));
    }

    #[tokio::test]
    async fn no_window_manager_refuses_layout() {
        let err = NoWindowManager
            .set_layout(Layout::Tabbed)
            .await
            .unwrap_err();
        assert!(matches!(err, WmError::Disconnected));
    }

    #[test]
    fn resize_dir_roundtrips() {
        assert_eq!("grow".parse::<ResizeDir>().unwrap(), ResizeDir::Grow);
        assert_eq!("shrink".parse::<ResizeDir>().unwrap(), ResizeDir::Shrink);
        assert_eq!(ResizeDir::Grow.to_string(), "grow");
        assert!("sideways".parse::<ResizeDir>().is_err());
    }

    #[test]
    fn window_id_parses_decimal_only() {
        assert_eq!(
            "42".parse::<WindowId>().unwrap(),
            WindowId::new(42).unwrap()
        );
        // 0 has no NonZeroU64 representation.
        assert!("0".parse::<WindowId>().is_err());
        // Non-numeric input: rejected.
        assert!("Firefox".parse::<WindowId>().is_err());
        // Negative: rejected (u64 doesn't accept "-1").
        assert!("-1".parse::<WindowId>().is_err());
        // Hex / leading-0x: rejected.
        assert!("0x2a".parse::<WindowId>().is_err());
    }

    #[test]
    fn window_id_display_is_decimal() {
        assert_eq!(WindowId::new(42).unwrap().to_string(), "42");
    }

    #[test]
    fn layout_roundtrips() {
        for (s, l) in [
            ("default", Layout::Default),
            ("tabbed", Layout::Tabbed),
            ("stacking", Layout::Stacking),
            ("splith", Layout::SplitH),
            ("splitv", Layout::SplitV),
        ] {
            assert_eq!(s.parse::<Layout>().unwrap(), l);
            assert_eq!(l.to_string(), s);
        }
        assert!("spinning".parse::<Layout>().is_err());
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
            async fn resize_width(&self, _w: &WindowId, _d: ResizeDir, _p: u32) -> WmResult<()> {
                Ok(())
            }
            async fn set_layout(&self, _l: Layout) -> WmResult<()> {
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
