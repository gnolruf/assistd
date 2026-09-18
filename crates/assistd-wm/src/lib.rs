#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::print_stdout,
        clippy::print_stderr
    )
)]

//! Window manager integration: the [`WindowManager`] trait plus the i3
//! and Sway backends that implement it over their IPC sockets.

use async_trait::async_trait;

pub(crate) mod backoff;
pub mod criteria;
pub mod error;
#[cfg(feature = "i3")]
pub mod i3;
pub(crate) mod snapshot;
#[cfg(feature = "sway")]
pub mod sway;

/// Per-call IPC timeout. A wedged compositor must not stall the
/// caller's turn along with it.
pub(crate) const WM_IPC_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);
pub use error::{WmError, WmResult};
#[cfg(feature = "i3")]
pub use i3::{I3Backend, I3Handle};
#[cfg(feature = "sway")]
pub use sway::{SwayBackend, SwayHandle};

/// Shutdown handle for whichever backend was started. Each variant
/// wraps that backend's supervisor task.
pub enum WmHandle {
    #[cfg(feature = "i3")]
    I3(I3Handle),
    #[cfg(feature = "sway")]
    Sway(SwayHandle),
}

impl WmHandle {
    /// Awaits the supervisor task. Flip the shutdown watch first or
    /// this blocks until the compositor connection drops.
    pub async fn shutdown(self) {
        match self {
            #[cfg(feature = "i3")]
            Self::I3(h) => h.shutdown().await,
            #[cfg(feature = "sway")]
            Self::Sway(h) => h.shutdown().await,
        }
    }
}

/// Compositor container id (`con_id`). Two windows of the same class
/// have distinct ids, so this is the only unambiguous window handle.
/// Neither compositor emits zero, so `NonZeroU64` makes an invalid id
/// unrepresentable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WindowId(pub std::num::NonZeroU64);

impl WindowId {
    /// Returns `None` for zero.
    pub fn new(raw: u64) -> Option<Self> {
        std::num::NonZeroU64::new(raw).map(WindowId)
    }

    /// The raw `u64`.
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
/// decimal integer.
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

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.parse::<u64>()
            .ok()
            .and_then(WindowId::new)
            .ok_or(ParseWindowIdError)
    }
}

/// Workspace identifier: either a number (`workspace number N`, robust
/// to renames) or a free-form name (`workspace "<name>"`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum WorkspaceId {
    /// Numeric workspace, addressed by `workspace number N`.
    Num(u32),
    /// Named workspace, addressed by `workspace "<name>"`.
    Name(String),
}

impl WorkspaceId {
    pub fn num(n: u32) -> Self {
        WorkspaceId::Num(n)
    }

    /// A named workspace. `"1:web"` is a name, not a number, because
    /// it does not parse as `u32`.
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

    /// A string that parses as `u32` becomes [`WorkspaceId::Num`];
    /// everything else is [`WorkspaceId::Name`].
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

/// One row of [`WindowManager::list_windows`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Window {
    pub id: WindowId,
    /// X11 `WM_CLASS` on i3; `app_id` (Wayland-native) or `class`
    /// (XWayland) on Sway. `None` when the window sets neither.
    pub app: Option<String>,
    /// `_NET_WM_NAME` / `WM_NAME`. Missing on some transient or just-
    /// mapped windows.
    pub title: Option<String>,
    /// `None` for scratchpad or otherwise un-anchored windows.
    pub workspace: Option<String>,
}

/// One row of [`WindowManager::list_workspaces`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceInfo {
    /// `-1` when the workspace name does not begin with a number.
    pub num: i32,
    /// User-visible label (e.g. `"1"`, `"1:web"`, `"scratch"`).
    pub name: String,
    /// True when this workspace holds focus on its output. Multi-monitor
    /// setups have one focused workspace per output.
    pub focused: bool,
    /// Output name (e.g. `"DP-1"`).
    pub output: String,
}

/// One row of [`WindowManager::list_outputs`].
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

/// Direction for a width-resize operation.
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
/// `"grow"` nor `"shrink"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseResizeDirError;

/// Layout to apply to the focused container.
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
/// layout name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseLayoutError;

/// Snapshot of the focused window and workspace, read from the
/// backend's event cache without an IPC round-trip. Each field is
/// independently optional because compositors deliver focus, title,
/// and workspace state through separate events.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FocusedWindowContext {
    pub id: Option<WindowId>,
    /// X11 `WM_CLASS`, or `app_id` on Wayland-native Sway.
    pub class: Option<String>,
    /// `_NET_WM_NAME` / `WM_NAME`.
    pub title: Option<String>,
    pub workspace: Option<String>,
}

/// `WM_CLASS` values of known terminal emulators, matched
/// case-insensitively because compositors and apps disagree on
/// capitalisation (`xterm` vs `XTerm`).
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

/// True when `class` is a known terminal emulator. Unknown classes are
/// reported as non-terminal.
pub fn is_terminal_class(class: &str) -> bool {
    TERMINAL_CLASSES
        .iter()
        .any(|t| t.eq_ignore_ascii_case(class))
}

/// How the compositor matches the window to act on. Becomes the
/// `[key="value"]` prefix on the IPC command.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PlacementCriteria {
    /// Wayland `app_id`. i3 is X11-only and rewrites this to a title
    /// match.
    AppId(String),
    /// X11 `WM_CLASS`.
    Class(String),
    /// Exact `_NET_WM_NAME` / `WM_NAME` match. The escape hatch when a
    /// toolkit leaves `WM_CLASS` empty on X11 (egui-winit 0.34 does).
    Title(String),
    /// Match a specific compositor container id (`[con_id="…"]`).
    ConId(WindowId),
}

/// Corner of the focused output a placed window anchors to. Offsets
/// are measured inward from this corner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnchorCorner {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
    Center,
}

/// Pixel rectangle: `(x, y)` is the top-left corner in global screen
/// coordinates, `(width, height)` the size in logical pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

/// Compositor window lifecycle event, broadcast by each backend from
/// its IPC event stream.
#[derive(Debug, Clone)]
pub enum WindowEvent {
    /// A window was mapped (i3/sway `window::new`).
    Opened {
        id: WindowId,
        title: Option<String>,
        class: Option<String>,
        app_id: Option<String>,
    },
    /// A window's title (`_NET_WM_NAME` / Wayland title) changed.
    TitleChanged {
        id: WindowId,
        new_title: Option<String>,
    },
    /// A window was unmapped or destroyed.
    Closed { id: WindowId },
}

impl WindowEvent {
    /// The window id if this is an `Opened` event matching `criteria`.
    pub fn matches_opened(&self, criteria: &PlacementCriteria) -> Option<WindowId> {
        let Self::Opened {
            id,
            title,
            class,
            app_id,
        } = self
        else {
            return None;
        };
        let matched = match criteria {
            PlacementCriteria::Title(want) => title.as_deref() == Some(want.as_str()),
            PlacementCriteria::Class(want) => class.as_deref() == Some(want.as_str()),
            PlacementCriteria::AppId(want) => app_id.as_deref() == Some(want.as_str()),
            PlacementCriteria::ConId(want) => id == want,
        };
        matched.then_some(*id)
    }
}

/// Where to place a floating window: a corner anchor, offsets from
/// it, and the target size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlacementAnchor {
    pub corner: AnchorCorner,
    /// Positive shifts right, negative left.
    pub offset_x: i32,
    /// Positive shifts down, negative up.
    pub offset_y: i32,
    pub width: u32,
    pub height: u32,
}

/// Async interface to a window manager. Each method is one IPC
/// operation bounded by [`WM_IPC_TIMEOUT`]; a call that exceeds it
/// returns [`WmError::Timeout`] and triggers reconnection.
#[async_trait]
pub trait WindowManager: Send + Sync + 'static {
    /// Focus the window with the given id.
    async fn focus(&self, window: &WindowId) -> WmResult<()>;

    /// Move the window to the given workspace.
    async fn move_to_workspace(&self, window: &WindowId, workspace: &WorkspaceId) -> WmResult<()>;

    /// Return the id of the currently focused window, or `None` when no
    /// window holds focus (e.g. an empty workspace).
    async fn focused_window(&self) -> WmResult<Option<WindowId>>;

    /// Enumerate every mapped window the compositor knows about.
    async fn list_windows(&self) -> WmResult<Vec<Window>>;

    /// Enumerate every workspace the compositor knows about.
    async fn list_workspaces(&self) -> WmResult<Vec<WorkspaceInfo>>;

    /// Snapshot of the focused window and active workspace. `Ok(None)`
    /// when nothing is focused or the backend keeps no snapshot.
    async fn focused_context(&self) -> WmResult<Option<FocusedWindowContext>> {
        Ok(None)
    }

    /// Resize the window's width by `pixels`.
    async fn resize_width(
        &self,
        window: &WindowId,
        direction: ResizeDir,
        pixels: u32,
    ) -> WmResult<()>;

    /// Set the layout of the focused container.
    async fn set_layout(&self, layout: Layout) -> WmResult<()>;

    /// Enumerate outputs (monitors). Backends without a rich output
    /// reply return [`WmError::Unsupported`], which is distinct from an
    /// empty list.
    async fn list_outputs(&self) -> WmResult<Vec<OutputInfo>> {
        Err(WmError::Unsupported("output enumeration"))
    }

    /// Pixel rect of the workspace focused on the active output.
    async fn focused_workspace_rect(&self) -> WmResult<Rect> {
        Err(WmError::Unsupported("focused workspace rect"))
    }

    /// Scale factor of the focused output (`1.0`, `1.5`, `2.0`, ...).
    /// Sway reports it over IPC; i3 derives it from
    /// `WINIT_X11_SCALE_FACTOR`, then `Xft.dpi`, then the output's
    /// physical size from `xrandr`. Backends that cannot determine it
    /// report `1.0` rather than fail.
    async fn focused_output_scale(&self) -> WmResult<f64> {
        Ok(1.0)
    }

    /// Float the matched window, resize it to the anchor's size, and
    /// move it to the anchored corner, as one chained IPC payload.
    ///
    /// Both i3 and Sway treat criteria matching no window as silent
    /// success, so `Ok(())` does not mean the window was placed.
    /// Callers needing at-least-once placement should call this after
    /// the window is mapped and retry once shortly after.
    async fn place_floating(
        &self,
        _criteria: &PlacementCriteria,
        _anchor: PlacementAnchor,
    ) -> WmResult<()> {
        Err(WmError::Unsupported("floating placement"))
    }

    /// Whether the backend is connected to a compositor. Lets callers
    /// short-circuit instead of collecting a [`WmError::Disconnected`]
    /// per operation.
    fn is_connected(&self) -> bool {
        true
    }
}

/// Placeholder [`WindowManager`] that refuses every operation.
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
    async fn place_floating(
        &self,
        _criteria: &PlacementCriteria,
        _anchor: PlacementAnchor,
    ) -> WmResult<()> {
        Err(WmError::Disconnected)
    }
    async fn focused_workspace_rect(&self) -> WmResult<Rect> {
        Err(WmError::Disconnected)
    }
    async fn focused_output_scale(&self) -> WmResult<f64> {
        Ok(1.0)
    }
    fn is_connected(&self) -> bool {
        false
    }
}

/// The crate version.
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
        assert!("0".parse::<WindowId>().is_err());
        assert!("Firefox".parse::<WindowId>().is_err());
        assert!("-1".parse::<WindowId>().is_err());
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
    async fn no_window_manager_refuses_place_floating() {
        let err = NoWindowManager
            .place_floating(
                &PlacementCriteria::AppId("dev.assistd.popup".into()),
                PlacementAnchor {
                    corner: AnchorCorner::TopRight,
                    offset_x: -10,
                    offset_y: 10,
                    width: 360,
                    height: 120,
                },
            )
            .await
            .unwrap_err();
        assert!(matches!(err, WmError::Disconnected));
    }

    #[test]
    fn window_event_matches_opened_by_title() {
        let ev = WindowEvent::Opened {
            id: WindowId::new(7).unwrap(),
            title: Some("dev.assistd.popup".into()),
            class: None,
            app_id: None,
        };
        assert_eq!(
            ev.matches_opened(&PlacementCriteria::Title("dev.assistd.popup".into())),
            WindowId::new(7)
        );
        assert_eq!(
            ev.matches_opened(&PlacementCriteria::Title("other".into())),
            None
        );
        assert_eq!(
            ev.matches_opened(&PlacementCriteria::Class("dev.assistd.popup".into())),
            None,
            "title-only event should not match class criteria"
        );
    }

    #[test]
    fn window_event_matches_opened_by_app_id_and_class_and_con_id() {
        let id = WindowId::new(11).unwrap();
        let ev = WindowEvent::Opened {
            id,
            title: None,
            class: Some("Firefox".into()),
            app_id: Some("org.mozilla.firefox".into()),
        };
        assert_eq!(
            ev.matches_opened(&PlacementCriteria::Class("Firefox".into())),
            Some(id)
        );
        assert_eq!(
            ev.matches_opened(&PlacementCriteria::AppId("org.mozilla.firefox".into())),
            Some(id)
        );
        assert_eq!(ev.matches_opened(&PlacementCriteria::ConId(id)), Some(id));
        assert_eq!(
            ev.matches_opened(&PlacementCriteria::ConId(WindowId::new(99).unwrap())),
            None
        );
    }

    #[test]
    fn window_event_non_opened_variants_never_match() {
        let id = WindowId::new(3).unwrap();
        assert_eq!(
            WindowEvent::Closed { id }.matches_opened(&PlacementCriteria::ConId(id)),
            None
        );
        assert_eq!(
            WindowEvent::TitleChanged {
                id,
                new_title: Some("x".into())
            }
            .matches_opened(&PlacementCriteria::Title("x".into())),
            None
        );
    }

    #[tokio::test]
    async fn default_place_floating_reports_unsupported() {
        // A minimal impl that opts in to nothing inherits the default
        // place_floating, which surfaces Unsupported so Hyprland-class
        // backends compile before adding their own floating semantics.
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

        let err = MinimalWm
            .place_floating(
                &PlacementCriteria::AppId("x".into()),
                PlacementAnchor {
                    corner: AnchorCorner::Center,
                    offset_x: 0,
                    offset_y: 0,
                    width: 100,
                    height: 100,
                },
            )
            .await
            .unwrap_err();
        assert!(matches!(err, WmError::Unsupported(op) if op == "floating placement"));
    }

    #[tokio::test]
    async fn default_list_outputs_reports_unsupported() {
        // The default trait impl errors so backends that don't override
        // (i.e. I3Backend) propagate "not supported" rather than an
        // empty Vec; see the OutputInfo doc-comment for the rationale.
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
        assert!(NoWindowManager.focused_context().await.unwrap().is_none());
    }
}
