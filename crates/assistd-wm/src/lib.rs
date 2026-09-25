//! Window manager integration: the [`WindowManager`] trait plus the i3
//! and Sway backends that implement it over their IPC sockets.

use std::convert::Infallible;
use std::fmt;
use std::num::NonZeroU64;
use std::str::FromStr;

use async_trait::async_trait;

#[cfg(any(feature = "i3", feature = "sway"))]
pub(crate) mod backoff;
pub mod criteria;
pub mod error;
#[cfg(feature = "i3")]
pub mod i3;
#[cfg(any(feature = "i3", feature = "sway"))]
mod ipc_backend;
#[cfg(any(feature = "i3", feature = "sway"))]
pub(crate) mod snapshot;
#[cfg(feature = "sway")]
pub mod sway;

pub use error::{TransportError, WmError, WmResult};
#[cfg(feature = "i3")]
pub use i3::{I3Backend, I3Handle};
#[cfg(feature = "sway")]
pub use sway::{SwayBackend, SwayHandle};

/// Per-call IPC timeout, so a wedged compositor cannot block a request
/// indefinitely.
#[cfg(any(feature = "i3", feature = "sway"))]
pub(crate) const WM_IPC_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// `WM_CLASS` values of known terminal emulators, matched
/// case-insensitively.
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
            Self::I3(handle) => handle.shutdown().await,
            #[cfg(feature = "sway")]
            Self::Sway(handle) => handle.shutdown().await,
        }
    }
}

/// Compositor container id (`con_id`), the only unambiguous window
/// handle. Never zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WindowId(pub NonZeroU64);

impl WindowId {
    /// Returns `None` for zero.
    pub fn new(raw: u64) -> Option<Self> {
        NonZeroU64::new(raw).map(WindowId)
    }

    /// The raw `u64`.
    pub fn get(self) -> u64 {
        self.0.get()
    }
}

impl fmt::Display for WindowId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.get())
    }
}

/// Error parsing a [`WindowId`] from a string that is not a positive
/// decimal integer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseWindowIdError;

impl fmt::Display for ParseWindowIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("expected positive decimal con_id")
    }
}

impl std::error::Error for ParseWindowIdError {}

impl FromStr for WindowId {
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
    /// A numeric workspace.
    pub fn num(n: u32) -> Self {
        WorkspaceId::Num(n)
    }

    /// A named workspace, even if `name` looks numeric.
    pub fn name(name: impl Into<String>) -> Self {
        WorkspaceId::Name(name.into())
    }
}

impl fmt::Display for WorkspaceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WorkspaceId::Num(n) => write!(f, "{n}"),
            WorkspaceId::Name(name) => f.write_str(name),
        }
    }
}

impl FromStr for WorkspaceId {
    type Err = Infallible;

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
    /// X11 `WM_CLASS` on i3; `app_id` or XWayland `class` on Sway.
    pub app: Option<String>,
    /// `_NET_WM_NAME` / `WM_NAME`.
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
    /// Focused on its output; one per output on multi-monitor setups.
    pub focused: bool,
    /// Output name (e.g. `"DP-1"`).
    pub output: String,
}

/// One row of [`WindowManager::list_outputs`].
#[derive(Debug, Clone, PartialEq)]
pub struct OutputInfo {
    /// Connector name (e.g. `"DP-1"`).
    pub name: String,
    /// Powered with a mode set.
    pub active: bool,
    /// X11 primary output; always `false` on Sway.
    pub primary: bool,
    /// `(width, height, refresh_mHz)`; `None` for disabled outputs.
    pub current_mode: Option<(u32, u32, u32)>,
    pub scale: Option<f64>,
    /// Workspace visible on this output; `None` for disabled outputs.
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

impl fmt::Display for ResizeDir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ResizeDir {
    type Err = ParseResizeDirError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "grow" => Ok(ResizeDir::Grow),
            "shrink" => Ok(ResizeDir::Shrink),
            _ => Err(ParseResizeDirError),
        }
    }
}

/// Error parsing a [`ResizeDir`] from a string other than `"grow"` or
/// `"shrink"`.
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

impl fmt::Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Layout {
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

/// Error parsing a [`Layout`] from a string that is not a known layout
/// name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseLayoutError;

/// Focused window and workspace, read from the backend's event cache.
/// Fields are independently optional since each arrives in its own event.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FocusedWindowContext {
    pub id: Option<WindowId>,
    /// X11 `WM_CLASS`, or `app_id` on Wayland-native Sway.
    pub class: Option<String>,
    /// `_NET_WM_NAME` / `WM_NAME`.
    pub title: Option<String>,
    pub workspace: Option<String>,
}

/// True when `class` is a known terminal emulator.
pub fn is_terminal_class(class: &str) -> bool {
    TERMINAL_CLASSES
        .iter()
        .any(|known| known.eq_ignore_ascii_case(class))
}

/// How the compositor matches the window to act on. Becomes the
/// `[key="value"]` prefix on the IPC command.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PlacementCriteria {
    /// Wayland `app_id`; i3 rewrites this to a title match.
    AppId(String),
    /// X11 `WM_CLASS`.
    Class(String),
    /// Exact `_NET_WM_NAME` / `WM_NAME`.
    Title(String),
    ConId(WindowId),
}

/// Corner of the focused output a placed window anchors to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnchorCorner {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
    Center,
}

/// Rectangle in global screen coordinates and logical pixels, with
/// `(x, y)` the top-left corner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

/// Compositor window lifecycle event.
#[derive(Debug, Clone)]
pub enum WindowEvent {
    /// A window was mapped (i3/sway `window::new`).
    Opened {
        id: WindowId,
        title: Option<String>,
        class: Option<String>,
        app_id: Option<String>,
    },
    /// A window's title changed.
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

/// Async interface to a window manager. Each method is one IPC call; one
/// that times out returns [`WmError::Timeout`] and triggers reconnection.
#[async_trait]
pub trait WindowManager: Send + Sync + 'static {
    /// Focus the window with the given id.
    async fn focus(&self, window: &WindowId) -> WmResult<()>;

    /// Move the window to the given workspace.
    async fn move_to_workspace(&self, window: &WindowId, workspace: &WorkspaceId) -> WmResult<()>;

    /// The focused window, or `None` when nothing holds focus.
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

    /// Enumerate outputs; [`WmError::Unsupported`] when the backend
    /// cannot.
    async fn list_outputs(&self) -> WmResult<Vec<OutputInfo>> {
        Err(WmError::Unsupported("output enumeration"))
    }

    /// Pixel rect of the workspace focused on the active output.
    async fn focused_workspace_rect(&self) -> WmResult<Rect> {
        Err(WmError::Unsupported("focused workspace rect"))
    }

    /// Scale factor of the focused output; `1.0` when it cannot be
    /// determined.
    async fn focused_output_scale(&self) -> WmResult<f64> {
        Ok(1.0)
    }

    /// Float, resize, and move the matched window in one IPC payload.
    /// Criteria matching no window still return `Ok(())`.
    async fn place_floating(
        &self,
        _criteria: &PlacementCriteria,
        _anchor: PlacementAnchor,
    ) -> WmResult<()> {
        Err(WmError::Unsupported("floating placement"))
    }

    /// Whether the backend is connected to a compositor.
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
mod tests;
