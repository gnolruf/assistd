//! i3 backend for [`crate::WindowManager`], over `tokio-i3ipc`. One
//! socket serves commands; a second feeds the event stream that keeps
//! the focus snapshot current.

use std::sync::Arc;

use async_trait::async_trait;
use futures_util::StreamExt;
use futures_util::stream::BoxStream;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio_i3ipc::{
    I3,
    event::{Event, Subscribe, WindowChange, WindowData, WorkspaceChange},
    reply,
};

use crate::criteria::{format_focus, format_layout, format_move_to_workspace, format_resize_width};
use crate::ipc_backend::{IpcBackend, IpcEvent, IpcProtocol, NodeIdentity, OpLabels, Workspace};
use crate::snapshot::WindowChangeKind;
use crate::{
    FocusedWindowContext, Layout, PlacementAnchor, PlacementCriteria, Rect, ResizeDir,
    TransportError, Window, WindowEvent, WindowId, WindowManager, WmError, WmResult, WorkspaceId,
    WorkspaceInfo,
};

/// [`WindowManager`] over a single i3 IPC command socket.
pub struct I3Backend {
    ipc: Arc<IpcBackend<I3Ipc>>,
}

/// The backend plus its supervisor task, returned by [`I3Backend::start`].
pub struct I3Handle {
    pub backend: Arc<I3Backend>,
    supervisor_task: JoinHandle<()>,
}

impl I3Handle {
    /// Awaits the supervisor task. Flip the shutdown watch first or
    /// this blocks until the socket drops.
    pub async fn shutdown(self) {
        let _ = self.supervisor_task.await;
    }
}

impl I3Backend {
    /// Connect to the i3 IPC sockets, seed the focus snapshot, and
    /// spawn the supervisor that drives events and reconnects on
    /// socket drops. Errors only when the initial connect fails;
    /// later socket failures are handled by the supervisor.
    pub async fn start(shutdown: watch::Receiver<bool>) -> WmResult<I3Handle> {
        let (ipc, supervisor_task) = IpcBackend::start(I3Ipc, shutdown).await?;
        Ok(I3Handle {
            backend: Arc::new(Self { ipc }),
            supervisor_task,
        })
    }

    /// Scale factor derived from the focused output's physical size in
    /// `xrandr`, using winit's DPI quantisation. `None` when no
    /// workspace is focused or `xrandr` doesn't list the output.
    async fn focused_output_randr_scale(&self) -> Option<f64> {
        let output_name = self.focused_output_name().await.ok().flatten()?;
        let xrandr = run_xrandr_query().await?;
        let (pixels, mm) = parse_xrandr_output_size(&xrandr, &output_name)?;
        Some(calc_randr_scale(pixels, mm))
    }

    /// Output hosting the focused workspace; `Ok(None)` when none is.
    async fn focused_output_name(&self) -> WmResult<Option<String>> {
        Ok(self
            .ipc
            .workspaces("i3 GET_WORKSPACES (focused output name)")
            .await?
            .into_iter()
            .find(|w| w.info.focused)
            .map(|w| w.info.output))
    }
}

#[async_trait]
impl WindowManager for I3Backend {
    async fn focus(&self, window: &WindowId) -> WmResult<()> {
        self.ipc.run_command(&format_focus(window)).await
    }

    async fn move_to_workspace(&self, window: &WindowId, workspace: &WorkspaceId) -> WmResult<()> {
        self.ipc
            .run_command(&format_move_to_workspace(window, workspace))
            .await
    }

    async fn focused_window(&self) -> WmResult<Option<WindowId>> {
        Ok(self.ipc.focused_window().await)
    }

    async fn focused_context(&self) -> WmResult<Option<FocusedWindowContext>> {
        Ok(self.ipc.focused_context().await)
    }

    async fn list_windows(&self) -> WmResult<Vec<Window>> {
        self.ipc.list_windows().await
    }

    async fn list_workspaces(&self) -> WmResult<Vec<WorkspaceInfo>> {
        self.ipc.list_workspaces().await
    }

    async fn resize_width(
        &self,
        window: &WindowId,
        direction: ResizeDir,
        pixels: u32,
    ) -> WmResult<()> {
        self.ipc
            .run_command(&format_resize_width(window, direction, pixels))
            .await
    }

    async fn set_layout(&self, layout: Layout) -> WmResult<()> {
        self.ipc.run_command(&format_layout(layout)).await
    }

    async fn focused_workspace_rect(&self) -> WmResult<Rect> {
        self.ipc.focused_workspace_rect().await
    }

    async fn focused_output_scale(&self) -> WmResult<f64> {
        if let Some(s) = env_scale("WINIT_X11_SCALE_FACTOR") {
            return Ok(s);
        }
        if let Some(out) = run_xrdb_query().await
            && let Some(s) = parse_xft_dpi_scale(&out)
        {
            return Ok(s);
        }
        if let Some(s) = self.focused_output_randr_scale().await {
            return Ok(s);
        }
        Ok(1.0)
    }

    async fn place_floating(
        &self,
        criteria: &PlacementCriteria,
        anchor: PlacementAnchor,
    ) -> WmResult<()> {
        self.ipc
            .place_floating(&translate_criteria_for_i3(criteria), anchor)
            .await
    }
}

struct I3Ipc;

impl IpcProtocol for I3Ipc {
    type Cmd = I3;
    type Events = BoxStream<'static, std::io::Result<Event>>;
    type Node = reply::Node;

    const NAME: &'static str = "i3";
    const OPS: OpLabels = OpLabels {
        run_command: "i3 RUN_COMMAND",
        get_tree: "i3 GET_TREE",
        get_tree_window_rect: "i3 GET_TREE (window rect)",
        get_workspaces: "i3 GET_WORKSPACES",
        get_workspaces_focused_rect: "i3 GET_WORKSPACES (focused rect)",
    };

    async fn connect(&self) -> WmResult<(I3, Self::Events)> {
        let cmd = I3::connect()
            .await
            .map_err(|e| WmError::ipc("connect to i3 IPC (cmd socket)", e))?;
        let mut events_conn = I3::connect()
            .await
            .map_err(|e| WmError::ipc("connect to i3 IPC (events socket)", e))?;
        events_conn
            .subscribe([Subscribe::Window, Subscribe::Workspace])
            .await
            .map_err(|e| WmError::ipc("subscribe to i3 window+workspace events", e))?;
        Ok((cmd, events_conn.listen().boxed()))
    }

    async fn next_event(events: &mut Self::Events) -> Option<Result<IpcEvent, TransportError>> {
        Some(events.next().await?.map(ipc_event).map_err(Into::into))
    }

    async fn run_command(
        cmd: &mut I3,
        payload: &str,
    ) -> Result<Result<(), String>, TransportError> {
        let results = cmd.run_command(payload).await?;
        Ok(match results.into_iter().find(|r| !r.success) {
            Some(r) => Err(r.error.unwrap_or_else(|| "unknown error".into())),
            None => Ok(()),
        })
    }

    async fn get_tree(cmd: &mut I3) -> Result<reply::Node, TransportError> {
        Ok(cmd.get_tree().await?)
    }

    async fn get_workspaces(cmd: &mut I3) -> Result<Vec<Workspace>, TransportError> {
        Ok(cmd
            .get_workspaces()
            .await?
            .into_iter()
            .map(|w| Workspace {
                rect: Rect {
                    x: w.rect.x as i32,
                    y: w.rect.y as i32,
                    width: w.rect.width.max(0) as u32,
                    height: w.rect.height.max(0) as u32,
                },
                info: WorkspaceInfo {
                    num: w.num,
                    name: w.name,
                    focused: w.focused,
                    output: w.output,
                },
            })
            .collect())
    }

    fn children(node: &reply::Node) -> impl Iterator<Item = &reply::Node> {
        node.nodes.iter().chain(node.floating_nodes.iter())
    }

    fn is_focused(node: &reply::Node) -> bool {
        node.focused
    }

    fn identity(node: &reply::Node) -> NodeIdentity {
        NodeIdentity {
            id: WindowId::new(node.id as u64),
            class: node
                .window_properties
                .as_ref()
                .and_then(|p| p.class.clone()),
            title: node.name.clone(),
        }
    }

    fn window_rect(node: &reply::Node, criteria: &PlacementCriteria) -> Option<Rect> {
        (node.window.is_some() && node_matches(node, criteria)).then(|| Rect {
            x: node.rect.x as i32,
            y: node.rect.y as i32,
            width: node.rect.width.max(0) as u32,
            height: node.rect.height.max(0) as u32,
        })
    }

    fn collect_windows(tree: &reply::Node) -> Vec<Window> {
        let mut out = Vec::new();
        collect_windows(tree, None, &mut out);
        out
    }
}

fn ipc_event(event: Event) -> IpcEvent {
    match event {
        Event::Window(w) => IpcEvent::Window {
            focus: focus_change(&w),
            event: window_event_from_i3(&w),
        },
        Event::Workspace(data) if matches!(data.change, WorkspaceChange::Focus) => {
            IpcEvent::WorkspaceFocused(data.current.and_then(|n| n.name))
        }
        _ => IpcEvent::Ignored,
    }
}

fn focus_change(w: &WindowData) -> Option<(WindowChangeKind, NodeIdentity)> {
    let kind = match w.change {
        WindowChange::Focus => WindowChangeKind::Focus,
        WindowChange::Title => WindowChangeKind::Title,
        WindowChange::Close => WindowChangeKind::Close,
        _ => return None,
    };
    Some((kind, I3Ipc::identity(&w.container)))
}

fn window_event_from_i3(w: &WindowData) -> Option<WindowEvent> {
    let id = WindowId::new(w.container.id as u64)?;
    match w.change {
        WindowChange::New => {
            let props = w.container.window_properties.as_ref();
            let class = props.and_then(|p| p.class.clone());
            Some(WindowEvent::Opened {
                id,
                title: w.container.name.clone(),
                class,
                app_id: None,
            })
        }
        WindowChange::Title => Some(WindowEvent::TitleChanged {
            id,
            new_title: w.container.name.clone(),
        }),
        WindowChange::Close => Some(WindowEvent::Closed { id }),
        _ => None,
    }
}

/// i3 has no `app_id` criterion, and egui-winit 0.34 leaves `WM_CLASS`
/// empty on X11, so `AppId` maps onto `title`. Callers must set their
/// window title to the same string they pass as `AppId`.
fn translate_criteria_for_i3(c: &PlacementCriteria) -> PlacementCriteria {
    match c {
        PlacementCriteria::AppId(s) => PlacementCriteria::Title(s.clone()),
        other => other.clone(),
    }
}

fn node_matches(node: &reply::Node, criteria: &PlacementCriteria) -> bool {
    let props = node.window_properties.as_ref();
    match criteria {
        PlacementCriteria::Title(want) => node
            .name
            .as_deref()
            .or_else(|| props.and_then(|p| p.title.as_deref()))
            .is_some_and(|t| t == want),
        PlacementCriteria::Class(want) => props
            .and_then(|p| p.class.as_deref())
            .is_some_and(|c| c == want),
        // AppId is rewritten to Title before reaching here; ConId is
        // matched by id elsewhere. No-match so a misuse fails loudly.
        _ => false,
    }
}

fn env_scale(name: &str) -> Option<f64> {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|s| s.is_finite() && *s > 0.0)
}

async fn run_xrdb_query() -> Option<String> {
    tokio::task::spawn_blocking(|| {
        let out = std::process::Command::new("xrdb")
            .arg("-query")
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        String::from_utf8(out.stdout).ok()
    })
    .await
    .ok()
    .flatten()
}

async fn run_xrandr_query() -> Option<String> {
    tokio::task::spawn_blocking(|| {
        let out = std::process::Command::new("xrandr")
            .arg("--query")
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        String::from_utf8(out.stdout).ok()
    })
    .await
    .ok()
    .flatten()
}

/// `Xft.dpi / 96`, clamped to `>= 1.0`.
fn parse_xft_dpi_scale(xrdb_output: &str) -> Option<f64> {
    xrdb_output
        .lines()
        .find_map(|line| line.strip_prefix("Xft.dpi:"))
        .and_then(|rest| rest.trim().parse::<f64>().ok())
        .filter(|dpi| dpi.is_finite() && *dpi > 0.0)
        .map(|dpi| (dpi / 96.0).max(1.0))
}

/// Pixel size and physical size in millimetres of a connected output,
/// from its `<name> connected WxH+X+Y ... <W>mm x <H>mm` line.
fn parse_xrandr_output_size(
    xrandr_output: &str,
    output_name: &str,
) -> Option<((u32, u32), (u64, u64))> {
    let prefix = format!("{output_name} connected");
    let line = xrandr_output.lines().find(|l| l.starts_with(&prefix))?;

    let pixels = line.split_whitespace().find_map(parse_geometry_size)?;

    let tokens: Vec<&str> = line.split_whitespace().collect();
    let mm = tokens.windows(3).rev().find_map(|w| {
        match (w[0].strip_suffix("mm"), w[1], w[2].strip_suffix("mm")) {
            (Some(wmm), "x", Some(hmm)) => {
                let w: u64 = wmm.parse().ok()?;
                let h: u64 = hmm.parse().ok()?;
                Some((w, h))
            }
            _ => None,
        }
    })?;
    Some((pixels, mm))
}

/// `(width, height)` from a `WIDTHxHEIGHT+X+Y` token. The `+X+Y` is
/// required so a mode-list entry like `2560x1440` doesn't match.
fn parse_geometry_size(token: &str) -> Option<(u32, u32)> {
    let plus = token.find('+')?;
    let dims = &token[..plus];
    let (w, h) = dims.split_once('x')?;
    Some((w.parse().ok()?, h.parse().ok()?))
}

/// Winit's RandR DPI formula (`x11/util/randr.rs::calc_dpi_factor`),
/// quantised to 1/12 steps so monitors with minor EDID rounding agree.
/// `1.0` for zero-sized inputs or absurd results.
fn calc_randr_scale(pixels: (u32, u32), mm: (u64, u64)) -> f64 {
    let (px_w, px_h) = (pixels.0 as f64, pixels.1 as f64);
    let (mm_w, mm_h) = (mm.0 as f64, mm.1 as f64);
    if mm_w == 0.0 || mm_h == 0.0 {
        return 1.0;
    }
    let ppmm = ((px_w * px_h) / (mm_w * mm_h)).sqrt();
    let factor = ((ppmm * (12.0 * 25.4 / 96.0)).round() / 12.0).max(1.0);
    if factor.is_finite() && factor <= 20.0 {
        factor
    } else {
        1.0
    }
}

fn collect_windows(node: &reply::Node, current_ws: Option<&str>, out: &mut Vec<Window>) {
    let next_ws = if matches!(node.node_type, reply::NodeType::Workspace) {
        node.name.as_deref()
    } else {
        current_ws
    };

    if node.window.is_some()
        && let Some(id) = WindowId::new(node.id as u64)
    {
        let (app, title) = match node.window_properties.as_ref() {
            Some(props) => (props.class.clone(), props.title.clone()),
            None => (None, None),
        };
        out.push(Window {
            id,
            app,
            title,
            workspace: next_ws.map(|s| s.to_string()),
        });
    }

    for child in I3Ipc::children(node) {
        collect_windows(child, next_ws, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translate_criteria_rewrites_only_app_id_to_title() {
        let con = PlacementCriteria::ConId(WindowId::new(42).unwrap());
        for (input, expected) in [
            (
                PlacementCriteria::AppId("dev.assistd.popup".into()),
                PlacementCriteria::Title("dev.assistd.popup".into()),
            ),
            (
                PlacementCriteria::Class("Firefox".into()),
                PlacementCriteria::Class("Firefox".into()),
            ),
            (
                PlacementCriteria::Title("Inbox".into()),
                PlacementCriteria::Title("Inbox".into()),
            ),
            (con.clone(), con),
        ] {
            assert_eq!(translate_criteria_for_i3(&input), expected, "{input:?}");
        }
    }

    #[test]
    fn xft_dpi_scale_from_xrdb_output() {
        for (xrdb, expected) in [
            (
                "*color0:\t#000000\nXft.dpi:\t144\nXft.antialias:\t1\n",
                Some(1.5),
            ),
            ("Xft.dpi:\t72\n", Some(1.0)),
            ("", None),
            ("Xft.antialias:\t1\n", None),
            ("Xft.dpi:\tnope\n", None),
            ("Xft.dpi:\t-50\n", None),
            ("Xft.dpi:\t0\n", None),
        ] {
            assert_eq!(parse_xft_dpi_scale(xrdb), expected, "{xrdb:?}");
        }
    }

    #[test]
    fn xrandr_output_size_for_connected_outputs_only() {
        let full = "\
Screen 0: minimum 8 x 8, current 2560 x 1440, maximum 32767 x 32767
HDMI-0 disconnected primary (normal left inverted right x axis y axis)
DP-0 connected 2560x1440+0+0 (normal left inverted right x axis y axis) 587mm x 330mm
   2560x1440     59.95*+ 280.00   120.00
";
        for (xrandr, output, expected) in [
            (full, "DP-0", Some(((2560, 1440), (587, 330)))),
            (full, "HDMI-0", None),
            (full, "DP-1", None),
            (
                "DP-0 connected 2560x1440+0+0 (normal left inverted right)\n",
                "DP-0",
                None,
            ),
        ] {
            assert_eq!(
                parse_xrandr_output_size(xrandr, output),
                expected,
                "{output} in {xrandr:?}"
            );
        }
    }

    #[test]
    fn randr_scale_matches_winit_quantization() {
        for (pixels, mm, expected) in [
            // ppmm ≈ 4.32; round(4.32 * 12 * 25.4 / 96) / 12 = 14 / 12.
            ((2560, 1440), (587, 330), 14.0 / 12.0),
            ((1024, 768), (400, 300), 1.0),
            ((1920, 1080), (0, 0), 1.0),
            ((1920, 1080), (500, 0), 1.0),
        ] {
            assert_eq!(
                calc_randr_scale(pixels, mm),
                expected,
                "{pixels:?} on {mm:?}mm"
            );
        }
    }

    #[test]
    fn parse_geometry_size_requires_position_suffix() {
        assert_eq!(parse_geometry_size("2560x1440+0+0"), Some((2560, 1440)));
        assert_eq!(parse_geometry_size("1920x1080+100+200"), Some((1920, 1080)));
        assert_eq!(parse_geometry_size("2560x1440"), None);
        assert_eq!(parse_geometry_size("notageometry"), None);
    }
}
