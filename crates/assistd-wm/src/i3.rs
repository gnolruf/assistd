//! i3 backend for [`crate::WindowManager`], over `tokio-i3ipc`. One
//! socket serves commands; a second feeds the event stream that keeps
//! the focus snapshot current.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures_util::StreamExt;
use tokio::sync::{Mutex, RwLock, broadcast, watch};
use tokio::task::JoinHandle;
use tokio_i3ipc::{
    I3,
    event::{Event, Subscribe, WindowChange, WorkspaceChange},
    reply,
};

use crate::criteria::{
    format_focus, format_layout, format_move_to_workspace, format_place_floating_pixels,
    format_resize_width,
};
use crate::snapshot::{
    self, Snapshot, WindowChangeKind, apply_window_event, apply_workspace_focus,
};
use crate::{
    FocusedWindowContext, Layout, PlacementAnchor, PlacementCriteria, Rect, ResizeDir,
    WM_IPC_TIMEOUT, Window, WindowEvent, WindowId, WindowManager, WmError, WmResult, WorkspaceId,
    WorkspaceInfo,
};

/// Window events buffered per subscriber. A lagged subscriber falls
/// back to polling, so a small buffer is fine.
const WINDOW_EVENTS_CAPACITY: usize = 32;

/// How long `find_window_rect_by_criteria` waits for a matching
/// `WindowEvent::Opened` after its first tree poll misses.
const WINDOW_EVENT_WAIT: Duration = Duration::from_millis(500);

/// [`WindowManager`] over a single i3 IPC command socket.
pub struct I3Backend {
    cmd: Arc<Mutex<Option<I3>>>,
    snapshot: Arc<RwLock<Snapshot>>,
    reconnect: Arc<tokio::sync::Notify>,
    window_events: broadcast::Sender<WindowEvent>,
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
        let (mut cmd, events_conn) = connect_pair().await?;
        let initial = match seed_snapshot(&mut cmd).await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("i3 seed_snapshot failed: {e:#}; starting with empty focus state");
                Snapshot::default()
            }
        };
        let snapshot = Arc::new(RwLock::new(initial));
        let reconnect = Arc::new(tokio::sync::Notify::new());
        let (window_events, _) = broadcast::channel(WINDOW_EVENTS_CAPACITY);
        let backend = Arc::new(Self {
            cmd: Arc::new(Mutex::new(Some(cmd))),
            snapshot: snapshot.clone(),
            reconnect: reconnect.clone(),
            window_events,
        });

        let supervisor_task = tokio::spawn(supervisor_loop(
            backend.clone(),
            events_conn,
            snapshot,
            reconnect,
            shutdown,
        ));

        Ok(I3Handle {
            backend,
            supervisor_task,
        })
    }

    /// Run one IPC call on the command socket under [`WM_IPC_TIMEOUT`].
    /// A timeout or transport error drops the connection and wakes the
    /// supervisor to reconnect.
    async fn with_conn<T>(
        &self,
        ctx: &'static str,
        op: impl AsyncFnOnce(&mut I3) -> std::io::Result<T>,
    ) -> WmResult<T> {
        let mut guard = self.cmd.lock().await;
        let conn = guard.as_mut().ok_or(WmError::Disconnected)?;
        let outcome = tokio::time::timeout(WM_IPC_TIMEOUT, op(conn)).await;
        let err = match outcome {
            Ok(Ok(value)) => return Ok(value),
            Ok(Err(e)) => WmError::ipc(ctx, e),
            Err(_) => WmError::Timeout(WM_IPC_TIMEOUT),
        };
        *guard = None;
        self.reconnect.notify_one();
        Err(err)
    }

    async fn run_command(&self, payload: &str) -> WmResult<()> {
        let results = self
            .with_conn("i3 RUN_COMMAND", async |conn| {
                conn.run_command(payload).await
            })
            .await?;
        for r in results {
            if !r.success {
                return Err(WmError::Rejected(format!(
                    "{payload}: {}",
                    r.error.unwrap_or_else(|| "unknown error".into())
                )));
            }
        }
        Ok(())
    }

    async fn workspaces(&self, ctx: &'static str) -> WmResult<Vec<reply::Workspace>> {
        self.with_conn(ctx, async |conn| conn.get_workspaces().await)
            .await
    }

    async fn tree(&self, ctx: &'static str) -> WmResult<reply::Node> {
        self.with_conn(ctx, async |conn| conn.get_tree().await)
            .await
    }
}

#[async_trait]
impl WindowManager for I3Backend {
    async fn focus(&self, window: &WindowId) -> WmResult<()> {
        self.run_command(&format_focus(window)).await
    }

    async fn move_to_workspace(&self, window: &WindowId, workspace: &WorkspaceId) -> WmResult<()> {
        self.run_command(&format_move_to_workspace(window, workspace))
            .await
    }

    async fn focused_window(&self) -> WmResult<Option<WindowId>> {
        Ok(snapshot::read_focused_id(&self.snapshot).await)
    }

    async fn focused_context(&self) -> WmResult<Option<FocusedWindowContext>> {
        Ok(snapshot::read_focused_context(&self.snapshot).await)
    }

    async fn list_windows(&self) -> WmResult<Vec<Window>> {
        let tree = self.tree("i3 GET_TREE").await?;
        let mut out = Vec::new();
        collect_windows(&tree, None, &mut out);
        Ok(out)
    }

    async fn list_workspaces(&self) -> WmResult<Vec<WorkspaceInfo>> {
        Ok(self
            .workspaces("i3 GET_WORKSPACES")
            .await?
            .into_iter()
            .map(|w| WorkspaceInfo {
                num: w.num,
                name: w.name,
                focused: w.focused,
                output: w.output,
            })
            .collect())
    }

    async fn resize_width(
        &self,
        window: &WindowId,
        direction: ResizeDir,
        pixels: u32,
    ) -> WmResult<()> {
        self.run_command(&format_resize_width(window, direction, pixels))
            .await
    }

    async fn set_layout(&self, layout: Layout) -> WmResult<()> {
        self.run_command(&format_layout(layout)).await
    }

    async fn focused_workspace_rect(&self) -> WmResult<Rect> {
        self.workspaces("i3 GET_WORKSPACES (focused rect)")
            .await?
            .into_iter()
            .find(|w| w.focused)
            .map(|w| Rect {
                x: w.rect.x as i32,
                y: w.rect.y as i32,
                width: w.rect.width.max(0) as u32,
                height: w.rect.height.max(0) as u32,
            })
            .ok_or_else(|| WmError::Rejected("no focused workspace".into()))
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
        let translated = translate_criteria_for_i3(criteria);
        let workspace = self.focused_workspace_rect().await?;
        // DPI scaling can map a 360-logical-px request to 420 physical
        // px, so place using the window's actual rect.
        let effective = match self.find_window_rect_by_criteria(&translated).await {
            Ok(actual) => {
                tracing::info!(
                    target: "tray",
                    "popup: actual window rect = {}x{} (configured {}x{}); placing accordingly",
                    actual.width, actual.height, anchor.width, anchor.height
                );
                PlacementAnchor {
                    width: actual.width,
                    height: actual.height,
                    ..anchor
                }
            }
            Err(e) => {
                tracing::warn!(
                    target: "tray",
                    "popup: could not query window rect ({e}); falling back to configured size"
                );
                anchor
            }
        };
        self.run_command(&format_place_floating_pixels(
            &translated,
            effective,
            workspace,
        ))
        .await
    }
}

impl I3Backend {
    /// Current rect of the window matching `criteria`. Subscribes to
    /// window events before the first tree poll so a `window::new`
    /// that lands between the poll and the wait is not missed.
    async fn find_window_rect_by_criteria(&self, criteria: &PlacementCriteria) -> WmResult<Rect> {
        let mut events = self.window_events.subscribe();

        if let Ok(rect) = self.find_window_rect_once(criteria).await {
            return Ok(rect);
        }

        let waited = tokio::time::timeout(WINDOW_EVENT_WAIT, async {
            loop {
                match events.recv().await {
                    Ok(ev) => {
                        if ev.matches_opened(criteria).is_some() {
                            return true;
                        }
                    }
                    Err(broadcast::error::RecvError::Closed) => return false,
                    Err(broadcast::error::RecvError::Lagged(_)) => return false,
                }
            }
        })
        .await
        .unwrap_or(false);

        let result = self.find_window_rect_once(criteria).await;
        if waited && result.is_ok() {
            tracing::debug!(
                target: "tray",
                "popup: window appeared via i3 window::new event"
            );
        }
        result
    }

    async fn find_window_rect_once(&self, criteria: &PlacementCriteria) -> WmResult<Rect> {
        let tree = self.tree("i3 GET_TREE (window rect)").await?;
        find_node_rect(&tree, criteria)
            .ok_or_else(|| WmError::Rejected(format!("no window matches {criteria:?}")))
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
            .workspaces("i3 GET_WORKSPACES (focused output name)")
            .await?
            .into_iter()
            .find(|w| w.focused)
            .map(|w| w.output))
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

fn find_node_rect(node: &reply::Node, criteria: &PlacementCriteria) -> Option<Rect> {
    if node.window.is_some() && node_matches(node, criteria) {
        return Some(Rect {
            x: node.rect.x as i32,
            y: node.rect.y as i32,
            width: node.rect.width.max(0) as u32,
            height: node.rect.height.max(0) as u32,
        });
    }
    for child in node.nodes.iter().chain(node.floating_nodes.iter()) {
        if let Some(r) = find_node_rect(child, criteria) {
            return Some(r);
        }
    }
    None
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

    for child in node.nodes.iter().chain(node.floating_nodes.iter()) {
        collect_windows(child, next_ws, out);
    }
}

async fn seed_snapshot(cmd: &mut I3) -> WmResult<Snapshot> {
    let tree = cmd
        .get_tree()
        .await
        .map_err(|e| WmError::ipc("i3 GET_TREE", e))?;
    let focused = walk_focused(&tree);
    let focused_id = focused.and_then(|n| WindowId::new(n.id as u64));
    let focused_class = focused
        .and_then(|n| n.window_properties.as_ref())
        .and_then(|p| p.class.clone());
    let focused_title = focused.and_then(|n| n.name.clone());

    let active_workspace = match cmd.get_workspaces().await {
        Ok(ws) => ws.into_iter().find(|w| w.focused).map(|w| w.name),
        Err(e) => {
            tracing::warn!("i3 GET_WORKSPACES on seed failed: {e:#}");
            None
        }
    };
    Ok(Snapshot {
        focused_id,
        focused_class,
        focused_title,
        active_workspace,
    })
}

async fn connect_pair() -> WmResult<(I3, I3)> {
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
    Ok((cmd, events_conn))
}

/// Drive one events connection. Returns `true` when the caller should
/// reconnect, `false` on shutdown.
async fn drive_events(
    events_conn: I3,
    snapshot: Arc<RwLock<Snapshot>>,
    reconnect: Arc<tokio::sync::Notify>,
    window_events: broadcast::Sender<WindowEvent>,
    shutdown: &mut watch::Receiver<bool>,
) -> bool {
    let mut stream = events_conn.listen();
    loop {
        tokio::select! {
            _ = shutdown.changed() => {
                if *shutdown.borrow() { return false; }
            }
            _ = reconnect.notified() => {
                return true;
            }
            evt = stream.next() => {
                match evt {
                    Some(Ok(Event::Window(w))) => {
                        handle_window_event(&w, &snapshot).await;
                        if let Some(ev) = window_event_from_i3(&w) {
                            let _ = window_events.send(ev);
                        }
                    }
                    Some(Ok(Event::Workspace(data))) => {
                        if matches!(data.change, WorkspaceChange::Focus) {
                            let name = data.current.as_ref().and_then(|n| n.name.clone());
                            apply_workspace_focus(&snapshot, name).await;
                        }
                    }
                    Some(Ok(_)) => {}
                    Some(Err(e)) => {
                        tracing::warn!("i3 event stream error: {e}");
                        return true;
                    }
                    None => return true,
                }
            }
        }
    }
}

fn window_event_from_i3(w: &tokio_i3ipc::event::WindowData) -> Option<WindowEvent> {
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

async fn supervisor_loop(
    backend: Arc<I3Backend>,
    initial_events: I3,
    snapshot: Arc<RwLock<Snapshot>>,
    reconnect: Arc<tokio::sync::Notify>,
    mut shutdown: watch::Receiver<bool>,
) {
    if !drive_events(
        initial_events,
        snapshot.clone(),
        reconnect.clone(),
        backend.window_events.clone(),
        &mut shutdown,
    )
    .await
    {
        tracing::info!("i3 supervisor exited (shutdown during initial events stream)");
        return;
    }

    let mut attempt: u32 = 0;
    loop {
        *backend.cmd.lock().await = None;
        tracing::warn!("i3 disconnected; reconnecting (attempt {})", attempt + 1);

        let delay = crate::backoff::backoff_delay(attempt);
        tokio::select! {
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    tracing::info!("i3 supervisor exited (shutdown during backoff)");
                    return;
                }
            }
            _ = tokio::time::sleep(delay) => {}
        }

        match connect_pair().await {
            Ok((mut cmd, events_conn)) => {
                if let Ok(s) = seed_snapshot(&mut cmd).await {
                    *snapshot.write().await = s;
                }
                *backend.cmd.lock().await = Some(cmd);
                attempt = 0;
                tracing::info!("i3 backend reconnected");
                if !drive_events(
                    events_conn,
                    snapshot.clone(),
                    reconnect.clone(),
                    backend.window_events.clone(),
                    &mut shutdown,
                )
                .await
                {
                    tracing::info!("i3 supervisor exited (shutdown during events stream)");
                    return;
                }
            }
            Err(e) => {
                tracing::warn!("i3 reconnect failed: {e}; will retry after backoff");
                attempt = attempt.saturating_add(1);
            }
        }
    }
}

async fn handle_window_event(w: &tokio_i3ipc::event::WindowData, snap: &Arc<RwLock<Snapshot>>) {
    let kind = match w.change {
        WindowChange::Focus => WindowChangeKind::Focus,
        WindowChange::Title => WindowChangeKind::Title,
        WindowChange::Close => WindowChangeKind::Close,
        _ => return,
    };
    let id = WindowId::new(w.container.id as u64);
    let class = w
        .container
        .window_properties
        .as_ref()
        .and_then(|p| p.class.clone());
    let title = w.container.name.clone();
    apply_window_event(snap, kind, id, class, title).await;
}

fn walk_focused(node: &reply::Node) -> Option<&reply::Node> {
    if node.focused {
        return Some(node);
    }
    for child in node.nodes.iter().chain(node.floating_nodes.iter()) {
        if let Some(n) = walk_focused(child) {
            return Some(n);
        }
    }
    None
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
