//! Floating activity popup spawned alongside the tray icon (feature
//! `tray-popup`). Subscribes to the same daemon event stream as the
//! tray, renders a small borderless window near the system-tray
//! corner, and delegates placement to the compositor through
//! `assistd-wm`.
//!
//! Spawned from [`crate::tray::run`] when the feature is enabled. Owns:
//!
//! - the watch channel the GUI thread reads from;
//! - a driver tokio task that ingests events, runs the visibility
//!   state machine, and pushes new snapshots;
//! - the WM backend + placement worker task that turns "popup just
//!   became visible" into a `floating enable, resize set, move
//!   position …` IPC sequence on i3 / sway;
//! - the dedicated OS thread that runs the eframe event loop.

use std::sync::Arc;
use std::thread::JoinHandle as ThreadJoinHandle;

use assistd_config::{Config, TrayPopupConfig};
use assistd_ipc::Event;
use tokio::sync::mpsc::{self, UnboundedSender};
use tokio::sync::watch;
use tokio::task::JoinHandle;

mod state;
mod visibility;
mod window;
mod wm_bridge;

pub use state::PopupState;
pub use visibility::DriverInput;

use visibility::{PlaceRequest, run as driver_run};
use wm_bridge::{
    WmBackendBundle, anchor_from_config, build_wm_backend, place_worker, popup_criteria,
};

/// Hook handed to the tray's subscribe loop. The subscribe loop calls
/// [`PopupSink::ingest`] after pushing each event into the tray icon's
/// `TrayItem`; it's a no-op when the popup feature is disabled at the
/// call site (we return `None` from [`spawn`] in that path).
#[derive(Clone)]
pub struct PopupSink {
    driver_tx: UnboundedSender<DriverInput>,
    wake_tool_call: bool,
    wake_delta: bool,
    wake_error: bool,
}

impl PopupSink {
    /// Forward a daemon event into the driver task and, if the event
    /// matches a configured wake rule, also send a `Show`. The event
    /// is boxed because `assistd_ipc::Event` is ~230 bytes and would
    /// dominate the [`DriverInput`] enum size otherwise.
    pub fn ingest(&self, ev: &Event) {
        if self.matches_wake_rule(ev) {
            let _ = self.driver_tx.send(DriverInput::Show);
        }
        let _ = self
            .driver_tx
            .send(DriverInput::Event(Box::new(ev.clone())));
    }

    /// Tell the popup the daemon socket dropped.
    pub fn set_disconnected(&self) {
        let _ = self.driver_tx.send(DriverInput::Disconnected);
    }

    /// Sender exposed so the tray-icon menu code can hand it to
    /// `TrayItem` for the left-click handler.
    pub fn show_sender(&self) -> UnboundedSender<DriverInput> {
        self.driver_tx.clone()
    }

    fn matches_wake_rule(&self, ev: &Event) -> bool {
        match ev {
            Event::ToolCall { .. } => self.wake_tool_call,
            Event::Delta { .. } | Event::LastDelta { .. } | Event::ReasoningDelta { .. } => {
                self.wake_delta
            }
            Event::Error { .. } => self.wake_error,
            _ => false,
        }
    }
}

/// Live popup. Holds every spawned task/thread; [`PopupHandle::shutdown`]
/// drains them in the right order.
pub struct PopupHandle {
    pub sink: PopupSink,
    driver_task: JoinHandle<()>,
    place_task: JoinHandle<()>,
    gui_thread: Option<ThreadJoinHandle<()>>,
    wm_handle: Option<assistd_wm::WmHandle>,
    wm_shutdown: watch::Sender<bool>,
}

impl PopupHandle {
    /// Shut everything down cooperatively: tell the driver to quit
    /// (which drops the watch sender, which wakes the egui-waker
    /// task, which sends `ViewportCommand::Close` to the GUI thread).
    /// Then drain the placement worker, join the GUI thread, and
    /// stop the WM backend.
    pub async fn shutdown(mut self) {
        let _ = self.sink.driver_tx.send(DriverInput::Shutdown);
        let _ = self.driver_task.await;
        // Dropping the senders inside the sink lets the place worker
        // observe EOF and exit.
        drop(self.sink);
        let _ = self.place_task.await;
        if let Some(t) = self.gui_thread.take() {
            // std::thread::join blocks; offload to spawn_blocking so
            // the tokio worker stays available for the egui-waker
            // task that's busy sending Close to the GUI loop.
            let join = tokio::task::spawn_blocking(move || t.join()).await;
            match join {
                Ok(Ok(())) => {}
                Ok(Err(panic)) => {
                    tracing::warn!(target: "tray", "popup: GUI thread panicked: {panic:?}");
                }
                Err(e) => {
                    tracing::warn!(target: "tray", "popup: GUI join task failed: {e}");
                }
            }
        }
        let _ = self.wm_shutdown.send(true);
        if let Some(h) = self.wm_handle {
            h.shutdown().await;
        }
    }
}

/// Spawn the popup subsystem. Returns `Ok(None)` when the popup is
/// disabled by config; otherwise builds the WM backend, spawns the
/// driver task, the placement worker, and the GUI thread, and returns
/// a [`PopupHandle`] the tray can shut down at exit.
pub async fn spawn(cfg: &Config) -> anyhow::Result<Option<PopupHandle>> {
    if !cfg.tray.popup.enabled {
        tracing::info!(target: "tray", "popup: disabled by config");
        return Ok(None);
    }
    let popup_cfg: TrayPopupConfig = cfg.tray.popup.clone();

    let (wm_shutdown_tx, wm_shutdown_rx) = watch::channel(false);
    let WmBackendBundle { manager, handle } = build_wm_backend(cfg, wm_shutdown_rx).await;
    let manager: Arc<dyn assistd_wm::WindowManager> = manager;

    let (state_tx, state_rx) = watch::channel(PopupState::default());
    let (driver_tx, driver_rx) = mpsc::unbounded_channel::<DriverInput>();
    let (place_tx, place_rx) = mpsc::unbounded_channel::<PlaceRequest>();

    let anchor = anchor_from_config(&popup_cfg);

    // Compute a best-effort initial position so eframe creates the
    // window at the configured corner from the very first map — this
    // is what kills the brief "popup flashes in the centre" on the
    // first wake. Ask the WM backend for the focused output's scale
    // and scale the configured size by it (a configured 360×120 popup
    // becomes 420×140 physical at scale 1.17). On sway this comes
    // straight from the compositor's output IPC reply; on i3 the
    // backend cascades through `WINIT_X11_SCALE_FACTOR`, `Xft.dpi` in
    // xrdb, and RandR-derived DPI to match what winit will apply
    // to the popup window. The actual rect snap via `place_floating`
    // then locks in the precise pixel position once the window is
    // mapped, so any residual error here is invisible.
    let scale = match manager.focused_output_scale().await {
        Ok(s) => {
            tracing::info!(target: "tray", "popup: focused-output scale = {s:.4}");
            s as f32
        }
        Err(e) => {
            tracing::warn!(
                target: "tray",
                "popup: could not query focused-output scale ({e}); assuming 1.0"
            );
            1.0
        }
    };
    let scaled_anchor = if (scale - 1.0).abs() > f32::EPSILON {
        scale_anchor_size(anchor, scale)
    } else {
        anchor
    };
    let initial_position = match manager.focused_workspace_rect().await {
        Ok(ws) => {
            let (x, y) = assistd_wm::criteria::compute_target_position(scaled_anchor, ws);
            tracing::info!(
                target: "tray",
                "popup: pre-positioning at ({x}, {y}) on workspace {}x{}",
                ws.width, ws.height
            );
            Some((x, y))
        }
        Err(e) => {
            tracing::warn!(
                target: "tray",
                "popup: could not query workspace rect ({e}); window will appear at compositor default until first place_floating"
            );
            None
        }
    };

    let place_task = tokio::spawn(place_worker(
        place_rx,
        manager.clone(),
        popup_criteria(),
        anchor,
    ));

    let driver_cfg = popup_cfg.clone();
    let driver_task = tokio::spawn(driver_run(state_tx, driver_rx, place_tx, driver_cfg));

    let app_id = assistd_config::defaults::DEFAULT_TRAY_POPUP_APP_ID.to_string();
    let (gui_event_tx, gui_event_rx) = std::sync::mpsc::channel::<DriverInput>();

    // Bridge std mpsc → driver mpsc on a dedicated tokio task. We can't
    // call `driver_tx.send` from inside the GUI thread directly without
    // borrowing the tokio runtime, so a small forwarder loop is the
    // simplest correct option.
    let bridge_tx = driver_tx.clone();
    let bridge_task = tokio::task::spawn_blocking(move || {
        while let Ok(msg) = gui_event_rx.recv() {
            if bridge_tx.send(msg).is_err() {
                break;
            }
        }
    });
    // We don't store bridge_task; it terminates when the GUI thread
    // closes its sender (on shutdown) which happens when the OS thread
    // joins.
    drop(bridge_task);

    let width = popup_cfg.width;
    let height = popup_cfg.height;
    let gui_state_rx = state_rx.clone();
    // Capture the workspace runtime handle before crossing the
    // thread boundary; window::run uses it to spawn the egui waker.
    let runtime = tokio::runtime::Handle::current();
    let gui_thread = std::thread::Builder::new()
        .name("assistd-popup-gui".into())
        .spawn(move || {
            if let Err(e) = window::run(
                gui_state_rx,
                gui_event_tx,
                &app_id,
                width,
                height,
                initial_position,
                runtime,
            ) {
                tracing::warn!(target: "tray", "popup: eframe loop exited with error: {e}");
            }
        })?;

    let sink = PopupSink {
        driver_tx: driver_tx.clone(),
        wake_tool_call: popup_cfg.wake_on.tool_call,
        wake_delta: popup_cfg.wake_on.delta,
        wake_error: popup_cfg.wake_on.error,
    };

    Ok(Some(PopupHandle {
        sink,
        driver_task,
        place_task,
        gui_thread: Some(gui_thread),
        wm_handle: handle,
        wm_shutdown: wm_shutdown_tx,
    }))
}

/// Scale an anchor's `width` / `height` by `scale`, keeping the
/// corner and offsets untouched. Used for HiDPI-aware pre-positioning.
fn scale_anchor_size(
    anchor: assistd_wm::PlacementAnchor,
    scale: f32,
) -> assistd_wm::PlacementAnchor {
    assistd_wm::PlacementAnchor {
        width: (anchor.width as f32 * scale).round() as u32,
        height: (anchor.height as f32 * scale).round() as u32,
        ..anchor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn scale_anchor_size_rounds_to_nearest_pixel() {
        use assistd_wm::{AnchorCorner, PlacementAnchor};
        let a = PlacementAnchor {
            corner: AnchorCorner::BottomRight,
            offset_x: -10,
            offset_y: -30,
            width: 360,
            height: 120,
        };
        let scaled = scale_anchor_size(a, 1.1666666);
        assert_eq!(scaled.width, 420);
        assert_eq!(scaled.height, 140);
        // Offsets and corner are not touched.
        assert_eq!(scaled.offset_x, -10);
        assert_eq!(scaled.offset_y, -30);
        assert_eq!(scaled.corner, AnchorCorner::BottomRight);
    }

    #[test]
    fn scale_anchor_size_identity_at_scale_one() {
        use assistd_wm::{AnchorCorner, PlacementAnchor};
        let a = PlacementAnchor {
            corner: AnchorCorner::TopLeft,
            offset_x: 5,
            offset_y: 5,
            width: 200,
            height: 100,
        };
        let scaled = scale_anchor_size(a, 1.0);
        assert_eq!(scaled.width, 200);
        assert_eq!(scaled.height, 100);
    }

    fn sink(
        wake_tool_call: bool,
        wake_delta: bool,
        wake_error: bool,
    ) -> (PopupSink, mpsc::UnboundedReceiver<DriverInput>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (
            PopupSink {
                driver_tx: tx,
                wake_tool_call,
                wake_delta,
                wake_error,
            },
            rx,
        )
    }

    #[test]
    fn sink_wake_rules_respect_config() {
        let (s, _rx) = sink(true, false, true);
        assert!(s.matches_wake_rule(&Event::ToolCall {
            id: "a".into(),
            name: "x".into(),
            args: json!({}),
        }));
        assert!(!s.matches_wake_rule(&Event::Delta {
            id: "a".into(),
            text: "x".into(),
        }));
        assert!(!s.matches_wake_rule(&Event::LastDelta {
            id: "a".into(),
            text: "x".into(),
        }));
        assert!(s.matches_wake_rule(&Event::Error {
            id: "a".into(),
            message: "x".into(),
        }));
        assert!(!s.matches_wake_rule(&Event::Done { id: "a".into() }));
    }

    #[test]
    fn sink_ingest_sends_show_and_event_when_wake_matches() {
        let (s, mut rx) = sink(true, true, true);
        s.ingest(&Event::ToolCall {
            id: "a".into(),
            name: "x".into(),
            args: json!({}),
        });
        let first = rx.try_recv().expect("show queued");
        assert!(matches!(first, DriverInput::Show));
        let second = rx.try_recv().expect("event queued");
        assert!(matches!(second, DriverInput::Event(_)));
    }

    #[test]
    fn sink_ingest_only_sends_event_when_wake_skipped() {
        let (s, mut rx) = sink(false, false, false);
        s.ingest(&Event::Done { id: "a".into() });
        let only = rx.try_recv().expect("event queued");
        assert!(matches!(only, DriverInput::Event(_)));
        assert!(rx.try_recv().is_err(), "no Show should have been queued");
    }
}
