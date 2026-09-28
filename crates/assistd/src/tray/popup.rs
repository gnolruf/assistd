//! Floating activity popup spawned alongside the tray icon.

use std::thread::JoinHandle as ThreadJoinHandle;

use assistd_config::Config;
use assistd_config::defaults::DEFAULT_TRAY_POPUP_APP_ID;
use assistd_ipc::{Event, IpcClient};
use assistd_wm::criteria::compute_target_position;
use assistd_wm::{PlacementAnchor, WindowManager, WmHandle};
use tokio::runtime::Handle as RuntimeHandle;
use tokio::sync::mpsc::{self, UnboundedSender};
use tokio::sync::watch;
use tokio::task::JoinHandle;

use crate::wm_backend::{WmBackend, start_backend};
use visibility::{PlaceRequest, drive_visibility};
use wm_bridge::{anchor_from_config, place_worker, popup_criteria};

mod state;
mod visibility;
mod window;
mod wm_bridge;

pub use state::PopupState;
pub use visibility::DriverInput;

/// Cloneable handle that feeds daemon events to the popup driver.
#[derive(Clone)]
pub struct PopupSink {
    driver_tx: UnboundedSender<DriverInput>,
    wake_tool_call: bool,
    wake_delta: bool,
    wake_error: bool,
}

impl PopupSink {
    /// Forward an event into the driver, plus a `Show` when the event
    /// matches a configured wake rule.
    pub fn ingest(&self, ev: &Event) {
        if self.matches_wake_rule(ev) {
            let _ = self.driver_tx.send(DriverInput::Show);
        }
        let _ = self
            .driver_tx
            .send(DriverInput::Event(Box::new(ev.clone())));
    }

    pub fn set_disconnected(&self) {
        let _ = self.driver_tx.send(DriverInput::Disconnected);
    }

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

/// Owns every task and thread the popup spawned.
pub struct PopupHandle {
    pub sink: PopupSink,
    driver_task: JoinHandle<()>,
    place_task: JoinHandle<()>,
    gui_thread: Option<ThreadJoinHandle<()>>,
    wm_handle: Option<WmHandle>,
    wm_shutdown: watch::Sender<bool>,
}

impl PopupHandle {
    pub async fn shutdown(mut self) {
        let _ = self.sink.driver_tx.send(DriverInput::Shutdown);
        let _ = self.driver_task.await;
        drop(self.sink);
        let _ = self.place_task.await;
        if let Some(t) = self.gui_thread.take() {
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

/// Spawn the popup subsystem; `Ok(None)` when disabled by config.
pub async fn spawn_popup(cfg: &Config, ipc: IpcClient) -> anyhow::Result<Option<PopupHandle>> {
    if !cfg.tray.popup.enabled {
        tracing::info!(target: "tray", "popup: disabled by config");
        return Ok(None);
    }
    let popup_cfg = &cfg.tray.popup;

    let (wm_shutdown_tx, wm_shutdown_rx) = watch::channel(false);
    let WmBackend { manager, handle } = start_backend(cfg, wm_shutdown_rx).await;

    let (state_tx, state_rx) = watch::channel(PopupState::default());
    let (driver_tx, driver_rx) = mpsc::unbounded_channel::<DriverInput>();
    let (place_tx, place_rx) = mpsc::unbounded_channel::<PlaceRequest>();

    let anchor = anchor_from_config(popup_cfg);
    let initial_position = initial_window_position(manager.as_ref(), anchor).await;

    let place_task = tokio::spawn(place_worker(place_rx, manager, popup_criteria(), anchor));
    let driver_task = tokio::spawn(drive_visibility(
        state_tx,
        driver_rx,
        place_tx,
        popup_cfg.clone(),
        ipc,
    ));
    let gui_thread = spawn_gui_thread(
        state_rx,
        driver_tx.clone(),
        popup_cfg.width,
        popup_cfg.height,
        initial_position,
    )?;

    let sink = PopupSink {
        driver_tx,
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

/// Where to map the window before the first `place_floating`, so it never
/// flashes at the compositor's default position.
async fn initial_window_position(
    manager: &dyn WindowManager,
    anchor: PlacementAnchor,
) -> Option<(i32, i32)> {
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
    match manager.focused_workspace_rect().await {
        Ok(ws) => {
            let (x, y) = compute_target_position(scaled_anchor, ws);
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
    }
}

fn spawn_gui_thread(
    state_rx: watch::Receiver<PopupState>,
    event_tx: UnboundedSender<DriverInput>,
    width: u32,
    height: u32,
    initial_position: Option<(i32, i32)>,
) -> std::io::Result<ThreadJoinHandle<()>> {
    let runtime = RuntimeHandle::current();
    std::thread::Builder::new()
        .name("assistd-popup-gui".into())
        .spawn(move || {
            if let Err(e) = window::run_gui_loop(
                state_rx,
                event_tx,
                DEFAULT_TRAY_POPUP_APP_ID,
                width,
                height,
                initial_position,
                runtime,
            ) {
                tracing::warn!(target: "tray", "popup: eframe loop exited with error: {e}");
            }
        })
}

fn scale_anchor_size(anchor: PlacementAnchor, scale: f32) -> PlacementAnchor {
    PlacementAnchor {
        width: (anchor.width as f32 * scale).round() as u32,
        height: (anchor.height as f32 * scale).round() as u32,
        ..anchor
    }
}

#[cfg(test)]
mod tests {
    use assistd_wm::AnchorCorner;
    use serde_json::json;

    use super::*;

    #[test]
    fn scale_anchor_size_rounds_to_nearest_pixel() {
        let a = PlacementAnchor {
            corner: AnchorCorner::BottomRight,
            offset_x: -10,
            offset_y: -30,
            width: 360,
            height: 120,
        };
        assert_eq!(
            scale_anchor_size(a, 1.1666666),
            PlacementAnchor {
                width: 420,
                height: 140,
                ..a
            }
        );
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
        assert!(matches!(first, DriverInput::Show), "{first:?}");
        let second = rx.try_recv().expect("event queued");
        assert!(
            matches!(&second, DriverInput::Event(ev) if matches!(**ev, Event::ToolCall { .. })),
            "{second:?}"
        );
    }

    #[test]
    fn sink_ingest_only_sends_event_when_wake_skipped() {
        let (s, mut rx) = sink(false, false, false);
        s.ingest(&Event::Done { id: "a".into() });
        let only = rx.try_recv().expect("event queued");
        assert!(
            matches!(&only, DriverInput::Event(ev) if matches!(**ev, Event::Done { .. })),
            "{only:?}"
        );
        assert!(rx.try_recv().is_err(), "no Show should have been queued");
    }
}
