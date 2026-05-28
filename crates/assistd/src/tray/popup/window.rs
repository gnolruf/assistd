//! eframe-backed window for the tray popup.
//!
//! Renders two labels — body and footer — from the latest
//! [`PopupState`] snapshot held in a `watch::Receiver`. The GUI
//! thread owns the entire eframe event loop; it sends the first-paint
//! Mapped event back to the tokio-side driver over the driver's
//! `UnboundedSender`, and observes the `visible` flag on the watch to
//! toggle the viewport between shown and hidden.
//!
//! The popup is intentionally a notification surface: it does not
//! dismiss itself when input focus moves to another window, and the
//! placement command (see `assistd_wm::criteria`) makes the window
//! sticky so it persists across workspace switches. Dismissal is
//! driven entirely by the driver's auto-hide timer.
//!
//! A small tokio task ([`wake_egui_on_state_change`]) runs on the
//! workspace runtime and calls `ctx.request_repaint()` whenever the
//! watch updates, so content changes don't have to wait for a stray
//! OS input event to wake the egui loop.

use eframe::egui::{self, Color32, FontFamily, FontId, RichText, ViewportBuilder, ViewportCommand};
use tokio::runtime::Handle;
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::watch;

use super::state::{PopupActivity, PopupState};
use super::visibility::DriverInput;

/// Run the eframe loop in the calling thread. Blocks until the
/// viewport is closed (we close it ourselves on a `Shutdown` command
/// from the driver via the `state.visible` channel; see
/// [`PopupApp::logic`]).
///
/// `runtime` is captured before the GUI thread starts so the
/// state-watch waker task can be spawned onto the workspace tokio
/// runtime from inside the eframe creator closure.
pub fn run(
    state_rx: watch::Receiver<PopupState>,
    event_tx: UnboundedSender<DriverInput>,
    app_id: &str,
    width: u32,
    height: u32,
    initial_position: Option<(i32, i32)>,
    runtime: Handle,
) -> Result<(), eframe::Error> {
    let mut viewport = ViewportBuilder::default()
        .with_app_id(app_id)
        // Explicitly set the X11 title — egui-winit 0.34 defaults it
        // to "egui window" and only propagates with_app_id to the
        // Wayland app_id, never to X11 WM_CLASS. Setting the title
        // gives the i3 backend a deterministic `[title="..."]`
        // criteria to match against.
        .with_title(app_id)
        .with_decorations(false)
        .with_resizable(false)
        .with_inner_size([width as f32, height as f32])
        .with_visible(false);
    if let Some((x, y)) = initial_position {
        // Pre-position the window so its first map lands at the
        // configured corner. winit propagates this to X11's window
        // attributes; i3 honours it for floating windows (which the
        // for_window rule makes the popup at map time). Without this,
        // i3's default-placement runs first and the popup briefly
        // flashes in the centre before place_floating shoves it into
        // place.
        viewport = viewport.with_position([x as f32, y as f32]);
    }
    let options = eframe::NativeOptions {
        viewport,
        // The popup runs on a dedicated OS thread because the tray's
        // tokio runtime owns the main thread (it has SIGINT/SIGTERM
        // handlers and the ksni service). winit's default main-thread
        // assertion is a portability hint, not a hard requirement — on
        // Linux X11 and Wayland a worker-thread event loop works once
        // we opt in. The error message that fires without these calls
        // points users at exactly these methods.
        event_loop_builder: Some(Box::new(|builder| {
            use winit::platform::wayland::EventLoopBuilderExtWayland;
            use winit::platform::x11::EventLoopBuilderExtX11;
            EventLoopBuilderExtX11::with_any_thread(builder, true);
            EventLoopBuilderExtWayland::with_any_thread(builder, true);
        })),
        ..Default::default()
    };
    let app_id_owned = app_id.to_string();
    let waker_rx = state_rx.clone();
    eframe::run_native(
        &app_id_owned,
        options,
        Box::new(move |cc| {
            // Wake egui on every watch change. Without this the GUI
            // thread sleeps between OS events, so a Show command or a
            // new LastDelta wouldn't be observed until the user
            // happened to move the mouse over the window.
            runtime.spawn(wake_egui_on_state_change(waker_rx, cc.egui_ctx.clone()));
            Ok(Box::new(PopupApp::new(state_rx, event_tx)))
        }),
    )
}

/// Forward `watch` change notifications to egui as repaint requests.
/// When the watch sender is dropped (popup is shutting down) the loop
/// exits and the final `ViewportCommand::Close` tells eframe's event
/// loop to return — without this signal `eframe::run_native` would
/// block the GUI thread forever and `gui_thread.join()` would hang on
/// shutdown.
async fn wake_egui_on_state_change(mut rx: watch::Receiver<PopupState>, ctx: egui::Context) {
    while rx.changed().await.is_ok() {
        ctx.request_repaint();
    }
    ctx.send_viewport_cmd(ViewportCommand::Close);
    // Egui only processes viewport commands on the next frame, so
    // poke the event loop once more so the close lands without
    // waiting for unrelated input.
    ctx.request_repaint();
}

/// eframe `App` for the popup. Visibility transitions and the
/// first-paint Mapped signal live in `logic`; `ui` only renders the
/// three labels. The popup deliberately ignores focus changes — see
/// the module-level comment for why.
struct PopupApp {
    state_rx: watch::Receiver<PopupState>,
    event_tx: UnboundedSender<DriverInput>,
    last_visible: bool,
    awaiting_first_paint: bool,
    /// True until `logic` runs for the first time. Used to force a
    /// `Visible(false)` viewport command on startup so the popup never
    /// flashes on screen — winit's `with_visible(false)` doesn't
    /// reliably suppress the initial map on X11, and once mapped the
    /// only thing that hides the window is an explicit
    /// `ViewportCommand::Visible(false)`.
    first_frame: bool,
    /// Cache of the most recently observed `PopupState`. `logic` runs
    /// before `ui` and refreshes it; `ui` reads it.
    current: PopupState,
}

impl PopupApp {
    fn new(state_rx: watch::Receiver<PopupState>, event_tx: UnboundedSender<DriverInput>) -> Self {
        Self {
            state_rx,
            event_tx,
            last_visible: false,
            awaiting_first_paint: false,
            first_frame: true,
            current: PopupState::default(),
        }
    }
}

impl eframe::App for PopupApp {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.current = self.state_rx.borrow_and_update().clone();

        let visibility_changed = self.current.visible != self.last_visible;
        if self.first_frame || visibility_changed {
            ctx.send_viewport_cmd(ViewportCommand::Visible(self.current.visible));
            if self.current.visible && visibility_changed {
                self.awaiting_first_paint = true;
            }
            self.last_visible = self.current.visible;
            self.first_frame = false;
        }

        if self.current.visible {
            if self.awaiting_first_paint {
                self.awaiting_first_paint = false;
                let _ = self.event_tx.send(DriverInput::Mapped);
            }

            ctx.request_repaint_after(std::time::Duration::from_millis(150));
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if !self.current.visible {
            return;
        }
        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                let header = activity_label(&self.current.activity);
                if let Some((label, color)) = header.as_ref() {
                    ui.label(
                        RichText::new(label)
                            .color(*color)
                            .font(FontId::new(11.0, FontFamily::Proportional)),
                    );
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if close_button(ui).clicked() {
                        let _ = self.event_tx.send(DriverInput::Dismiss);
                    }
                });
            });
            ui.add_space(2.0);
            if self.current.body.is_empty() {
                ui.label(
                    RichText::new("(no reply yet)")
                        .italics()
                        .color(Color32::GRAY)
                        .font(FontId::new(12.0, FontFamily::Proportional)),
                );
            } else {
                ui.label(
                    RichText::new(&self.current.body)
                        .font(FontId::new(12.0, FontFamily::Proportional)),
                );
            }
            if let Some(footer) = self.current.footer.as_ref() {
                ui.add_space(2.0);
                ui.horizontal_wrapped(|ui| {
                    ui.label(
                        RichText::new(&footer.name)
                            .monospace()
                            .color(Color32::LIGHT_BLUE)
                            .font(FontId::new(11.0, FontFamily::Monospace)),
                    );
                    ui.label(
                        RichText::new(&footer.args_summary)
                            .color(Color32::GRAY)
                            .font(FontId::new(11.0, FontFamily::Monospace)),
                    );
                });
            }
        });
    }
}

/// Render the top-right close button. Returns the egui [`Response`] so
/// the caller can detect clicks. Uses U+00D7 MULTIPLICATION SIGN (the
/// Latin-1 "×") rather than U+2715 / U+2717: the bundled default eframe
/// font (Ubuntu-Light) ships Latin-1 supplement but not Dingbats, so
/// the dingbat "✕" rendered as the tofu fallback box.
fn close_button(ui: &mut egui::Ui) -> egui::Response {
    let btn = egui::Button::new(
        RichText::new("\u{00D7}")
            .color(Color32::GRAY)
            .font(FontId::new(14.0, FontFamily::Proportional)),
    )
    .frame(false);
    ui.add(btn).on_hover_text("Close (interrupts the agent)")
}

/// Map [`PopupActivity`] to the one-line status header rendered above
/// the body. Returns `None` when the popup should be header-less (the
/// idle case — the body already conveys "last completed turn").
///
/// The leading marker is U+00B7 MIDDLE DOT (Latin-1 supplement) rather
/// than U+25CF BLACK CIRCLE: the default eframe font doesn't cover
/// Geometric Shapes, so the original "●" rendered as a tofu box.
fn activity_label(activity: &PopupActivity) -> Option<(String, Color32)> {
    match activity {
        PopupActivity::Idle => None,
        PopupActivity::Streaming => Some(("\u{00B7} replying…".into(), Color32::LIGHT_GREEN)),
        PopupActivity::Thinking => Some(("\u{00B7} thinking…".into(), Color32::YELLOW)),
        PopupActivity::RunningTool { name } => {
            Some((format!("\u{00B7} running {name}…"), Color32::LIGHT_BLUE))
        }
        PopupActivity::Listening => {
            Some(("\u{00B7} listening…".into(), Color32::from_rgb(255, 140, 0)))
        }
    }
}
