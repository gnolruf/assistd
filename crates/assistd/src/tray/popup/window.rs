//! eframe-backed window for the tray popup.
//!
//! Renders three labels — title, body, footer — from the latest
//! [`PopupState`] snapshot held in a `watch::Receiver`. The GUI
//! thread owns the entire eframe event loop; it sends focus-lost and
//! first-paint events back to the tokio-side driver via a `std`
//! mpsc, and observes the `visible` flag on the watch to toggle the
//! viewport between shown and hidden.

use std::sync::mpsc::Sender;

use eframe::egui::{self, Color32, FontFamily, FontId, RichText, ViewportBuilder, ViewportCommand};
use tokio::sync::watch;

use super::state::PopupState;
use super::visibility::DriverInput;

/// Run the eframe loop in the calling thread. Blocks until the
/// viewport is closed (we close it ourselves on a `Shutdown` command
/// from the driver via the `state.visible` channel; see
/// [`PopupApp::logic`]).
///
/// The app_id constant is wired through both this builder and the
/// `place_floating` criteria so the compositor's
/// `[app_id="dev.assistd.popup"]` rule matches our window.
pub fn run(
    state_rx: watch::Receiver<PopupState>,
    event_tx: Sender<DriverInput>,
    app_id: &str,
    width: u32,
    height: u32,
) -> Result<(), eframe::Error> {
    let viewport = ViewportBuilder::default()
        .with_app_id(app_id)
        .with_decorations(false)
        .with_resizable(false)
        .with_inner_size([width as f32, height as f32])
        .with_visible(false);
    let options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };
    let app_id_owned = app_id.to_string();
    eframe::run_native(
        &app_id_owned,
        options,
        Box::new(move |_cc| Ok(Box::new(PopupApp::new(state_rx, event_tx)))),
    )
}

/// eframe `App` for the popup. Visibility transitions, focus tracking,
/// and the first-paint Mapped signal live in `logic`; `ui` only
/// renders the three labels.
struct PopupApp {
    state_rx: watch::Receiver<PopupState>,
    event_tx: Sender<DriverInput>,
    last_visible: bool,
    last_focused: bool,
    awaiting_first_paint: bool,
    /// Cache of the most recently observed `PopupState`. `logic` runs
    /// before `ui` and refreshes it; `ui` reads it.
    current: PopupState,
}

impl PopupApp {
    fn new(state_rx: watch::Receiver<PopupState>, event_tx: Sender<DriverInput>) -> Self {
        Self {
            state_rx,
            event_tx,
            last_visible: false,
            last_focused: true,
            awaiting_first_paint: false,
            current: PopupState::default(),
        }
    }
}

impl eframe::App for PopupApp {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.current = self.state_rx.borrow_and_update().clone();

        if self.current.visible != self.last_visible {
            ctx.send_viewport_cmd(ViewportCommand::Visible(self.current.visible));
            if self.current.visible {
                ctx.send_viewport_cmd(ViewportCommand::Focus);
                self.awaiting_first_paint = true;
            }
            self.last_visible = self.current.visible;
        }

        if self.current.visible {
            let focused = ctx.input(|i| i.viewport().focused.unwrap_or(true));
            if self.last_focused && !focused {
                let _ = self.event_tx.send(DriverInput::FocusLost);
            }
            self.last_focused = focused;

            if self.awaiting_first_paint {
                self.awaiting_first_paint = false;
                let _ = self.event_tx.send(DriverInput::Mapped);
            }

            ctx.request_repaint_after(std::time::Duration::from_millis(150));
        } else {
            // Reset edge detector so the next show doesn't immediately
            // re-trigger a hide before egui has refreshed focus.
            self.last_focused = true;
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if !self.current.visible {
            return;
        }
        ui.vertical(|ui| {
            ui.label(
                RichText::new(&self.current.title)
                    .strong()
                    .font(FontId::new(13.0, FontFamily::Proportional)),
            );
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
