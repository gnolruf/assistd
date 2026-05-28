//! eframe-backed window for the tray popup.

use eframe::egui::{self, Color32, FontFamily, FontId, RichText, ViewportBuilder, ViewportCommand};
use tokio::runtime::Handle;
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::watch;

use super::state::{PopupActivity, PopupState};
use super::visibility::DriverInput;

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
        // egui-winit only propagates app_id to Wayland; set the title
        // explicitly so X11 WM_CLASS gives the i3 backend a stable
        // `[title="..."]` to match against.
        .with_title(app_id)
        .with_decorations(false)
        .with_resizable(false)
        .with_inner_size([width as f32, height as f32])
        .with_visible(false);
    if let Some((x, y)) = initial_position {
        viewport = viewport.with_position([x as f32, y as f32]);
    }
    let options = eframe::NativeOptions {
        viewport,
        // The tray runtime owns the main thread, so run the popup's
        // winit loop on a worker thread instead.
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
            runtime.spawn(wake_egui_on_state_change(waker_rx, cc.egui_ctx.clone()));
            Ok(Box::new(PopupApp::new(state_rx, event_tx)))
        }),
    )
}

/// Forward watch updates to egui as repaint requests; when the sender
/// drops, close the viewport so `eframe::run_native` can return.
async fn wake_egui_on_state_change(mut rx: watch::Receiver<PopupState>, ctx: egui::Context) {
    while rx.changed().await.is_ok() {
        ctx.request_repaint();
    }
    ctx.send_viewport_cmd(ViewportCommand::Close);
    ctx.request_repaint();
}

struct PopupApp {
    state_rx: watch::Receiver<PopupState>,
    event_tx: UnboundedSender<DriverInput>,
    last_visible: bool,
    awaiting_first_paint: bool,
    first_frame: bool,
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

// U+00D7 MULTIPLICATION SIGN, not U+2715/2717: the default eframe font
// covers Latin-1 supplement but not Dingbats.
fn close_button(ui: &mut egui::Ui) -> egui::Response {
    let btn = egui::Button::new(
        RichText::new("\u{00D7}")
            .color(Color32::GRAY)
            .font(FontId::new(14.0, FontFamily::Proportional)),
    )
    .frame(false);
    ui.add(btn).on_hover_text("Close (interrupts the agent)")
}

// Leading marker is U+00B7 MIDDLE DOT, not U+25CF BLACK CIRCLE: the
// default eframe font doesn't cover Geometric Shapes.
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
