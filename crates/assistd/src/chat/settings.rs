//! The settings panel over `config.toml`, opened by `/settings` in the chat
//! or on its own by `assistd settings`, which needs no running daemon.

use std::io;
use std::path::PathBuf;
use std::pin::pin;

use anyhow::{Context, Result};
use assistd_config::{Config, ConfigDocument};
use clap::Args;
use crossterm::event::{Event as TermEvent, EventStream};
use futures_util::StreamExt;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;

use super::input::paste_as_keys;
use super::{TerminalGuard, TerminalInput, shutdown_signal};

mod catalog;
mod color_picker;
mod list_editor;
mod panel;
mod preview;
mod render;
mod theme_list;

pub(crate) use list_editor::{ListAction, ListEditor};
pub(crate) use panel::{PanelHost, SettingsPanel};
pub(crate) use render::{
    editor_window, list_hint, place_cursor, render_list_editor, render_settings,
    render_wrapped_editor, titled_box,
};

#[derive(Args)]
pub(crate) struct SettingsArgs {
    /// Path to config file [default: ~/.config/assistd/config.toml]
    #[arg(long, short)]
    pub config: Option<PathBuf>,
}

/// Edit the config file full-screen until the panel closes or SIGINT or SIGTERM arrives.
/// Fails before touching the terminal when the file cannot be read or parsed.
pub(crate) async fn run(args: SettingsArgs) -> Result<()> {
    let path = Config::path_or_default(args.config)?;
    let mut panel = SettingsPanel::new(ConfigDocument::open(&path)?, PanelHost::Standalone);
    let mut signalled = pin!(shutdown_signal().context("installing signal handlers")?);
    let _guard = TerminalGuard::enter(TerminalInput::Keyboard)?;
    let mut terminal =
        Terminal::new(CrosstermBackend::new(io::stdout())).context("Terminal::new")?;
    let mut events = EventStream::new();
    loop {
        terminal.draw(|frame| render_settings(frame, frame.area(), &mut panel))?;
        tokio::select! {
            event = events.next() => match event {
                Some(Ok(TermEvent::Key(key))) => {
                    if panel.on_key(key).is_break() {
                        break;
                    }
                }
                Some(Ok(TermEvent::Paste(text))) => {
                    if paste_as_keys(&text)
                        .into_iter()
                        .any(|key| panel.on_key(key).is_break())
                    {
                        break;
                    }
                }
                Some(Ok(_)) => {}
                Some(Err(e)) => return Err(e).context("reading terminal events"),
                None => break,
            },
            () = &mut signalled => break,
        }
    }
    Ok(())
}
