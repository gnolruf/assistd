//! The settings panel over `config.toml`, opened by `/settings` in the chat
//! or on its own by `assistd settings`, which needs no running daemon.

use std::io;
use std::path::PathBuf;

use anyhow::{Context, Result};
use assistd_config::{Config, ConfigDocument};
use clap::Args;
use crossterm::event::{Event as TermEvent, EventStream};
use futures_util::StreamExt;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;

use super::TerminalGuard;

mod catalog;
mod panel;
mod render;

pub(crate) use panel::SettingsPanel;
pub(crate) use render::render_settings;

#[derive(Args)]
pub(crate) struct SettingsArgs {
    /// Path to config file [default: ~/.config/assistd/config.toml]
    #[arg(long, short)]
    pub config: Option<PathBuf>,
}

/// Edit the config file full-screen until the panel closes. Fails before
/// touching the terminal when the file cannot be read or parsed.
pub(crate) async fn run(args: SettingsArgs) -> Result<()> {
    let path = match args.config {
        Some(path) => path,
        None => Config::default_path()?,
    };
    let mut panel = SettingsPanel::new(ConfigDocument::open(&path)?);
    let _guard = TerminalGuard::enter()?;
    let mut terminal =
        Terminal::new(CrosstermBackend::new(io::stdout())).context("Terminal::new")?;
    let mut events = EventStream::new();
    loop {
        terminal.draw(|frame| render_settings(frame, frame.area(), &panel))?;
        match events.next().await {
            Some(Ok(TermEvent::Key(key))) => {
                if panel.on_key(key).is_break() {
                    break;
                }
            }
            Some(Ok(_)) => {}
            Some(Err(e)) => return Err(e).context("reading terminal events"),
            None => break,
        }
    }
    Ok(())
}
