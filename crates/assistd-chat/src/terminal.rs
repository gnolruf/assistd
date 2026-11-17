//! The terminal as the chat and the settings panel hold it: raw mode on the
//! alternate screen, the window title, and the inline-graphics probe.

use std::io::{self, Write};

use anyhow::{Context, Result};
use crossterm::{cursor, event, execute, queue, style, terminal};
use ratatui_image::picker::{Picker, ProtocolType};
use tracing::info;

const WINDOW_TITLE: &str = "assistd";
/// xterm window-ops sequences that save and restore the terminal's title.
const PUSH_WINDOW_TITLE: &str = "\x1b[22;0t";
const POP_WINDOW_TITLE: &str = "\x1b[23;0t";

/// Which input the terminal reports under a [`TerminalGuard`].
#[derive(Clone, Copy)]
pub(super) enum TerminalInput {
    /// Keys and pastes; the mouse stays with the terminal, which keeps its own text selection.
    Keyboard,
    /// Keys, pastes, mouse events and focus changes.
    KeyboardMouseFocus,
}

/// Restores the terminal on drop.
pub(super) struct TerminalGuard {
    input: TerminalInput,
}

impl TerminalGuard {
    /// Enter raw mode and the alternate screen with modifier-aware key reports
    /// and bracketed paste, saving the window title, and install a panic hook that restores the
    /// terminal before the previous hook runs.
    pub(super) fn enter(input: TerminalInput) -> Result<Self> {
        terminal::enable_raw_mode().context("enable_raw_mode")?;
        if let Err(e) = Self::enable_reports(input) {
            let _ = terminal::disable_raw_mode();
            return Err(e).context("EnterAlternateScreen");
        }
        let previous_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let _ = Self::cleanup(input);
            previous_hook(info);
        }));
        Ok(Self { input })
    }

    fn enable_reports(input: TerminalInput) -> io::Result<()> {
        let mut stdout = io::stdout();
        queue!(
            stdout,
            terminal::EnterAlternateScreen,
            style::Print(PUSH_WINDOW_TITLE),
            event::EnableBracketedPaste,
            event::PushKeyboardEnhancementFlags(
                event::KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
            )
        )?;
        if let TerminalInput::KeyboardMouseFocus = input {
            queue!(stdout, event::EnableMouseCapture, event::EnableFocusChange)?;
        }
        stdout.flush()
    }

    fn cleanup(input: TerminalInput) -> io::Result<()> {
        terminal::disable_raw_mode()?;
        let mut stdout = io::stdout();
        if let TerminalInput::KeyboardMouseFocus = input {
            queue!(
                stdout,
                event::DisableFocusChange,
                event::DisableMouseCapture
            )?;
        }
        queue!(
            stdout,
            event::PopKeyboardEnhancementFlags,
            event::DisableBracketedPaste,
            style::Print(POP_WINDOW_TITLE),
            terminal::LeaveAlternateScreen,
            cursor::Show,
        )?;
        stdout.flush()
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = Self::cleanup(self.input);
    }
}

/// Retitle the terminal window after the open session when its title
/// differs from `shown`, the title last set. Control characters are dropped
/// so a model-written title cannot end the escape sequence early.
pub(super) fn sync_window_title(shown: &mut String, session_title: Option<&str>) -> io::Result<()> {
    let wanted = session_title.map_or_else(
        || WINDOW_TITLE.to_string(),
        |title| {
            let printable: String = title.chars().filter(|c| !c.is_control()).collect();
            format!("{WINDOW_TITLE} — {printable}")
        },
    );
    if *shown != wanted {
        execute!(io::stdout(), terminal::SetTitle(&wanted))?;
        *shown = wanted;
    }
    Ok(())
}

/// A picker for terminals that can draw images inline; `None` means
/// `/attach` shows filenames only.
pub(super) fn probe_graphics() -> Option<Picker> {
    match Picker::from_query_stdio() {
        Ok(p)
            if matches!(
                p.protocol_type(),
                ProtocolType::Kitty | ProtocolType::Sixel | ProtocolType::Iterm2
            ) =>
        {
            info!(
                "terminal graphics: {:?} (font_size {:?})",
                p.protocol_type(),
                p.font_size()
            );
            Some(p)
        }
        Ok(p) => {
            info!(
                "terminal graphics: {:?} → /attach will display filenames only",
                p.protocol_type()
            );
            None
        }
        Err(e) => {
            info!("terminal graphics probe failed ({e}); /attach will display filenames only");
            None
        }
    }
}
