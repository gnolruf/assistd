//! The chat TUI's colors, resolved from `[chat.theme]`.

use assistd_config::{ThemeColor, ThemeConfig};
use ratatui::style::{Color, Modifier, Style};

/// Each role of [`ThemeConfig`] as the color ratatui draws it in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Theme {
    pub accent: Color,
    pub user_message: Color,
    pub assistant_text: Color,
    pub status_text: Color,
    pub code: Color,
    pub border: Color,
    pub success: Color,
    pub error: Color,
    pub warning: Color,
    pub link: Color,
    selection: Color,
}

impl Theme {
    pub(crate) fn new(config: &ThemeConfig) -> Self {
        Self {
            accent: color(config.accent),
            user_message: color(config.user_message),
            assistant_text: color(config.assistant_text),
            status_text: color(config.status_text),
            code: color(config.code),
            border: color(config.border),
            success: color(config.success),
            error: color(config.error),
            warning: color(config.warning),
            link: color(config.link),
            selection: color(config.selection),
        }
    }

    /// The selected entry: on the `selection` background, or in reverse
    /// video when that is the terminal default.
    pub(crate) fn selection(&self) -> Style {
        match self.selection {
            Color::Reset => Style::default().add_modifier(Modifier::REVERSED),
            background => Style::default().bg(background),
        }
    }
}

impl Default for Theme {
    fn default() -> Self {
        Self::new(&ThemeConfig::default())
    }
}

/// The ratatui color that draws `color`.
pub(crate) fn color(color: ThemeColor) -> Color {
    match color {
        ThemeColor::Default => Color::Reset,
        ThemeColor::Black => Color::Black,
        ThemeColor::Red => Color::Red,
        ThemeColor::Green => Color::Green,
        ThemeColor::Yellow => Color::Yellow,
        ThemeColor::Blue => Color::Blue,
        ThemeColor::Magenta => Color::Magenta,
        ThemeColor::Cyan => Color::Cyan,
        ThemeColor::Gray => Color::Gray,
        ThemeColor::DarkGray => Color::DarkGray,
        ThemeColor::LightRed => Color::LightRed,
        ThemeColor::LightGreen => Color::LightGreen,
        ThemeColor::LightYellow => Color::LightYellow,
        ThemeColor::LightBlue => Color::LightBlue,
        ThemeColor::LightMagenta => Color::LightMagenta,
        ThemeColor::LightCyan => Color::LightCyan,
        ThemeColor::White => Color::White,
        ThemeColor::Rgb { red, green, blue } => Color::Rgb(red, green, blue),
    }
}
