//! The chat TUI's color for each thing it draws, under `[chat.theme]`.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::defaults::DEFAULT_THEME;
use crate::errors::ThemeColorError;
use crate::settings::{DescribeSettings, describe_fields};

/// Every name a [`ThemeColor`] can be written as, besides `#rrggbb`.
const NAMED_COLORS: [(&str, ThemeColor); 18] = [
    ("default", ThemeColor::Default),
    ("bley", ThemeColor::BLEY),
    ("black", ThemeColor::Black),
    ("red", ThemeColor::Red),
    ("green", ThemeColor::Green),
    ("yellow", ThemeColor::Yellow),
    ("blue", ThemeColor::Blue),
    ("magenta", ThemeColor::Magenta),
    ("cyan", ThemeColor::Cyan),
    ("gray", ThemeColor::Gray),
    ("dark_gray", ThemeColor::DarkGray),
    ("light_red", ThemeColor::LightRed),
    ("light_green", ThemeColor::LightGreen),
    ("light_yellow", ThemeColor::LightYellow),
    ("light_blue", ThemeColor::LightBlue),
    ("light_magenta", ThemeColor::LightMagenta),
    ("light_cyan", ThemeColor::LightCyan),
    ("white", ThemeColor::White),
];

/// A terminal color: the terminal's own default, one of the 16 ANSI colors
/// (`gray` is ANSI 7, `dark_gray` 8, `white` 15), or a 24-bit `#rrggbb`;
/// `bley` names assistd's blue-gray, `#5e749c`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub enum ThemeColor {
    Default,
    Black,
    Red,
    Green,
    Yellow,
    Blue,
    Magenta,
    Cyan,
    Gray,
    DarkGray,
    LightRed,
    LightGreen,
    LightYellow,
    LightBlue,
    LightMagenta,
    LightCyan,
    White,
    Rgb { red: u8, green: u8, blue: u8 },
}

impl ThemeColor {
    /// assistd's blue-gray accent, named `bley`.
    pub const BLEY: Self = Self::Rgb {
        red: 94,
        green: 116,
        blue: 156,
    };

    /// Every color with a name: `default`, `bley`, then the 16 ANSI colors in order.
    pub fn named() -> impl ExactSizeIterator<Item = Self> {
        NAMED_COLORS.iter().map(|&(_, color)| color)
    }
}

impl FromStr for ThemeColor {
    type Err = ThemeColorError;

    /// Accepts a name from the list above, any case, or `#rrggbb`.
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let refused = || ThemeColorError(text.to_string());
        if let Some(hex) = text.strip_prefix('#') {
            return rgb_from_hex(hex).ok_or_else(refused);
        }
        NAMED_COLORS
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(text))
            .map(|&(_, color)| color)
            .ok_or_else(refused)
    }
}

impl TryFrom<String> for ThemeColor {
    type Error = ThemeColorError;

    fn try_from(text: String) -> Result<Self, Self::Error> {
        text.parse()
    }
}

impl fmt::Display for ThemeColor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = NAMED_COLORS.iter().find(|(_, color)| color == self);
        match (name, self) {
            (Some((name, _)), _) => f.write_str(name),
            (None, Self::Rgb { red, green, blue }) => {
                write!(f, "#{red:02x}{green:02x}{blue:02x}")
            }
            (None, _) => f.write_str("default"),
        }
    }
}

impl From<ThemeColor> for String {
    fn from(color: ThemeColor) -> Self {
        color.to_string()
    }
}

/// The color of each thing the chat TUI draws. Every value is a
/// [`ThemeColor`]; `default` leaves the terminal's own color. Bold text keeps
/// the color of the text around it.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct ThemeConfig {
    /// Logo, headings, pointers, list markers, tool commands and notices.
    pub accent: ThemeColor,
    /// Background of the user's messages.
    pub user_message: ThemeColor,
    /// Reply prose; bold text in it keeps this color.
    pub assistant_text: ThemeColor,
    /// Secondary text: thinking, summaries, hints, counts, link URLs,
    /// running tool calls and sub-agents, and the bottom status line.
    pub status_text: ThemeColor,
    /// Inline code, code blocks and tool output.
    pub code: ThemeColor,
    /// Rules, table separators and box borders.
    pub border: ThemeColor,
    /// Tool calls that exited 0, sub-agents that answered, a heard wake word
    /// and the settings panel's notes.
    pub success: ThemeColor,
    /// Errors, failed tool calls and their stderr, failed sub-agents, and
    /// refused settings changes.
    pub error: ThemeColor,
    /// The confirmation prompt, unsaved settings and config problems.
    pub warning: ThemeColor,
    /// Link text in replies.
    pub link: ThemeColor,
    /// Background of the selected entry in lists and popups; `default`
    /// shows it in reverse video.
    pub selection: ThemeColor,
}

impl ThemeConfig {
    /// The themes assistd ships, by name: its own first, then palettes
    /// popular across terminal applications, all for dark backgrounds.
    pub const BUILT_IN: [(&'static str, Self); 6] = [
        ("assistd", DEFAULT_THEME),
        ("gruvbox", GRUVBOX),
        ("catppuccin", CATPPUCCIN),
        ("nord", NORD),
        ("dracula", DRACULA),
        ("tokyo_night", TOKYO_NIGHT),
    ];

    /// The built-in theme called `name`.
    pub fn built_in(name: &str) -> Option<Self> {
        Self::BUILT_IN
            .iter()
            .find(|(built_in, _)| *built_in == name)
            .map(|&(_, theme)| theme)
    }
}

impl DescribeSettings for ThemeConfig {
    describe_fields! {
        accent => ("accent", "Logo, headings, pointers, tool commands and notices"),
        user_message => ("user messages", "Background of user messages"),
        assistant_text => ("reply text", "Reply prose and its bold text"),
        status_text => ("status text", "Thinking, summaries, hints, running tools, the status line"),
        code => ("code", "Inline code, code blocks and tool output"),
        border => ("borders", "Rules, table separators and box borders"),
        success => ("success", "Passed tool calls, answered sub-agents, settings notes"),
        error => ("errors", "Errors, failed tool calls and refused settings changes"),
        warning => ("warnings", "Confirmation prompts, unsaved settings, config problems"),
        link => ("links", "Link text in replies"),
        selection => ("selection", "Background of the selected entry; default shows it reversed"),
    }
}

impl Default for ThemeConfig {
    fn default() -> Self {
        DEFAULT_THEME
    }
}

/// Whether `name` can name a saved theme: ASCII letters, digits, `_` and `-`.
pub fn is_theme_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
}

const fn rgb(red: u8, green: u8, blue: u8) -> ThemeColor {
    ThemeColor::Rgb { red, green, blue }
}

const GRUVBOX: ThemeConfig = ThemeConfig {
    accent: rgb(0xfe, 0x80, 0x19),
    user_message: rgb(0x3c, 0x38, 0x36),
    assistant_text: rgb(0xeb, 0xdb, 0xb2),
    status_text: rgb(0x92, 0x83, 0x74),
    code: rgb(0x8e, 0xc0, 0x7c),
    border: rgb(0x50, 0x49, 0x45),
    success: rgb(0xb8, 0xbb, 0x26),
    error: rgb(0xfb, 0x49, 0x34),
    warning: rgb(0xfa, 0xbd, 0x2f),
    link: rgb(0x83, 0xa5, 0x98),
    selection: rgb(0x50, 0x49, 0x45),
};

const CATPPUCCIN: ThemeConfig = ThemeConfig {
    accent: rgb(0xcb, 0xa6, 0xf7),
    user_message: rgb(0x31, 0x32, 0x44),
    assistant_text: rgb(0xcd, 0xd6, 0xf4),
    status_text: rgb(0x7f, 0x84, 0x9c),
    code: rgb(0x94, 0xe2, 0xd5),
    border: rgb(0x45, 0x47, 0x5a),
    success: rgb(0xa6, 0xe3, 0xa1),
    error: rgb(0xf3, 0x8b, 0xa8),
    warning: rgb(0xf9, 0xe2, 0xaf),
    link: rgb(0x89, 0xb4, 0xfa),
    selection: rgb(0x45, 0x47, 0x5a),
};

const NORD: ThemeConfig = ThemeConfig {
    accent: rgb(0x88, 0xc0, 0xd0),
    user_message: rgb(0x3b, 0x42, 0x52),
    assistant_text: rgb(0xd8, 0xde, 0xe9),
    status_text: rgb(0x61, 0x6e, 0x88),
    code: rgb(0x8f, 0xbc, 0xbb),
    border: rgb(0x4c, 0x56, 0x6a),
    success: rgb(0xa3, 0xbe, 0x8c),
    error: rgb(0xbf, 0x61, 0x6a),
    warning: rgb(0xeb, 0xcb, 0x8b),
    link: rgb(0x81, 0xa1, 0xc1),
    selection: rgb(0x43, 0x4c, 0x5e),
};

const DRACULA: ThemeConfig = ThemeConfig {
    accent: rgb(0xbd, 0x93, 0xf9),
    user_message: rgb(0x44, 0x47, 0x5a),
    assistant_text: rgb(0xf8, 0xf8, 0xf2),
    status_text: rgb(0x62, 0x72, 0xa4),
    code: rgb(0xff, 0xb8, 0x6c),
    border: rgb(0x44, 0x47, 0x5a),
    success: rgb(0x50, 0xfa, 0x7b),
    error: rgb(0xff, 0x55, 0x55),
    warning: rgb(0xf1, 0xfa, 0x8c),
    link: rgb(0x8b, 0xe9, 0xfd),
    selection: rgb(0x44, 0x47, 0x5a),
};

const TOKYO_NIGHT: ThemeConfig = ThemeConfig {
    accent: rgb(0x7a, 0xa2, 0xf7),
    user_message: rgb(0x29, 0x2e, 0x42),
    assistant_text: rgb(0xc0, 0xca, 0xf5),
    status_text: rgb(0x56, 0x5f, 0x89),
    code: rgb(0x7d, 0xcf, 0xff),
    border: rgb(0x3b, 0x42, 0x61),
    success: rgb(0x9e, 0xce, 0x6a),
    error: rgb(0xf7, 0x76, 0x8e),
    warning: rgb(0xe0, 0xaf, 0x68),
    link: rgb(0xbb, 0x9a, 0xf7),
    selection: rgb(0x28, 0x34, 0x57),
};

/// The color six hex digits spell, or `None` when `hex` is anything else.
fn rgb_from_hex(hex: &str) -> Option<ThemeColor> {
    if hex.len() != 6 || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    let channel = |at: usize| u8::from_str_radix(&hex[at..at + 2], 16).ok();
    Some(ThemeColor::Rgb {
        red: channel(0)?,
        green: channel(2)?,
        blue: channel(4)?,
    })
}

#[cfg(test)]
mod tests;
