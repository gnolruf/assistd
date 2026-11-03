//! The box that picks a theme color: the named colors as a grid of swatches,
//! and red, green and blue channels for any 24-bit color.

use assistd_config::ThemeColor;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

/// Columns one palette cell takes: pointer, swatch, the longest name, a gap.
pub(crate) const PALETTE_CELL_WIDTH: usize = 19;
/// Channel step of Shift+←/→ and PgUp/PgDn.
const COARSE_STEP: u8 = 16;
/// Channels a named color opens on, since it has no fixed RGB value.
const NEUTRAL_RGB: [u8; 3] = [128, 128, 128];
const HEX_DIGITS: usize = 6;

/// What the panel should do after a key in the picker.
pub(crate) enum PickerAction {
    Pending,
    /// The candidate color changed; the preview should follow it.
    Changed,
    Cancel,
    Apply(ThemeColor),
}

/// Where the picker's cursor is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PickerFocus {
    /// A named color, by its index in [`ThemeColor::named`].
    Palette(usize),
    /// Red, green or blue, by index.
    Channel(usize),
}

/// A candidate color, chosen from the palette or mixed from channels.
pub(crate) struct ColorPicker {
    palette: Vec<ThemeColor>,
    focus: PickerFocus,
    rgb: [u8; 3],
    /// Palette cells per row as last drawn; at least 1.
    columns: usize,
    /// Hex digits typed after `#`, until all six are in.
    hex: Option<String>,
}

impl ColorPicker {
    /// Opens on `current`: its palette cell, or the channels for an unnamed color.
    pub(crate) fn new(current: ThemeColor) -> Self {
        let palette: Vec<ThemeColor> = ThemeColor::named().collect();
        let rgb = match current {
            ThemeColor::Rgb { red, green, blue } => [red, green, blue],
            _ => NEUTRAL_RGB,
        };
        let focus = palette
            .iter()
            .position(|named| *named == current)
            .map_or(PickerFocus::Channel(0), PickerFocus::Palette);
        Self {
            palette,
            focus,
            rgb,
            columns: 1,
            hex: None,
        }
    }

    pub(crate) fn palette(&self) -> &[ThemeColor] {
        &self.palette
    }

    pub(crate) fn focus(&self) -> PickerFocus {
        self.focus
    }

    pub(crate) fn rgb(&self) -> [u8; 3] {
        self.rgb
    }

    /// The `#` and digits typed so far, while a hex color is being typed.
    pub(crate) fn hex_input(&self) -> Option<&str> {
        self.hex.as_deref()
    }

    /// The color Enter would apply.
    pub(crate) fn candidate(&self) -> ThemeColor {
        match self.focus {
            PickerFocus::Palette(index) => self.palette[index],
            PickerFocus::Channel(_) => self.mixed(),
        }
    }

    /// The color the channels mix.
    pub(crate) fn mixed(&self) -> ThemeColor {
        let [red, green, blue] = self.rgb;
        ThemeColor::Rgb { red, green, blue }
    }

    /// Records how many palette cells fit across `width` columns, so ↑/↓
    /// move by the rows on screen.
    pub(crate) fn set_width(&mut self, width: u16) {
        self.columns = (usize::from(width) / PALETTE_CELL_WIDTH).max(1);
    }

    pub(crate) fn on_key(&mut self, ev: KeyEvent) -> PickerAction {
        if self.hex.is_some() {
            return self.on_hex_key(ev);
        }
        let before = self.candidate();
        match ev.code {
            KeyCode::Esc => return PickerAction::Cancel,
            KeyCode::Enter => return PickerAction::Apply(before),
            KeyCode::Char('#') => {
                self.hex = Some(String::new());
                return PickerAction::Pending;
            }
            KeyCode::Tab | KeyCode::BackTab => self.switch_region(),
            code => match self.focus {
                PickerFocus::Palette(index) => self.move_in_palette(index, code),
                PickerFocus::Channel(channel) => self.on_channel_key(channel, ev),
            },
        }
        if self.candidate() == before {
            PickerAction::Pending
        } else {
            PickerAction::Changed
        }
    }

    fn on_hex_key(&mut self, ev: KeyEvent) -> PickerAction {
        let Some(hex) = self.hex.as_mut() else {
            return PickerAction::Pending;
        };
        match ev.code {
            KeyCode::Esc => self.hex = None,
            KeyCode::Backspace if hex.pop().is_none() => self.hex = None,
            KeyCode::Char(digit) if digit.is_ascii_hexdigit() => {
                hex.push(digit);
                if hex.len() == HEX_DIGITS {
                    return self.finish_hex();
                }
            }
            _ => {}
        }
        PickerAction::Pending
    }

    fn finish_hex(&mut self) -> PickerAction {
        let typed = self.hex.take().unwrap_or_default();
        match format!("#{typed}").parse() {
            Ok(ThemeColor::Rgb { red, green, blue }) => {
                self.rgb = [red, green, blue];
                self.focus = PickerFocus::Channel(0);
                PickerAction::Changed
            }
            _ => PickerAction::Pending,
        }
    }

    fn switch_region(&mut self) {
        self.focus = match self.focus {
            PickerFocus::Palette(_) => PickerFocus::Channel(0),
            PickerFocus::Channel(_) => PickerFocus::Palette(0),
        };
    }

    /// Arrows move across the grid; ↓ off its last row goes to the channels.
    fn move_in_palette(&mut self, index: usize, code: KeyCode) {
        let last = self.palette.len() - 1;
        self.focus = match code {
            KeyCode::Left => PickerFocus::Palette(index.saturating_sub(1)),
            KeyCode::Right => PickerFocus::Palette((index + 1).min(last)),
            KeyCode::Up => PickerFocus::Palette(index.saturating_sub(self.columns)),
            KeyCode::Down if index + self.columns <= last => {
                PickerFocus::Palette(index + self.columns)
            }
            KeyCode::Down => PickerFocus::Channel(0),
            _ => self.focus,
        };
    }

    /// ↑/↓ pick a channel, ↑ off red returning to the palette; ←/→ and
    /// PgUp/PgDn adjust it.
    fn on_channel_key(&mut self, channel: usize, ev: KeyEvent) {
        let coarse = ev.modifiers.contains(KeyModifiers::SHIFT);
        let step = if coarse { COARSE_STEP } else { 1 };
        let value = &mut self.rgb[channel];
        match ev.code {
            KeyCode::Up if channel == 0 => {
                self.focus = PickerFocus::Palette(self.palette.len() - 1);
            }
            KeyCode::Up => self.focus = PickerFocus::Channel(channel - 1),
            KeyCode::Down => self.focus = PickerFocus::Channel((channel + 1).min(2)),
            KeyCode::Left => *value = value.saturating_sub(step),
            KeyCode::Right => *value = value.saturating_add(step),
            KeyCode::PageDown => *value = value.saturating_sub(COARSE_STEP),
            KeyCode::PageUp => *value = value.saturating_add(COARSE_STEP),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests;
