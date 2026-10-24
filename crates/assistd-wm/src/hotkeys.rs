//! Hotkeys a compositor binds on the daemon's behalf, in i3/Sway `bindsym`
//! key syntax.

use std::fmt;

/// A modifier key, named as `bindsym` writes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Modifier {
    Super,
    Alt,
    Control,
    Shift,
}

impl Modifier {
    /// The `bindsym` name: `Mod4`, `Mod1`, `Control` or `Shift`.
    pub fn as_str(self) -> &'static str {
        match self {
            Modifier::Super => "Mod4",
            Modifier::Alt => "Mod1",
            Modifier::Control => "Control",
            Modifier::Shift => "Shift",
        }
    }
}

/// A key combination like `Mod4+space`. `keysym` must be an xkb keysym
/// name, which holds no whitespace or command separator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyCombo {
    modifiers: Vec<Modifier>,
    keysym: &'static str,
}

impl KeyCombo {
    pub fn new(modifiers: Vec<Modifier>, keysym: &'static str) -> Self {
        Self { modifiers, keysym }
    }
}

impl fmt::Display for KeyCombo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for modifier in &self.modifiers {
            write!(f, "{}+", modifier.as_str())?;
        }
        f.write_str(self.keysym)
    }
}

/// A hotkey to bind, and whether its release is reported as well as its
/// press.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hotkey {
    pub keys: KeyCombo,
    pub report_release: bool,
}

/// Which edge of a key press fired a hotkey.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyEdge {
    Press,
    Release,
}

#[cfg(feature = "sway")]
impl KeyEdge {
    fn as_str(self) -> &'static str {
        match self {
            KeyEdge::Press => "press",
            KeyEdge::Release => "release",
        }
    }
}

/// A bound hotkey firing. `index` is the hotkey's position in the list
/// that was bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HotkeyEvent {
    pub index: usize,
    pub edge: KeyEdge,
}

/// The `bindsym`/`unbindsym` commands for a list of hotkeys, each bound
/// to a `nop` whose tag names the hotkey and edge.
#[cfg(feature = "sway")]
#[derive(Debug)]
pub(crate) struct BindingCommands {
    hotkeys: Vec<Hotkey>,
}

#[cfg(feature = "sway")]
impl BindingCommands {
    const TAG_PREFIX: &'static str = "nop assistd hotkey ";

    pub(crate) fn new(hotkeys: Vec<Hotkey>) -> Self {
        Self { hotkeys }
    }

    pub(crate) fn bind_commands(&self) -> impl Iterator<Item = String> + '_ {
        self.edges().map(|(index, keys, edge)| {
            format!(
                "bindsym {} {keys} {}{index} {}",
                Self::flags(edge),
                Self::TAG_PREFIX,
                edge.as_str()
            )
        })
    }

    pub(crate) fn unbind_commands(&self) -> impl Iterator<Item = String> + '_ {
        self.edges()
            .map(|(_, keys, edge)| format!("unbindsym {} {keys}", Self::flags(edge)))
    }

    /// The hotkey a binding event's command belongs to, if it is one of
    /// these bindings.
    pub(crate) fn event_for(&self, command: &str) -> Option<HotkeyEvent> {
        let (index, edge) = command.strip_prefix(Self::TAG_PREFIX)?.split_once(' ')?;
        let index: usize = index.parse().ok()?;
        let hotkey = self.hotkeys.get(index)?;
        let edge = match edge {
            "press" => KeyEdge::Press,
            "release" if hotkey.report_release => KeyEdge::Release,
            _ => return None,
        };
        Some(HotkeyEvent { index, edge })
    }

    fn edges(&self) -> impl Iterator<Item = (usize, &KeyCombo, KeyEdge)> {
        self.hotkeys.iter().enumerate().flat_map(|(index, hotkey)| {
            let release = hotkey.report_release.then_some(KeyEdge::Release);
            [Some(KeyEdge::Press), release]
                .into_iter()
                .flatten()
                .map(move |edge| (index, &hotkey.keys, edge))
        })
    }

    fn flags(edge: KeyEdge) -> &'static str {
        match edge {
            KeyEdge::Press => "--no-repeat",
            KeyEdge::Release => "--release",
        }
    }
}
