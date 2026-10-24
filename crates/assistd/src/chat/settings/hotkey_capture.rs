//! Translation of a key press in the terminal into the `global-hotkey` spec
//! a hotkey setting stores, such as `Super+Shift+L`.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

/// What a key press does while a hotkey is being captured.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Capture {
    /// Bind the hotkey to this spec.
    Bind(String),
    /// Clear the hotkey, disabling it.
    Clear,
    Cancel,
    /// The key has no hotkey name; the reason says which.
    Unsupported(String),
}

/// Plain Esc cancels and plain Backspace clears; any other key, with the
/// modifiers held, is bound.
pub(crate) fn capture(ev: KeyEvent) -> Capture {
    match (ev.code, ev.modifiers.is_empty()) {
        (KeyCode::Esc, true) => return Capture::Cancel,
        (KeyCode::Backspace, true) => return Capture::Clear,
        _ => {}
    }
    let shifted = ev.modifiers.contains(KeyModifiers::SHIFT)
        || matches!(ev.code, KeyCode::BackTab)
        || matches!(ev.code, KeyCode::Char(c) if c.is_ascii_uppercase());
    let key = match ev.code {
        KeyCode::BackTab => "Tab".to_string(),
        KeyCode::Char(c) if c.is_ascii_uppercase() => c.to_string(),
        code => match key_name(code) {
            Some(key) => key,
            None => return Capture::Unsupported(format!("{code} cannot be a hotkey")),
        },
    };
    let mut spec: String = [
        (KeyModifiers::SUPER | KeyModifiers::META, "Super+"),
        (KeyModifiers::CONTROL, "Ctrl+"),
        (KeyModifiers::ALT, "Alt+"),
    ]
    .into_iter()
    .filter(|(flag, _)| ev.modifiers.intersects(*flag))
    .map(|(_, name)| name)
    .collect();
    if shifted {
        spec.push_str("Shift+");
    }
    spec.push_str(&key);
    Capture::Bind(spec)
}

/// The `global-hotkey` name of `code` unshifted; `None` for keys it has no
/// name for, shifted symbols among them.
fn key_name(code: KeyCode) -> Option<String> {
    let name = match code {
        KeyCode::Char(' ') => "Space",
        KeyCode::Char(c) if c.is_ascii_lowercase() => return Some(c.to_ascii_uppercase().into()),
        KeyCode::Char(c) if c.is_ascii_digit() || "`\\[],=-.';/".contains(c) => {
            return Some(c.into());
        }
        KeyCode::F(number @ 1..=24) => return Some(format!("F{number}")),
        KeyCode::Enter => "Enter",
        KeyCode::Tab => "Tab",
        KeyCode::Backspace => "Backspace",
        KeyCode::Delete => "Delete",
        KeyCode::Insert => "Insert",
        KeyCode::Home => "Home",
        KeyCode::End => "End",
        KeyCode::PageUp => "PageUp",
        KeyCode::PageDown => "PageDown",
        KeyCode::Up => "Up",
        KeyCode::Down => "Down",
        KeyCode::Left => "Left",
        KeyCode::Right => "Right",
        KeyCode::Esc => "Escape",
        KeyCode::CapsLock => "CapsLock",
        KeyCode::ScrollLock => "ScrollLock",
        KeyCode::NumLock => "NumLock",
        KeyCode::PrintScreen => "PrintScreen",
        KeyCode::Pause => "Pause",
        KeyCode::Menu => "ContextMenu",
        _ => return None,
    };
    Some(name.to_string())
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use global_hotkey::hotkey::{Code, HotKey, Modifiers};

    use super::*;

    fn bound(code: KeyCode, modifiers: KeyModifiers) -> HotKey {
        match capture(KeyEvent::new(code, modifiers)) {
            Capture::Bind(spec) => HotKey::from_str(&spec).unwrap(),
            other => panic!("{code:?} {modifiers:?}: {other:?}"),
        }
    }

    #[test]
    fn captured_presses_become_specs_the_daemon_parses() {
        for (code, modifiers, mods, key) in [
            (
                KeyCode::Char(' '),
                KeyModifiers::SUPER,
                Modifiers::META,
                Code::Space,
            ),
            (
                KeyCode::Char('L'),
                KeyModifiers::SUPER | KeyModifiers::SHIFT,
                Modifiers::META | Modifiers::SHIFT,
                Code::KeyL,
            ),
            (
                KeyCode::Char('\''),
                KeyModifiers::CONTROL | KeyModifiers::ALT,
                Modifiers::CONTROL | Modifiers::ALT,
                Code::Quote,
            ),
            (
                KeyCode::BackTab,
                KeyModifiers::SHIFT,
                Modifiers::SHIFT,
                Code::Tab,
            ),
            (
                KeyCode::F(13),
                KeyModifiers::NONE,
                Modifiers::empty(),
                Code::F13,
            ),
            (
                KeyCode::Esc,
                KeyModifiers::SUPER,
                Modifiers::META,
                Code::Escape,
            ),
        ] {
            assert_eq!(bound(code, modifiers), HotKey::new(Some(mods), key));
        }
        assert_eq!(
            capture(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
            Capture::Cancel
        );
        assert_eq!(
            capture(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE)),
            Capture::Clear
        );
        assert!(matches!(
            capture(KeyEvent::new(KeyCode::Char(':'), KeyModifiers::SHIFT)),
            Capture::Unsupported(_)
        ));
    }
}
