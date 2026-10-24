//! Translation of `global-hotkey` specs into i3/Sway `bindsym` key
//! combinations.

use assistd_wm::{KeyCombo, Modifier};
use global_hotkey::hotkey::{Code, HotKey, Modifiers};

/// xkb keysym names for every [`Code`] a compositor binding can carry.
const KEYSYMS: &[(Code, &str)] = &[
    (Code::KeyA, "a"),
    (Code::KeyB, "b"),
    (Code::KeyC, "c"),
    (Code::KeyD, "d"),
    (Code::KeyE, "e"),
    (Code::KeyF, "f"),
    (Code::KeyG, "g"),
    (Code::KeyH, "h"),
    (Code::KeyI, "i"),
    (Code::KeyJ, "j"),
    (Code::KeyK, "k"),
    (Code::KeyL, "l"),
    (Code::KeyM, "m"),
    (Code::KeyN, "n"),
    (Code::KeyO, "o"),
    (Code::KeyP, "p"),
    (Code::KeyQ, "q"),
    (Code::KeyR, "r"),
    (Code::KeyS, "s"),
    (Code::KeyT, "t"),
    (Code::KeyU, "u"),
    (Code::KeyV, "v"),
    (Code::KeyW, "w"),
    (Code::KeyX, "x"),
    (Code::KeyY, "y"),
    (Code::KeyZ, "z"),
    (Code::Digit0, "0"),
    (Code::Digit1, "1"),
    (Code::Digit2, "2"),
    (Code::Digit3, "3"),
    (Code::Digit4, "4"),
    (Code::Digit5, "5"),
    (Code::Digit6, "6"),
    (Code::Digit7, "7"),
    (Code::Digit8, "8"),
    (Code::Digit9, "9"),
    (Code::Backquote, "grave"),
    (Code::Backslash, "backslash"),
    (Code::BracketLeft, "bracketleft"),
    (Code::BracketRight, "bracketright"),
    (Code::Comma, "comma"),
    (Code::Equal, "equal"),
    (Code::Minus, "minus"),
    (Code::Period, "period"),
    (Code::Quote, "apostrophe"),
    (Code::Semicolon, "semicolon"),
    (Code::Slash, "slash"),
    (Code::Backspace, "BackSpace"),
    (Code::CapsLock, "Caps_Lock"),
    (Code::ContextMenu, "Menu"),
    (Code::Enter, "Return"),
    (Code::Space, "space"),
    (Code::Tab, "Tab"),
    (Code::Delete, "Delete"),
    (Code::End, "End"),
    (Code::Help, "Help"),
    (Code::Home, "Home"),
    (Code::Insert, "Insert"),
    (Code::PageDown, "Page_Down"),
    (Code::PageUp, "Page_Up"),
    (Code::ArrowDown, "Down"),
    (Code::ArrowLeft, "Left"),
    (Code::ArrowRight, "Right"),
    (Code::ArrowUp, "Up"),
    (Code::Escape, "Escape"),
    (Code::PrintScreen, "Print"),
    (Code::ScrollLock, "Scroll_Lock"),
    (Code::NumLock, "Num_Lock"),
    (Code::Pause, "Pause"),
    (Code::Numpad0, "KP_0"),
    (Code::Numpad1, "KP_1"),
    (Code::Numpad2, "KP_2"),
    (Code::Numpad3, "KP_3"),
    (Code::Numpad4, "KP_4"),
    (Code::Numpad5, "KP_5"),
    (Code::Numpad6, "KP_6"),
    (Code::Numpad7, "KP_7"),
    (Code::Numpad8, "KP_8"),
    (Code::Numpad9, "KP_9"),
    (Code::NumpadAdd, "KP_Add"),
    (Code::NumpadDecimal, "KP_Decimal"),
    (Code::NumpadDivide, "KP_Divide"),
    (Code::NumpadEnter, "KP_Enter"),
    (Code::NumpadEqual, "KP_Equal"),
    (Code::NumpadMultiply, "KP_Multiply"),
    (Code::NumpadSubtract, "KP_Subtract"),
    (Code::F1, "F1"),
    (Code::F2, "F2"),
    (Code::F3, "F3"),
    (Code::F4, "F4"),
    (Code::F5, "F5"),
    (Code::F6, "F6"),
    (Code::F7, "F7"),
    (Code::F8, "F8"),
    (Code::F9, "F9"),
    (Code::F10, "F10"),
    (Code::F11, "F11"),
    (Code::F12, "F12"),
    (Code::F13, "F13"),
    (Code::F14, "F14"),
    (Code::F15, "F15"),
    (Code::F16, "F16"),
    (Code::F17, "F17"),
    (Code::F18, "F18"),
    (Code::F19, "F19"),
    (Code::F20, "F20"),
    (Code::F21, "F21"),
    (Code::F22, "F22"),
    (Code::F23, "F23"),
    (Code::F24, "F24"),
    (Code::AudioVolumeDown, "XF86AudioLowerVolume"),
    (Code::AudioVolumeMute, "XF86AudioMute"),
    (Code::AudioVolumeUp, "XF86AudioRaiseVolume"),
    (Code::MediaPlay, "XF86AudioPlay"),
    (Code::MediaPause, "XF86AudioPause"),
    (Code::MediaPlayPause, "XF86AudioPlay"),
    (Code::MediaStop, "XF86AudioStop"),
    (Code::MediaTrackNext, "XF86AudioNext"),
    (Code::MediaTrackPrevious, "XF86AudioPrev"),
    (Code::MediaRecord, "XF86AudioRecord"),
    (Code::MicrophoneMuteToggle, "XF86AudioMicMute"),
    (Code::LaunchAssistant, "XF86Assistant"),
    (Code::LaunchMail, "XF86Mail"),
    (Code::LaunchApp1, "XF86MyComputer"),
    (Code::LaunchApp2, "XF86Calculator"),
];

/// The `bindsym` combination for `hotkey`; `None` when its key has no
/// keysym in [`KEYSYMS`].
pub(super) fn key_combo(hotkey: &HotKey) -> Option<KeyCombo> {
    let keysym = KEYSYMS
        .iter()
        .find(|(code, _)| *code == hotkey.key)
        .map(|(_, keysym)| *keysym)?;
    Some(KeyCombo::new(modifiers(hotkey.mods), keysym))
}

fn modifiers(mods: Modifiers) -> Vec<Modifier> {
    [
        (Modifiers::META, Modifier::Super),
        (Modifiers::ALT, Modifier::Alt),
        (Modifiers::CONTROL, Modifier::Control),
        (Modifiers::SHIFT, Modifier::Shift),
    ]
    .into_iter()
    .filter(|(flag, _)| mods.contains(*flag))
    .map(|(_, modifier)| modifier)
    .collect()
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::*;

    fn translate(spec: &str) -> Option<String> {
        let hotkey = HotKey::from_str(spec).ok()?;
        key_combo(&hotkey).map(|combo| combo.to_string())
    }

    #[test]
    fn translates_global_hotkey_specs_to_bindsym_syntax() {
        for (spec, expected) in [
            ("Super+Space", "Mod4+space"),
            ("shift+ctrl+alt+super+KeyL", "Mod4+Mod1+Control+Shift+l"),
            ("CmdOrCtrl+Digit1", "Control+1"),
            ("Ctrl+Quote", "Control+apostrophe"),
            ("PageDown", "Page_Down"),
            ("Super+F13", "Mod4+F13"),
            ("Alt+MediaPlayPause", "Mod1+XF86AudioPlay"),
        ] {
            assert_eq!(translate(spec).as_deref(), Some(expected), "{spec:?}");
        }
        for untranslatable in ["Super+Fn", "Super+Lang1", "Super+ShiftLeft"] {
            assert_eq!(translate(untranslatable), None, "{untranslatable:?}");
        }
    }
}
