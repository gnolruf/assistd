use super::*;
#[cfg(feature = "sway")]
use crate::hotkeys::BindingCommands;

fn id(raw: u64) -> WindowId {
    WindowId::new(raw).expect("test ids are non-zero")
}

#[test]
fn layout_round_trips() {
    for (keyword, layout) in [
        ("default", Layout::Default),
        ("tabbed", Layout::Tabbed),
        ("stacking", Layout::Stacking),
        ("splith", Layout::SplitH),
        ("splitv", Layout::SplitV),
    ] {
        assert_eq!(keyword.parse::<Layout>(), Ok(layout));
        assert_eq!(layout.to_string(), keyword);
    }
    assert_eq!("spinning".parse::<Layout>(), Err(ParseLayoutError));
}

#[test]
fn window_id_round_trips_positive_decimal_only() {
    assert_eq!("42".parse::<WindowId>(), Ok(id(42)));
    assert_eq!(id(42).to_string(), "42");
    for bad in ["0", "Firefox", "-1", "0x2a", ""] {
        assert_eq!(bad.parse::<WindowId>(), Err(ParseWindowIdError), "{bad:?}");
    }
}

#[cfg(feature = "sway")]
#[test]
fn binding_commands_round_trip_their_nop_tags() {
    let commands = BindingCommands::new(vec![
        Hotkey {
            keys: KeyCombo::new(vec![Modifier::Super], "space"),
            report_release: true,
        },
        Hotkey {
            keys: KeyCombo::new(vec![Modifier::Super, Modifier::Shift], "l"),
            report_release: false,
        },
    ]);
    assert_eq!(
        commands.bind_commands().collect::<Vec<_>>(),
        [
            "bindsym --no-repeat Mod4+space nop assistd hotkey 0 press",
            "bindsym --release Mod4+space nop assistd hotkey 0 release",
            "bindsym --no-repeat Mod4+Shift+l nop assistd hotkey 1 press",
        ]
    );
    assert_eq!(
        commands.unbind_commands().collect::<Vec<_>>(),
        [
            "unbindsym --no-repeat Mod4+space",
            "unbindsym --release Mod4+space",
            "unbindsym --no-repeat Mod4+Shift+l",
        ]
    );
    assert_eq!(
        commands.event_for("nop assistd hotkey 0 release"),
        Some(HotkeyEvent {
            index: 0,
            edge: KeyEdge::Release
        })
    );
    for foreign in [
        "nop assistd hotkey 1 release",
        "nop assistd hotkey 2 press",
        "exec assistd ptt-start",
    ] {
        assert_eq!(commands.event_for(foreign), None, "{foreign:?}");
    }
}
