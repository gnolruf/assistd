use super::*;

fn key(code: KeyCode) -> KeyEvent {
    KeyEvent::new(code, KeyModifiers::NONE)
}

fn typed(picker: &mut ColorPicker, text: &str) {
    for c in text.chars() {
        picker.on_key(key(KeyCode::Char(c)));
    }
}

#[test]
fn picker_moves_from_palette_to_channels_and_takes_typed_hex() {
    let mut picker = ColorPicker::new(ThemeColor::Blue);
    picker.set_width(u16::try_from(PALETTE_CELL_WIDTH * 6).unwrap());
    assert_eq!(picker.candidate(), ThemeColor::Blue);
    assert!(matches!(
        picker.on_key(key(KeyCode::Down)),
        PickerAction::Changed
    ));
    assert_eq!(picker.candidate(), ThemeColor::named().nth(12).unwrap());
    picker.on_key(key(KeyCode::Down));
    assert_eq!(picker.focus(), PickerFocus::Channel(0));
    picker.on_key(KeyEvent::new(KeyCode::Right, KeyModifiers::SHIFT));
    assert_eq!(picker.rgb(), [144, 128, 128]);

    typed(&mut picker, "#5e74");
    assert_eq!(picker.hex_input(), Some("5e74"));
    assert!(matches!(
        picker.on_key(key(KeyCode::Enter)),
        PickerAction::Pending
    ));
    typed(&mut picker, "9c");
    let applied = picker.on_key(key(KeyCode::Enter));
    let expected = "#5e749c".parse::<ThemeColor>().unwrap();
    assert!(matches!(applied, PickerAction::Apply(color) if color == expected));
    assert!(matches!(
        picker.on_key(key(KeyCode::Esc)),
        PickerAction::Cancel
    ));
}
