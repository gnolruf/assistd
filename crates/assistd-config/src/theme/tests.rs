use super::*;

#[test]
fn colors_parse_from_names_and_hex_and_round_trip() {
    for text in [
        "default",
        "bley",
        "dark_gray",
        "light_cyan",
        "#5e749d",
        "#000000",
    ] {
        let color: ThemeColor = text.parse().unwrap();
        assert_eq!(color.to_string(), text);
    }
    assert_eq!(
        "DARK_GRAY".parse::<ThemeColor>().unwrap(),
        ThemeColor::DarkGray
    );
    assert_eq!("#5E749C".parse::<ThemeColor>().unwrap(), ThemeColor::BLEY);
    assert_eq!(ThemeColor::BLEY.to_string(), "bley");
    for text in [
        "", "grey", "#5e749", "#5e749cc", "#+f+f+f", "#gggggg", "5e749c",
    ] {
        assert!(text.parse::<ThemeColor>().is_err(), "{text} parsed");
    }
}
