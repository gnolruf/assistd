use super::*;

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
