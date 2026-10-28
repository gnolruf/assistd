use serde_json::json;

use super::*;

/// The second monitor sits past 1920 on x; only the primary has `*`.
const XRANDR_DUAL: &str = "Monitors: 2\n \
    0: +*eDP-1 1920/300x1080/180+0+0  eDP-1\n \
    1: +DP-2 2560/600x1440/340+1920+0  DP-2\n";

#[test]
fn hyprland_geom_from_json() {
    let cases = [
        (
            json!({"at": [100, 200], "size": [800, 600]}),
            Some("100,200 800x600"),
        ),
        (json!({"at": [100], "size": [800, 600]}), None),
    ];
    for (window, expected) in cases {
        assert_eq!(
            parse_hyprland_geom(&window).as_deref(),
            expected,
            "{window}"
        );
    }
}

#[test]
fn sway_tree_walk_finds_focused_rect() {
    let cases = [
        (
            json!({
                "focused": true,
                "rect": {"x": 10, "y": 20, "width": 300, "height": 400}
            }),
            Some("10,20 300x400"),
        ),
        (
            json!({
                "focused": false,
                "nodes": [
                    {"focused": false, "nodes": [
                        {"focused": true, "rect": {"x": 5, "y": 6, "width": 7, "height": 8}}
                    ]}
                ]
            }),
            Some("5,6 7x8"),
        ),
        (
            json!({
                "focused": false,
                "floating_nodes": [
                    {"focused": true, "rect": {"x": 1, "y": 2, "width": 3, "height": 4}}
                ]
            }),
            Some("1,2 3x4"),
        ),
        (json!({"focused": false, "nodes": []}), None),
        (json!({"focused": true}), None),
    ];
    for (tree, expected) in cases {
        assert_eq!(find_focused_sway_rect(&tree).as_deref(), expected, "{tree}");
    }
}

#[test]
fn parse_xrandr_picks_the_named_monitor() {
    let cases = [
        ("eDP-1", Some("1920x1080+0+0")),
        ("DP-2", Some("2560x1440+1920+0")),
        ("VGA-1", None),
    ];
    for (monitor, expected) in cases {
        assert_eq!(
            parse_xrandr_monitor_geom(XRANDR_DUAL, monitor).as_deref(),
            expected,
            "{monitor}"
        );
    }
}

#[test]
fn strip_xrandr_geom_token_handles_negative_offsets() {
    assert_eq!(
        strip_xrandr_geom_token("1920/598x1080/336-1920+0").as_deref(),
        Some("1920x1080-1920+0")
    );
}
