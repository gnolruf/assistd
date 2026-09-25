//! Window and monitor geometry parsed from xrandr, swaymsg and hyprctl
//! output, formatted for the capture binary's `-g` flag.

use serde_json::Value;

/// Find `monitor` in `xrandr --listmonitors` output and return its
/// geometry as maim's `WxH+X+Y`, matching the trailing connector name.
pub(super) fn parse_xrandr_monitor_geom(listing: &str, monitor: &str) -> Option<String> {
    for line in listing.lines() {
        if !line.starts_with(|c: char| c.is_whitespace() || c.is_ascii_digit()) {
            continue;
        }
        let trimmed = line.trim();
        let name = trimmed.split_whitespace().next_back()?;
        if name != monitor {
            continue;
        }
        for token in trimmed.split_whitespace() {
            if let Some(geom) = strip_xrandr_geom_token(token) {
                return Some(geom);
            }
        }
    }
    None
}

/// Reduce an xrandr geometry token `<w>/<wmm>x<h>/<hmm>±<x>±<y>` to
/// maim's `<w>x<h>±<x>±<y>`.
fn strip_xrandr_geom_token(token: &str) -> Option<String> {
    let (lhs, after_x) = token.split_once('x')?;
    let (w_with_mm, _) = lhs.split_once('/')?;
    let w: u32 = w_with_mm.parse().ok()?;
    let h_end = after_x.find(['+', '-']).filter(|i| *i > 0)?;
    let (h_with_mm, _) = after_x[..h_end].split_once('/')?;
    let h: u32 = h_with_mm.parse().ok()?;
    let offsets = &after_x[h_end..];
    let y_start = offsets[1..].find(['+', '-'])? + 1;
    let (x_part, y_part) = offsets.split_at(y_start);
    x_part.parse::<i32>().ok()?;
    y_part.parse::<i32>().ok()?;
    Some(format!("{w}x{h}{x_part}{y_part}"))
}

/// The focused node's rect in a `swaymsg -t get_tree` tree, as grim's
/// `X,Y WxH`.
pub(super) fn find_focused_sway_rect(node: &Value) -> Option<String> {
    if node.get("focused").and_then(|f| f.as_bool()) == Some(true) {
        let rect = node.get("rect")?;
        let x = rect.get("x")?.as_i64()?;
        let y = rect.get("y")?.as_i64()?;
        let w = rect.get("width")?.as_i64()?;
        let h = rect.get("height")?.as_i64()?;
        return Some(format!("{x},{y} {w}x{h}"));
    }
    for key in ["nodes", "floating_nodes"] {
        if let Some(children) = node.get(key).and_then(|n| n.as_array()) {
            for child in children {
                if let Some(rect) = find_focused_sway_rect(child) {
                    return Some(rect);
                }
            }
        }
    }
    None
}

/// `hyprctl activewindow -j` geometry as grim's `X,Y WxH`.
pub(super) fn parse_hyprland_geom(window: &Value) -> Option<String> {
    let at = window.get("at")?.as_array()?;
    let size = window.get("size")?.as_array()?;
    let x = at.first()?.as_i64()?;
    let y = at.get(1)?.as_i64()?;
    let w = size.first()?.as_i64()?;
    let h = size.get(1)?.as_i64()?;
    Some(format!("{x},{y} {w}x{h}"))
}

#[cfg(test)]
mod tests {
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
            (json!({"at": [0, 0]}), None),
            (json!({"at": ["100", "200"], "size": ["800", "600"]}), None),
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

    #[test]
    fn strip_xrandr_geom_token_rejects_unrelated_tokens() {
        assert_eq!(strip_xrandr_geom_token("HDMI-1"), None);
        assert_eq!(strip_xrandr_geom_token("1920/598x1200/336"), None);
    }
}
