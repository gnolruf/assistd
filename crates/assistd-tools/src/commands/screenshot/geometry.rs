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
    if node.get("focused").and_then(Value::as_bool) == Some(true) {
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
mod tests;
