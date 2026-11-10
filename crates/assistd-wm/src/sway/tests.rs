use serde_json::{Value, json};

use super::*;

fn node(id: i64, node_type: &str, name: &str, nodes: &[Value]) -> Value {
    let rect = json!({ "x": 0, "y": 0, "width": 0, "height": 0 });
    json!({
        "id": id,
        "name": name,
        "type": node_type,
        "border": "none",
        "current_border_width": 0,
        "layout": "splith",
        "orientation": "horizontal",
        "rect": rect,
        "window_rect": rect,
        "deco_rect": rect,
        "geometry": rect,
        "urgent": false,
        "focused": false,
        "focus": [],
        "nodes": nodes,
        "floating_nodes": [],
        "sticky": false,
    })
}

fn view(id: i64, app_id: &str, title: &str) -> Value {
    let mut view = node(id, "con", title, &[]);
    view["app_id"] = json!(app_id);
    view["pid"] = json!(id);
    view
}

#[test]
fn split_container_is_not_listed_as_a_window() {
    let split = node(
        3,
        "con",
        "",
        &[view(4, "foot", "shell"), view(5, "firefox", "docs")],
    );
    let workspace = node(2, "workspace", "1", &[split]);
    let tree: Node = serde_json::from_value(node(1, "root", "root", &[workspace]))
        .expect("tree fixture deserializes");

    let windows = SwayIpc::collect_windows(&tree);

    let listed: Vec<_> = windows
        .iter()
        .map(|window| {
            (
                window.id.get(),
                window.app.as_deref(),
                window.workspace.as_deref(),
            )
        })
        .collect();
    assert_eq!(
        listed,
        [
            (4, Some("foot"), Some("1")),
            (5, Some("firefox"), Some("1"))
        ]
    );
}
