use super::*;

fn id(raw: u64) -> WindowId {
    WindowId::new(raw).expect("test ids are non-zero")
}

fn snap() -> RwLock<Snapshot> {
    RwLock::new(Snapshot::default())
}

async fn event(
    snapshot: &RwLock<Snapshot>,
    kind: WindowChangeKind,
    raw: u64,
    class: &str,
    title: &str,
) {
    apply_window_event(
        snapshot,
        kind,
        Some(id(raw)),
        Some(class.into()),
        Some(title.into()),
    )
    .await;
}

fn focused(raw: u64, class: &str, title: &str) -> Option<FocusedWindowContext> {
    Some(FocusedWindowContext {
        id: Some(id(raw)),
        class: Some(class.into()),
        title: Some(title.into()),
        workspace: None,
    })
}

#[tokio::test]
async fn focus_event_overwrites_all_fields() {
    let snapshot = snap();
    event(&snapshot, WindowChangeKind::Focus, 42, "Firefox", "GitHub").await;
    event(&snapshot, WindowChangeKind::Focus, 7, "kitty", "shell").await;
    assert_eq!(
        read_focused_context(&snapshot).await,
        focused(7, "kitty", "shell")
    );
}

#[tokio::test]
async fn title_event_for_other_id_is_ignored() {
    let snapshot = snap();
    event(
        &snapshot,
        WindowChangeKind::Focus,
        42,
        "Firefox",
        "foreground",
    )
    .await;
    event(
        &snapshot,
        WindowChangeKind::Title,
        99,
        "Firefox",
        "background drift",
    )
    .await;
    assert_eq!(
        read_focused_context(&snapshot).await,
        focused(42, "Firefox", "foreground")
    );
}

#[tokio::test]
async fn close_event_for_focused_id_clears_focus() {
    let snapshot = snap();
    event(&snapshot, WindowChangeKind::Focus, 42, "Firefox", "GitHub").await;
    event(&snapshot, WindowChangeKind::Close, 42, "Firefox", "GitHub").await;
    assert_eq!(read_focused_context(&snapshot).await, None);
}
