use super::*;

fn id(n: u64) -> WindowId {
    WindowId::new(n).expect("test ids are non-zero")
}

fn snap() -> RwLock<Snapshot> {
    RwLock::new(Snapshot::default())
}

async fn event(s: &RwLock<Snapshot>, kind: WindowChangeKind, raw: u64, class: &str, title: &str) {
    apply_window_event(
        s,
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
    let s = snap();
    event(&s, WindowChangeKind::Focus, 42, "Firefox", "GitHub").await;
    event(&s, WindowChangeKind::Focus, 7, "kitty", "shell").await;
    assert_eq!(read_focused_context(&s).await, focused(7, "kitty", "shell"));
}

#[tokio::test]
async fn title_event_for_focused_id_updates_title_and_class() {
    let s = snap();
    event(&s, WindowChangeKind::Focus, 42, "Firefox", "Old").await;
    event(&s, WindowChangeKind::Title, 42, "firefox", "New").await;
    assert_eq!(
        read_focused_context(&s).await,
        focused(42, "firefox", "New")
    );
}

#[tokio::test]
async fn title_event_for_other_id_is_ignored() {
    let s = snap();
    event(&s, WindowChangeKind::Focus, 42, "Firefox", "foreground").await;
    event(
        &s,
        WindowChangeKind::Title,
        99,
        "Firefox",
        "background drift",
    )
    .await;
    assert_eq!(
        read_focused_context(&s).await,
        focused(42, "Firefox", "foreground")
    );
}

#[tokio::test]
async fn close_event_for_focused_id_clears_focus() {
    let s = snap();
    event(&s, WindowChangeKind::Focus, 42, "Firefox", "GitHub").await;
    event(&s, WindowChangeKind::Close, 42, "Firefox", "GitHub").await;
    assert_eq!(read_focused_context(&s).await, None);
}

#[tokio::test]
async fn close_event_for_other_id_is_ignored() {
    let s = snap();
    event(&s, WindowChangeKind::Focus, 42, "Firefox", "GitHub").await;
    event(&s, WindowChangeKind::Close, 99, "Other", "Other").await;
    assert_eq!(
        read_focused_context(&s).await,
        focused(42, "Firefox", "GitHub")
    );
}

#[tokio::test]
async fn read_focused_context_returns_none_for_empty() {
    assert_eq!(read_focused_context(&snap()).await, None);
}

#[tokio::test]
async fn workspace_focus_alone_yields_a_partial_context() {
    let s = snap();
    apply_workspace_focus(&s, Some("3".into())).await;
    assert_eq!(
        read_focused_context(&s).await,
        Some(FocusedWindowContext {
            workspace: Some("3".into()),
            ..FocusedWindowContext::default()
        })
    );
}
