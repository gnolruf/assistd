use super::*;

fn id(n: u64) -> WindowId {
    WindowId::new(n).expect("test ids are non-zero")
}

fn snap() -> Arc<RwLock<Snapshot>> {
    Arc::new(RwLock::new(Snapshot::default()))
}

#[tokio::test]
async fn focus_event_overwrites_all_fields() {
    let s = snap();
    apply_window_event(
        &s,
        WindowChangeKind::Focus,
        Some(id(42)),
        Some("Firefox".into()),
        Some("GitHub".into()),
    )
    .await;
    let r = s.read().await;
    assert_eq!(r.focused_id, Some(id(42)));
    assert_eq!(r.focused_class.as_deref(), Some("Firefox"));
    assert_eq!(r.focused_title.as_deref(), Some("GitHub"));
}

#[tokio::test]
async fn title_event_for_focused_id_updates_title() {
    let s = snap();
    apply_window_event(
        &s,
        WindowChangeKind::Focus,
        Some(id(42)),
        Some("Firefox".into()),
        Some("Old".into()),
    )
    .await;
    apply_window_event(
        &s,
        WindowChangeKind::Title,
        Some(id(42)),
        Some("Firefox".into()),
        Some("New".into()),
    )
    .await;
    assert_eq!(s.read().await.focused_title.as_deref(), Some("New"));
}

#[tokio::test]
async fn title_event_for_other_id_is_ignored() {
    let s = snap();
    apply_window_event(
        &s,
        WindowChangeKind::Focus,
        Some(id(42)),
        Some("Firefox".into()),
        Some("foreground".into()),
    )
    .await;
    apply_window_event(
        &s,
        WindowChangeKind::Title,
        Some(id(99)),
        Some("Firefox".into()),
        Some("background drift".into()),
    )
    .await;
    assert_eq!(s.read().await.focused_title.as_deref(), Some("foreground"));
}

#[tokio::test]
async fn close_event_for_focused_id_clears_focus() {
    let s = snap();
    apply_window_event(
        &s,
        WindowChangeKind::Focus,
        Some(id(42)),
        Some("Firefox".into()),
        Some("GitHub".into()),
    )
    .await;
    apply_window_event(
        &s,
        WindowChangeKind::Close,
        Some(id(42)),
        Some("Firefox".into()),
        Some("GitHub".into()),
    )
    .await;
    let r = s.read().await;
    assert!(r.focused_id.is_none());
    assert!(r.focused_class.is_none());
    assert!(r.focused_title.is_none());
}

#[tokio::test]
async fn close_event_for_other_id_is_ignored() {
    let s = snap();
    apply_window_event(
        &s,
        WindowChangeKind::Focus,
        Some(id(42)),
        Some("Firefox".into()),
        Some("GitHub".into()),
    )
    .await;
    apply_window_event(
        &s,
        WindowChangeKind::Close,
        Some(id(99)),
        Some("Other".into()),
        Some("Other".into()),
    )
    .await;
    assert_eq!(s.read().await.focused_id, Some(id(42)));
}

#[tokio::test]
async fn workspace_focus_updates_active_workspace() {
    let s = snap();
    apply_workspace_focus(&s, Some("3".into())).await;
    assert_eq!(s.read().await.active_workspace.as_deref(), Some("3"));
}

#[tokio::test]
async fn read_focused_context_returns_none_for_empty() {
    let s = snap();
    assert!(read_focused_context(&s).await.is_none());
}

#[tokio::test]
async fn read_focused_context_returns_some_for_partial() {
    let s = snap();
    apply_workspace_focus(&s, Some("3".into())).await;
    let ctx = read_focused_context(&s).await.unwrap();
    assert!(ctx.id.is_none());
    assert_eq!(ctx.workspace.as_deref(), Some("3"));
}
