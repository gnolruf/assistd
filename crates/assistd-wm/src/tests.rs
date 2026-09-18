use super::*;

#[tokio::test]
async fn no_window_manager_reports_no_focused_window() {
    assert!(NoWindowManager.focused_window().await.unwrap().is_none());
}

fn id1() -> WindowId {
    WindowId::new(1).expect("1 is non-zero")
}

#[tokio::test]
async fn no_window_manager_refuses_focus() {
    assert!(NoWindowManager.focus(&id1()).await.is_err());
}

#[tokio::test]
async fn no_window_manager_refuses_move() {
    assert!(
        NoWindowManager
            .move_to_workspace(&id1(), &WorkspaceId::Num(3))
            .await
            .is_err()
    );
}

#[tokio::test]
async fn no_window_manager_refuses_list_windows() {
    assert!(NoWindowManager.list_windows().await.is_err());
}

#[tokio::test]
async fn no_window_manager_refuses_list_workspaces() {
    assert!(NoWindowManager.list_workspaces().await.is_err());
}

#[tokio::test]
async fn no_window_manager_refuses_resize() {
    let err = NoWindowManager
        .resize_width(&id1(), ResizeDir::Grow, 10)
        .await
        .unwrap_err();
    assert!(matches!(err, WmError::Disconnected));
}

#[tokio::test]
async fn no_window_manager_refuses_layout() {
    let err = NoWindowManager
        .set_layout(Layout::Tabbed)
        .await
        .unwrap_err();
    assert!(matches!(err, WmError::Disconnected));
}

#[test]
fn resize_dir_roundtrips() {
    assert_eq!("grow".parse::<ResizeDir>().unwrap(), ResizeDir::Grow);
    assert_eq!("shrink".parse::<ResizeDir>().unwrap(), ResizeDir::Shrink);
    assert_eq!(ResizeDir::Grow.to_string(), "grow");
    assert!("sideways".parse::<ResizeDir>().is_err());
}

#[test]
fn window_id_parses_decimal_only() {
    assert_eq!(
        "42".parse::<WindowId>().unwrap(),
        WindowId::new(42).unwrap()
    );
    assert!("0".parse::<WindowId>().is_err());
    assert!("Firefox".parse::<WindowId>().is_err());
    assert!("-1".parse::<WindowId>().is_err());
    assert!("0x2a".parse::<WindowId>().is_err());
}

#[test]
fn window_id_display_is_decimal() {
    assert_eq!(WindowId::new(42).unwrap().to_string(), "42");
}

#[test]
fn layout_roundtrips() {
    for (s, l) in [
        ("default", Layout::Default),
        ("tabbed", Layout::Tabbed),
        ("stacking", Layout::Stacking),
        ("splith", Layout::SplitH),
        ("splitv", Layout::SplitV),
    ] {
        assert_eq!(s.parse::<Layout>().unwrap(), l);
        assert_eq!(l.to_string(), s);
    }
    assert!("spinning".parse::<Layout>().is_err());
}

#[tokio::test]
async fn no_window_manager_refuses_list_outputs() {
    assert!(NoWindowManager.list_outputs().await.is_err());
}

#[tokio::test]
async fn no_window_manager_refuses_place_floating() {
    let err = NoWindowManager
        .place_floating(
            &PlacementCriteria::AppId("dev.assistd.popup".into()),
            PlacementAnchor {
                corner: AnchorCorner::TopRight,
                offset_x: -10,
                offset_y: 10,
                width: 360,
                height: 120,
            },
        )
        .await
        .unwrap_err();
    assert!(matches!(err, WmError::Disconnected));
}

#[test]
fn window_event_matches_opened_by_title() {
    let ev = WindowEvent::Opened {
        id: WindowId::new(7).unwrap(),
        title: Some("dev.assistd.popup".into()),
        class: None,
        app_id: None,
    };
    assert_eq!(
        ev.matches_opened(&PlacementCriteria::Title("dev.assistd.popup".into())),
        WindowId::new(7)
    );
    assert_eq!(
        ev.matches_opened(&PlacementCriteria::Title("other".into())),
        None
    );
    assert_eq!(
        ev.matches_opened(&PlacementCriteria::Class("dev.assistd.popup".into())),
        None,
        "title-only event should not match class criteria"
    );
}

#[test]
fn window_event_matches_opened_by_app_id_and_class_and_con_id() {
    let id = WindowId::new(11).unwrap();
    let ev = WindowEvent::Opened {
        id,
        title: None,
        class: Some("Firefox".into()),
        app_id: Some("org.mozilla.firefox".into()),
    };
    assert_eq!(
        ev.matches_opened(&PlacementCriteria::Class("Firefox".into())),
        Some(id)
    );
    assert_eq!(
        ev.matches_opened(&PlacementCriteria::AppId("org.mozilla.firefox".into())),
        Some(id)
    );
    assert_eq!(ev.matches_opened(&PlacementCriteria::ConId(id)), Some(id));
    assert_eq!(
        ev.matches_opened(&PlacementCriteria::ConId(WindowId::new(99).unwrap())),
        None
    );
}

#[test]
fn window_event_non_opened_variants_never_match() {
    let id = WindowId::new(3).unwrap();
    assert_eq!(
        WindowEvent::Closed { id }.matches_opened(&PlacementCriteria::ConId(id)),
        None
    );
    assert_eq!(
        WindowEvent::TitleChanged {
            id,
            new_title: Some("x".into())
        }
        .matches_opened(&PlacementCriteria::Title("x".into())),
        None
    );
}

/// Implements only the required methods, so every default is
/// observable.
struct MinimalWm;

#[async_trait]
impl WindowManager for MinimalWm {
    async fn focus(&self, _w: &WindowId) -> WmResult<()> {
        Ok(())
    }
    async fn move_to_workspace(&self, _w: &WindowId, _ws: &WorkspaceId) -> WmResult<()> {
        Ok(())
    }
    async fn focused_window(&self) -> WmResult<Option<WindowId>> {
        Ok(None)
    }
    async fn list_windows(&self) -> WmResult<Vec<Window>> {
        Ok(Vec::new())
    }
    async fn list_workspaces(&self) -> WmResult<Vec<WorkspaceInfo>> {
        Ok(Vec::new())
    }
    async fn resize_width(&self, _w: &WindowId, _d: ResizeDir, _p: u32) -> WmResult<()> {
        Ok(())
    }
    async fn set_layout(&self, _l: Layout) -> WmResult<()> {
        Ok(())
    }
}

#[tokio::test]
async fn default_place_floating_reports_unsupported() {
    let err = MinimalWm
        .place_floating(
            &PlacementCriteria::AppId("x".into()),
            PlacementAnchor {
                corner: AnchorCorner::Center,
                offset_x: 0,
                offset_y: 0,
                width: 100,
                height: 100,
            },
        )
        .await
        .unwrap_err();
    assert!(matches!(err, WmError::Unsupported(op) if op == "floating placement"));
}

#[tokio::test]
async fn default_list_outputs_reports_unsupported() {
    let err = MinimalWm.list_outputs().await.unwrap_err();
    assert!(matches!(err, WmError::Unsupported(op) if op == "output enumeration"));
}

#[test]
fn no_window_manager_reports_disconnected() {
    assert!(!NoWindowManager.is_connected());
}

#[test]
fn version_is_not_empty() {
    assert!(!version().is_empty());
}

#[test]
fn is_terminal_class_matches_known_emulators() {
    assert!(is_terminal_class("Alacritty"));
    assert!(is_terminal_class("kitty"));
    assert!(is_terminal_class("WezTerm"));
    assert!(is_terminal_class("foot"));
}

#[test]
fn is_terminal_class_is_case_insensitive() {
    assert!(is_terminal_class("alacritty"));
    assert!(is_terminal_class("XTERM"));
    assert!(is_terminal_class("xterm"));
}

#[test]
fn is_terminal_class_rejects_non_terminals() {
    assert!(!is_terminal_class("firefox"));
    assert!(!is_terminal_class("code"));
    assert!(!is_terminal_class(""));
    assert!(!is_terminal_class("Slack"));
}

#[tokio::test]
async fn no_window_manager_focused_context_is_none() {
    assert!(NoWindowManager.focused_context().await.unwrap().is_none());
}
