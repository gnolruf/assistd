use super::*;

fn id(n: u64) -> WindowId {
    WindowId::new(n).expect("test ids are non-zero")
}

fn anchor() -> PlacementAnchor {
    PlacementAnchor {
        corner: AnchorCorner::TopRight,
        offset_x: -10,
        offset_y: 10,
        width: 360,
        height: 120,
    }
}

#[tokio::test]
async fn no_window_manager_reports_disconnected_for_every_operation() {
    let wm = NoWindowManager;
    let criteria = PlacementCriteria::AppId("dev.assistd.popup".into());
    let results = [
        ("focus", wm.focus(&id(1)).await),
        (
            "move_to_workspace",
            wm.move_to_workspace(&id(1), &WorkspaceId::Num(3)).await,
        ),
        ("list_windows", wm.list_windows().await.map(drop)),
        ("list_workspaces", wm.list_workspaces().await.map(drop)),
        (
            "resize_width",
            wm.resize_width(&id(1), ResizeDir::Grow, 10).await,
        ),
        ("set_layout", wm.set_layout(Layout::Tabbed).await),
        ("list_outputs", wm.list_outputs().await.map(drop)),
        (
            "place_floating",
            wm.place_floating(&criteria, anchor()).await,
        ),
        (
            "focused_workspace_rect",
            wm.focused_workspace_rect().await.map(drop),
        ),
    ];
    for (op, result) in results {
        assert!(
            matches!(result, Err(WmError::Disconnected)),
            "{op}: {result:?}"
        );
    }
    assert_eq!(wm.focused_window().await.unwrap(), None);
    assert_eq!(wm.focused_context().await.unwrap(), None);
    assert!(!wm.is_connected());
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
        .place_floating(&PlacementCriteria::AppId("x".into()), anchor())
        .await
        .unwrap_err();
    assert!(matches!(err, WmError::Unsupported("floating placement")));
}

#[tokio::test]
async fn default_list_outputs_reports_unsupported() {
    let err = MinimalWm.list_outputs().await.unwrap_err();
    assert!(matches!(err, WmError::Unsupported("output enumeration")));
}

#[test]
fn resize_dir_round_trips() {
    for (s, dir) in [("grow", ResizeDir::Grow), ("shrink", ResizeDir::Shrink)] {
        assert_eq!(s.parse::<ResizeDir>(), Ok(dir));
        assert_eq!(dir.to_string(), s);
    }
    assert_eq!("sideways".parse::<ResizeDir>(), Err(ParseResizeDirError));
}

#[test]
fn layout_round_trips() {
    for (s, l) in [
        ("default", Layout::Default),
        ("tabbed", Layout::Tabbed),
        ("stacking", Layout::Stacking),
        ("splith", Layout::SplitH),
        ("splitv", Layout::SplitV),
    ] {
        assert_eq!(s.parse::<Layout>(), Ok(l));
        assert_eq!(l.to_string(), s);
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

#[test]
fn window_event_matches_opened_by_each_criterion() {
    let titled = WindowEvent::Opened {
        id: id(7),
        title: Some("dev.assistd.popup".into()),
        class: None,
        app_id: None,
    };
    let firefox = WindowEvent::Opened {
        id: id(11),
        title: None,
        class: Some("Firefox".into()),
        app_id: Some("org.mozilla.firefox".into()),
    };
    for (event, criteria, expected) in [
        (
            &titled,
            PlacementCriteria::Title("dev.assistd.popup".into()),
            Some(id(7)),
        ),
        (&titled, PlacementCriteria::Title("other".into()), None),
        (
            &titled,
            PlacementCriteria::Class("dev.assistd.popup".into()),
            None,
        ),
        (
            &firefox,
            PlacementCriteria::Class("Firefox".into()),
            Some(id(11)),
        ),
        (
            &firefox,
            PlacementCriteria::AppId("org.mozilla.firefox".into()),
            Some(id(11)),
        ),
        (&firefox, PlacementCriteria::ConId(id(11)), Some(id(11))),
        (&firefox, PlacementCriteria::ConId(id(99)), None),
    ] {
        assert_eq!(
            event.matches_opened(&criteria),
            expected,
            "{event:?} vs {criteria:?}"
        );
    }
}

#[test]
fn window_event_non_opened_variants_never_match() {
    assert_eq!(
        WindowEvent::Closed { id: id(3) }.matches_opened(&PlacementCriteria::ConId(id(3))),
        None
    );
    assert_eq!(
        WindowEvent::TitleChanged {
            id: id(3),
            new_title: Some("x".into())
        }
        .matches_opened(&PlacementCriteria::Title("x".into())),
        None
    );
}

#[test]
fn is_terminal_class_matches_known_emulators_case_insensitively() {
    for (class, expected) in [
        ("Alacritty", true),
        ("kitty", true),
        ("WezTerm", true),
        ("foot", true),
        ("alacritty", true),
        ("XTERM", true),
        ("xterm", true),
        ("firefox", false),
        ("code", false),
        ("Slack", false),
        ("", false),
    ] {
        assert_eq!(is_terminal_class(class), expected, "{class:?}");
    }
}
