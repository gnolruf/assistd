use super::*;

fn id(raw: u64) -> WindowId {
    WindowId::new(raw).expect("test ids are non-zero")
}

#[tokio::test]
async fn no_window_manager_reports_disconnected_for_every_operation() {
    let wm = NoWindowManager;
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
#[derive(Debug)]
struct MinimalWm;

#[async_trait]
impl WindowManager for MinimalWm {
    async fn focus(&self, _window: &WindowId) -> WmResult<()> {
        Ok(())
    }
    async fn move_to_workspace(
        &self,
        _window: &WindowId,
        _workspace: &WorkspaceId,
    ) -> WmResult<()> {
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
    async fn resize_width(
        &self,
        _window: &WindowId,
        _direction: ResizeDir,
        _pixels: u32,
    ) -> WmResult<()> {
        Ok(())
    }
    async fn set_layout(&self, _layout: Layout) -> WmResult<()> {
        Ok(())
    }
}

#[tokio::test]
async fn default_list_outputs_reports_unsupported() {
    let err = MinimalWm.list_outputs().await.unwrap_err();
    assert!(matches!(err, WmError::Unsupported("output enumeration")));
}

#[test]
fn resize_dir_round_trips() {
    for (keyword, dir) in [("grow", ResizeDir::Grow), ("shrink", ResizeDir::Shrink)] {
        assert_eq!(keyword.parse::<ResizeDir>(), Ok(dir));
        assert_eq!(dir.to_string(), keyword);
    }
    assert_eq!("sideways".parse::<ResizeDir>(), Err(ParseResizeDirError));
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
