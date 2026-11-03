use crossterm::event::KeyModifiers;

use super::*;

fn press(panel: &mut SettingsPanel, code: KeyCode) {
    let _ = panel.on_key(KeyEvent::new(code, KeyModifiers::NONE));
}

fn typed(panel: &mut SettingsPanel, text: &str) {
    for c in text.chars() {
        press(panel, KeyCode::Char(c));
    }
}

/// The panel on the Theme tab's theme list, over an empty config file.
fn theme_list_panel(root: &tempfile::TempDir) -> SettingsPanel {
    let document = ConfigDocument::open(&root.path().join("config.toml")).unwrap();
    let mut panel = SettingsPanel::new(document);
    while panel.tab() != Tab::Theme {
        press(&mut panel, KeyCode::Tab);
    }
    press(&mut panel, KeyCode::Enter);
    assert!(matches!(panel.mode(), PanelMode::Themes { .. }));
    panel
}

#[test]
fn themes_load_save_and_delete_through_the_list() {
    let root = tempfile::tempdir().unwrap();
    let mut panel = theme_list_panel(&root);
    assert_eq!(panel.active_theme_name(), "assistd");
    let nord = ThemeConfig::built_in("nord").unwrap();
    while !matches!(panel.mode(), PanelMode::Themes { list, .. } if list.highlighted().is_some_and(|e| e.name == "nord"))
    {
        press(&mut panel, KeyCode::Down);
    }
    assert_eq!(panel.theme(), Theme::new(&nord));
    press(&mut panel, KeyCode::Enter);
    assert_eq!(panel.document().config().chat.theme, nord);
    assert_eq!(panel.active_theme_name(), "nord");

    press(&mut panel, KeyCode::Enter);
    for _ in 0..ThemeConfig::BUILT_IN.len() {
        press(&mut panel, KeyCode::Down);
    }
    press(&mut panel, KeyCode::Enter);
    typed(&mut panel, "nord");
    press(&mut panel, KeyCode::Enter);
    assert!(matches!(panel.feedback(), Some(Feedback::Error(_))));
    press(&mut panel, KeyCode::Esc);
    press(&mut panel, KeyCode::Enter);
    typed(&mut panel, "mine");
    press(&mut panel, KeyCode::Enter);
    assert_eq!(
        panel.document().config().chat.themes.get("mine"),
        Some(&nord)
    );

    press(&mut panel, KeyCode::Char('d'));
    press(&mut panel, KeyCode::Char('y'));
    assert!(panel.document().config().chat.themes.is_empty());
    for _ in 0..ThemeConfig::BUILT_IN.len() {
        press(&mut panel, KeyCode::Up);
    }
    press(&mut panel, KeyCode::Enter);
    assert_eq!(panel.active_theme_name(), "assistd");
    assert!(
        panel
            .document()
            .file_value("chat.theme")
            .is_none_or(|table| { table.as_table().is_some_and(toml::Table::is_empty) })
    );
}
