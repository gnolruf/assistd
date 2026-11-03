use super::super::catalog::{SECTIONS, Tab};
use super::*;

#[test]
fn every_theme_setting_has_a_sample() {
    let theme = Theme::default();
    let keys = SECTIONS
        .iter()
        .filter(|section| section.tab == Tab::Theme)
        .flat_map(|section| section.settings)
        .map(|setting| setting.key);
    for key in keys {
        let role = theme_role(key);
        assert!(
            sample_pane(role).is_some() || !sample_chrome(role, 40, &theme).is_empty(),
            "{key} has no preview"
        );
    }
}
