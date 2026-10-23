use std::collections::HashSet;

use assistd_config::{Config, ConfigDocument};

use super::*;

fn collect_leaf_keys(table: &toml::Table, prefix: &str, keys: &mut Vec<String>) {
    for (name, value) in table {
        let key = if prefix.is_empty() {
            name.clone()
        } else {
            format!("{prefix}.{name}")
        };
        match value {
            toml::Value::Table(child) => collect_leaf_keys(child, &key, keys),
            _ => keys.push(key),
        }
    }
}

fn sample(kind: SettingKind) -> toml::Value {
    match kind {
        SettingKind::Toggle => toml::Value::Boolean(false),
        SettingKind::Integer => toml::Value::Integer(1),
        SettingKind::Decimal => toml::Value::Float(0.5),
        SettingKind::Text | SettingKind::Prose => toml::Value::from("x"),
        SettingKind::Choice(options) => toml::Value::from(options[0]),
        SettingKind::List | SettingKind::External => toml::Value::Array(Vec::new()),
    }
}

#[test]
fn catalog_lists_every_schema_key_and_only_schema_keys() {
    let defaults = toml::Table::try_from(Config::default()).unwrap();
    let mut schema_keys = Vec::new();
    collect_leaf_keys(&defaults, "", &mut schema_keys);
    let settings = || {
        SECTIONS
            .iter()
            .flat_map(|section| section.enable.iter().chain(section.settings))
    };
    let listed: HashSet<&str> = settings().map(|setting| setting.key).collect();
    for key in &schema_keys {
        assert!(listed.contains(key.as_str()), "{key} is not in the catalog");
    }

    let root = tempfile::tempdir().unwrap();
    let mut document = ConfigDocument::open(&root.path().join("config.toml")).unwrap();
    for setting in settings() {
        let value = document
            .default_value(setting.key)
            .cloned()
            .unwrap_or_else(|| sample(setting.kind));
        if let Err(e) = document.set(setting.key, &value) {
            panic!("{}: {e}", setting.key);
        }
    }
}
