use std::os::unix::fs::PermissionsExt;

use super::*;

const COMMENTED: &str = "# assistd configuration\n\n\
    [model]\n\
    name = \"owner/repo:Q4_K_M\"  # owner/repo:quant\n\
    port = 8385\n";

#[test]
fn edits_keep_comments_and_refuse_values_the_schema_rejects() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("config.toml");
    fs::write(&path, COMMENTED).unwrap();
    let mut document = ConfigDocument::open(&path).unwrap();

    document
        .set("model.name", &toml::Value::from("other/repo:Q8_0"))
        .unwrap();
    document
        .set("voice.wakeword.silence_ms", &toml::Value::from(600))
        .unwrap();
    document.reset("model.port").unwrap();
    assert!(
        document
            .set("model.port", &toml::Value::from("not a port"))
            .is_err()
    );
    assert!(
        document
            .set("model.no_such_key", &toml::Value::from(1))
            .is_err()
    );
    assert!(document.is_modified());
    document.save().unwrap();

    let saved = fs::read_to_string(&path).unwrap();
    assert!(saved.starts_with("# assistd configuration\n"));
    assert!(saved.contains("name = \"other/repo:Q8_0\"  # owner/repo:quant"));
    assert!(!saved.contains("port"));
    assert!(!saved.contains("no_such_key"));
    let reloaded = Config::load(&path).unwrap();
    assert_eq!(reloaded.voice.wakeword.silence_ms.get(), 600);
    assert_eq!(reloaded.model.port, Config::default().model.port);
    assert_eq!(
        fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o600
    );
    assert!(!document.is_modified());
}

#[test]
fn save_refuses_a_config_that_does_not_validate() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("config.toml");
    let mut document = ConfigDocument::open(&path).unwrap();

    document
        .set("chat.max_reasoning_tokens", &toml::Value::from(1_000_000))
        .unwrap();

    assert!(matches!(document.save(), Err(ConfigError::Validation(_))));
    assert!(!path.exists());
}

#[test]
fn save_names_who_reads_the_changed_keys_and_refuses_a_file_changed_on_disk() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("config.toml");
    fs::write(&path, COMMENTED).unwrap();
    let mut document = ConfigDocument::open(&path).unwrap();

    document
        .set("notifications.auto_hide_ms", &toml::Value::from(900))
        .unwrap();
    document
        .set("chat.theme.accent", &toml::Value::from("red"))
        .unwrap();
    document
        .set("model.port", &toml::Value::from(8385))
        .unwrap();
    assert_eq!(
        document.save().unwrap(),
        BTreeSet::from([Consumer::Tray, Consumer::Chat])
    );

    let hand_edited = format!("{}\n[memory]\nenabled = false\n", document.document);
    fs::write(&path, &hand_edited).unwrap();
    document
        .set("notifications.brief_when_away", &toml::Value::from(false))
        .unwrap();
    assert!(matches!(
        document.save(),
        Err(ConfigError::ChangedOnDisk { .. })
    ));
    assert_eq!(fs::read_to_string(&path).unwrap(), hand_edited);
    assert!(document.is_modified());

    assert_eq!(
        document.overwrite().unwrap(),
        BTreeSet::from([Consumer::Daemon])
    );
    assert!(!fs::read_to_string(&path).unwrap().contains("[memory]"));
}
