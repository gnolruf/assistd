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
    document.save().unwrap();

    let saved = fs::read_to_string(&path).unwrap();
    assert!(saved.starts_with("# assistd configuration\n"));
    assert!(saved.contains("name = \"other/repo:Q8_0\"  # owner/repo:quant"));
    assert!(!saved.contains("port"));
    assert!(!saved.contains("no_such_key"));
    let reloaded = Config::load_from_file(&path).unwrap();
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

    document.set("model.name", &toml::Value::from("")).unwrap();

    assert!(matches!(document.save(), Err(ConfigError::Validation(_))));
    assert!(!path.exists());
}
