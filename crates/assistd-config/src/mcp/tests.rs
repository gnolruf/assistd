use super::*;

#[test]
fn debug_lists_secret_names_but_not_values() {
    let toml = r#"
        [[servers]]
        name = "local"
        command = "server"
        env = { API_TOKEN = "stdio-secret" }
    "#;
    let cfg: McpConfig = toml::from_str(toml).unwrap();
    let rendered = format!("{cfg:?}");
    assert!(rendered.contains("API_TOKEN"), "{rendered}");
    assert!(!rendered.contains("stdio-secret"), "{rendered}");
}
