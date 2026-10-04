use super::*;

const MCP_SERVER_WITH_ENV: &str = r#"
    [[mcp.servers]]
    name = "github"
    command = "github-mcp"
    env = { API_TOKEN = "secret" }
"#;

const MCP_SERVER_WITHOUT_ENV: &str = r#"
    [[mcp.servers]]
    name = "filesystem"
    command = "npx"
"#;

fn mode_of(path: &Path) -> u32 {
    fs::metadata(path).unwrap().permissions().mode() & 0o777
}

#[test]
fn write_default_creates_owner_only_file_and_new_parent_dirs() {
    let root = tempfile::tempdir().unwrap();
    let outer = root.path().join("outer");
    let inner = outer.join("assistd");
    let path = inner.join("config.toml");

    Config::write_default(&path).unwrap();

    assert_eq!(mode_of(&path), 0o600);
    assert_eq!(mode_of(&inner), 0o700);
    assert_eq!(mode_of(&outer), 0o700);
    assert_eq!(Config::load_from_file(&path).unwrap(), Config::default());
}

#[test]
fn write_default_leaves_existing_parent_mode_alone() {
    let root = tempfile::tempdir().unwrap();
    fs::set_permissions(root.path(), fs::Permissions::from_mode(0o755)).unwrap();

    Config::write_default(&root.path().join("config.toml")).unwrap();

    assert_eq!(mode_of(root.path()), 0o755);
}

#[test]
fn write_default_refuses_to_overwrite_an_existing_file() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("config.toml");
    fs::write(&path, "# mine\n").unwrap();

    let error = Config::write_default(&path).unwrap_err();

    assert!(matches!(error, ConfigError::AlreadyExists(ref existing) if existing == &path));
    assert_eq!(fs::read_to_string(&path).unwrap(), "# mine\n");
}

#[test]
fn readable_file_with_mcp_env_is_exposed() {
    let config: Config = toml::from_str(MCP_SERVER_WITH_ENV).unwrap();
    assert!(config.exposes_mcp_env(0o644));
    assert!(config.exposes_mcp_env(0o640));
    assert!(config.exposes_mcp_env(0o604));
}

#[test]
fn owner_only_file_with_mcp_env_is_not_exposed() {
    let config: Config = toml::from_str(MCP_SERVER_WITH_ENV).unwrap();
    assert!(!config.exposes_mcp_env(0o600));
}

#[test]
fn readable_file_without_mcp_env_is_not_exposed() {
    let config: Config = toml::from_str(MCP_SERVER_WITHOUT_ENV).unwrap();
    assert!(!config.exposes_mcp_env(0o644));
    assert!(!Config::default().exposes_mcp_env(0o644));
}
