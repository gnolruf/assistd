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
fn write_sample_creates_owner_only_file_and_new_parent_dirs() {
    let root = tempfile::tempdir().unwrap();
    let outer = root.path().join("outer");
    let inner = outer.join("assistd");
    let path = inner.join("config.toml");

    Config::write_sample(&path).unwrap();

    assert_eq!(mode_of(&path), 0o600);
    assert_eq!(mode_of(&inner), 0o700);
    assert_eq!(mode_of(&outer), 0o700);
    assert_eq!(Config::load_from_file(&path).unwrap(), Config::default());
}

#[test]
fn write_sample_leaves_existing_parent_mode_alone() {
    let root = tempfile::tempdir().unwrap();
    fs::set_permissions(root.path(), fs::Permissions::from_mode(0o755)).unwrap();

    Config::write_sample(&root.path().join("config.toml")).unwrap();

    assert_eq!(mode_of(root.path()), 0o755);
}

#[test]
fn write_sample_refuses_to_overwrite_an_existing_file() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("config.toml");
    fs::write(&path, "# mine\n").unwrap();

    let error = Config::write_sample(&path).unwrap_err();

    assert!(matches!(error, ConfigError::AlreadyExists(ref existing) if existing == &path));
    assert_eq!(fs::read_to_string(&path).unwrap(), "# mine\n");
}

#[test]
fn only_a_readable_file_with_mcp_env_is_exposed() {
    let with_env: Config = toml::from_str(MCP_SERVER_WITH_ENV).unwrap();
    assert!(with_env.exposes_mcp_secrets(0o640));
    assert!(with_env.exposes_mcp_secrets(0o604));
    assert!(!with_env.exposes_mcp_secrets(0o600));
    let without_env: Config = toml::from_str(MCP_SERVER_WITHOUT_ENV).unwrap();
    assert!(!without_env.exposes_mcp_secrets(0o644));
}

#[test]
fn a_removed_key_fails_to_load() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("config.toml");
    fs::write(&path, "[tools.bash]\nallowed_programs = [\"ls\"]\n").unwrap();

    let err = Config::load_from_file(&path).unwrap_err();

    assert!(
        matches!(
            err,
            ConfigError::Removed {
                key: "tools.bash.allowed_programs",
                ..
            }
        ),
        "{err}"
    );
}

#[test]
fn a_rejected_mcp_header_value_is_not_echoed_by_load() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("config.toml");
    fs::write(
        &path,
        "[[mcp.servers]]\nname = \"remote\"\nurl = \"https://h/mcp\"\n\
         headers = { Authorization = \"Bearer hunter2\\n\" }\n",
    )
    .unwrap();

    let message = Config::load_from_file(&path).unwrap_err().to_string();

    assert!(message.contains("Authorization"), "{message}");
    assert!(message.contains("line 4"), "{message}");
    assert!(!message.contains("hunter2"), "{message}");
}
