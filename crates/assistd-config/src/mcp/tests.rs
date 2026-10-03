use super::*;
use crate::Config;

#[test]
fn parses_stdio_server_minimally() {
    let toml = r#"
        enabled = true

        [[servers]]
        name = "filesystem"
        command = "npx"
        args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    "#;
    let cfg: McpConfig = toml::from_str(toml).unwrap();
    assert_eq!(
        cfg,
        McpConfig {
            enabled: true,
            servers: vec![McpServerConfig {
                name: "filesystem".into(),
                command: "npx".into(),
                args: vec![
                    "-y".into(),
                    "@modelcontextprotocol/server-filesystem".into(),
                    "/tmp".into(),
                ],
                env: HashMap::new(),
                request_timeout_secs: DEFAULT_MCP_REQUEST_TIMEOUT_SECS,
            }],
        }
    );
}

#[test]
fn keys_of_the_removed_sse_transport_are_unknown() {
    let toml = r#"
        [[mcp.servers]]
        name = "x"
        transport = "stdio"
        command = "npx"
        url = "https://example.com/sse"
    "#;
    assert_eq!(
        unknown_keys(toml),
        ["mcp.servers[0].transport", "mcp.servers[0].url"]
    );
}

#[test]
fn zero_request_timeout_is_rejected_at_load() {
    let toml = r#"
        [[servers]]
        name = "x"
        command = "npx"
        request_timeout_secs = 0
    "#;
    toml::from_str::<McpConfig>(toml).expect_err("a zero timeout must not parse");
}

fn unknown_keys(toml_src: &str) -> Vec<String> {
    let cfg: Config = toml::from_str(toml_src).expect("config must parse");
    let raw: toml::Table = toml::from_str(toml_src).expect("config must be valid TOML");
    cfg.unknown_keys(&raw).expect("config must serialize")
}

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
