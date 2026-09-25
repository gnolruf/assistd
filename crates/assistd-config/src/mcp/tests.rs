use super::*;

#[test]
fn parses_stdio_server_minimally() {
    let toml = r#"
        enabled = true

        [[servers]]
        name = "filesystem"
        transport = "stdio"
        command = "npx"
        args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    "#;
    let cfg: McpConfig = toml::from_str(toml).unwrap();
    assert_eq!(
        cfg,
        McpConfig {
            enabled: true,
            servers: vec![McpServerConfig::Stdio {
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
fn parses_sse_server_minimally() {
    let toml = r#"
        enabled = true

        [[servers]]
        name = "remote"
        transport = "sse"
        url = "https://mcp.example.com/sse"

        [servers.headers]
        Authorization = "Bearer xyz"
    "#;
    let cfg: McpConfig = toml::from_str(toml).unwrap();
    assert_eq!(
        cfg,
        McpConfig {
            enabled: true,
            servers: vec![McpServerConfig::Sse {
                name: "remote".into(),
                url: Url::parse("https://mcp.example.com/sse").unwrap(),
                headers: HashMap::from([("Authorization".into(), "Bearer xyz".into())]),
                request_timeout_secs: DEFAULT_MCP_REQUEST_TIMEOUT_SECS,
            }],
        }
    );
}

#[test]
fn transport_specific_keys_do_not_cross_variants() {
    let stdio_with_url = r#"
        [[servers]]
        name = "x"
        transport = "stdio"
        command = "npx"
        url = "https://example.com/sse"
    "#;
    let err = toml::from_str::<McpConfig>(stdio_with_url)
        .expect_err("`url` on a stdio server must not parse");
    assert!(err.to_string().contains("url"), "{err}");

    let sse_with_command = r#"
        [[servers]]
        name = "x"
        transport = "sse"
        url = "https://example.com/sse"
        command = "npx"
    "#;
    let err = toml::from_str::<McpConfig>(sse_with_command)
        .expect_err("`command` on an sse server must not parse");
    assert!(err.to_string().contains("command"), "{err}");
}

#[test]
fn malformed_url_is_rejected_at_load() {
    let toml = r#"
        [[servers]]
        name = "x"
        transport = "sse"
        url = "not a url"
    "#;
    toml::from_str::<McpConfig>(toml).expect_err("a malformed url must not parse");
}

#[test]
fn zero_request_timeout_is_rejected_at_load() {
    let toml = r#"
        [[servers]]
        name = "x"
        transport = "stdio"
        command = "npx"
        request_timeout_secs = 0
    "#;
    toml::from_str::<McpConfig>(toml).expect_err("a zero timeout must not parse");
}
