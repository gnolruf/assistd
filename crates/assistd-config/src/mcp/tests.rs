use super::*;

fn parse_server(entry: &str) -> Result<McpServerConfig, String> {
    toml::from_str::<McpConfig>(&format!("[[servers]]\nname = \"s\"\n{entry}"))
        .map(|cfg| cfg.servers.into_iter().next().unwrap())
        .map_err(|e| e.message().to_owned())
}

#[test]
fn debug_lists_secret_names_but_not_values() {
    let toml = r#"
        [[servers]]
        name = "local"
        command = "server"
        env = { API_TOKEN = "stdio-secret" }

        [[servers]]
        name = "remote"
        url = "https://mcp.example.com/mcp"
        headers = { Authorization = "Bearer http-secret" }
    "#;
    let cfg: McpConfig = toml::from_str(toml).unwrap();
    let rendered = format!("{cfg:?}");
    assert!(rendered.contains("API_TOKEN"), "{rendered}");
    assert!(rendered.contains("Authorization"), "{rendered}");
    assert!(!rendered.contains("stdio-secret"), "{rendered}");
    assert!(!rendered.contains("http-secret"), "{rendered}");
}

#[test]
fn entry_shape_picks_the_transport() {
    let stdio = parse_server("command = \"npx\"\nargs = [\"-y\"]\n").unwrap();
    assert!(matches!(
        stdio.transport,
        McpTransport::Stdio(McpStdioTransport { ref args, .. }) if args == &["-y"]
    ));
    for url in [
        "https://drivemcp.googleapis.com/mcp/v1",
        "http://127.0.0.1:8080/mcp",
        "http://[::1]/mcp",
        "http://localhost/mcp",
    ] {
        let http = parse_server(&format!(
            "url = \"{url}\"\nheaders = {{ Authorization = \"Bearer k\" }}\n"
        ))
        .unwrap_or_else(|e| panic!("{url}: {e}"));
        let McpTransport::Http(http) = http.transport else {
            panic!("{url} must be an HTTP server");
        };
        assert_eq!(http.headers.to_header_map().len(), 1);
    }
}

#[test]
fn invalid_entries_are_rejected_without_echoing_secrets() {
    let cases = [
        ("command = \"npx\"\nurl = \"https://h/mcp\"\n", "both"),
        ("args = [\"-y\"]\n", "needs `command`"),
        (
            "command = \"npx\"\nheaders = { A = \"b\" }\n",
            "do not apply",
        ),
        (
            "url = \"https://h/mcp\"\nenv = { A = \"b\" }\n",
            "do not apply",
        ),
        ("url = \"http://mcp.example.com/mcp\"\n", "loopback"),
        ("url = \"ftp://h/mcp\"\n", "https"),
        (
            "url = \"https://user:hunter2@h/mcp\"\n",
            "user name or password",
        ),
        (
            "url = \"https://h/mcp\"\nheaders = { Authorization = \"Bearer hunter2\\n\" }\n",
            "Authorization",
        ),
        (
            "url = \"https://h/mcp\"\nheaders = { a = \"1\", A = \"2\" }\n",
            "more than once",
        ),
    ];
    for (entry, expected) in cases {
        let message = parse_server(entry).expect_err(entry);
        assert!(message.contains(expected), "{entry}: {message}");
        assert!(!message.contains("hunter2"), "{entry}: {message}");
    }
}
