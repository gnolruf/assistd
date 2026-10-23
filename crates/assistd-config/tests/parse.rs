//! Every section is optional, and a key the schema doesn't know is
//! skipped and reported rather than rejected.

use assistd_config::Config;

#[test]
fn deleted_sections_are_ignored_and_reported() {
    let (cfg, unknown) = parse_reporting_unknown_keys("[remote]\nport = 8384\n\n[timeouts]\n");
    assert_eq!(cfg, Config::default());
    assert_eq!(unknown, ["remote", "timeouts"]);
}

#[test]
fn misspelled_key_is_reported_rather_than_silently_defaulted() {
    let (cfg, unknown) = parse_reporting_unknown_keys("[tools.bash]\ndenylst = []\n");
    assert_eq!(cfg, Config::default());
    assert_eq!(unknown, ["tools.bash.denylst"]);
}

#[test]
fn stale_mcp_server_keys_are_reported_by_index() {
    let toml_src = "\
[[mcp.servers]]
name = \"a\"
command = \"npx\"

[[mcp.servers]]
name = \"b\"
transport = \"stdio\"
command = \"npx\"
";
    let (cfg, unknown) = parse_reporting_unknown_keys(toml_src);
    assert_eq!(cfg.mcp.servers.len(), 2);
    assert_eq!(unknown, ["mcp.servers[1].transport"]);
}

#[test]
fn map_valued_keys_are_not_reported() {
    let toml_src = "\
[[mcp.servers]]
name = \"a\"
command = \"npx\"
env = { API_KEY = \"x\" }
";
    let (_, unknown) = parse_reporting_unknown_keys(toml_src);
    assert!(unknown.is_empty(), "{unknown:?}");
}

#[test]
fn custom_args_with_a_refused_flag_is_a_parse_error() {
    let err =
        toml::from_str::<Config>("[model]\ncustom_args = \"--flash-attn on --host 0.0.0.0\"\n")
            .expect_err("custom_args must not override the bind host");
    let message = err.to_string();
    assert!(message.contains("custom_args"), "{message}");
    assert!(message.contains("--host"), "{message}");
}

#[test]
fn written_default_round_trips_with_no_unknown_keys() {
    let serialized = toml::to_string_pretty(&Config::default()).expect("serialize default");
    let (back, unknown) = parse_reporting_unknown_keys(&serialized);
    assert_eq!(back, Config::default());
    assert!(unknown.is_empty(), "{unknown:?}");
}

#[test]
fn defaults_validate() {
    Config::default()
        .validate()
        .expect("the code defaults must be a valid configuration");
}

#[test]
fn non_loopback_server_hosts_are_rejected() {
    let cfg: Config =
        toml::from_str("[model]\nhost = \"0.0.0.0\"\n[embedding]\nenabled = true\nhost = \"::\"\n")
            .expect("config must parse");
    let err = cfg
        .validate()
        .expect_err("wildcard hosts must not validate");
    let message = err.to_string();
    assert!(message.contains("model.host"), "{message}");
    assert!(message.contains("embedding.host"), "{message}");
}

#[test]
fn relative_writable_paths_are_rejected() {
    for entry in [".", "docs", "~alice/notes", "./tmp"] {
        let cfg: Config = toml::from_str(&format!(
            "[tools.write]\nwritable_paths = [\"/tmp\", \"{entry}\"]\n"
        ))
        .expect("config must parse");
        let err = cfg
            .validate()
            .expect_err("a relative writable_paths entry must not validate");
        let message = err.to_string();
        assert!(
            message.contains("tools.write.writable_paths"),
            "{entry}: {message}"
        );
    }
}

#[test]
fn home_relative_and_absolute_writable_paths_validate() {
    let cfg: Config =
        toml::from_str("[tools.write]\nwritable_paths = [\"~\", \"~/notes\", \"/tmp\"]\n")
            .expect("config must parse");
    cfg.validate()
        .expect("tilde and absolute entries are valid");
}

#[test]
fn history_and_response_must_fit_the_context_together() {
    let parse = |response: u32| -> Config {
        toml::from_str(&format!(
            "[model]\ncontext_length = 65536\n\
             [chat]\nmax_history_tokens = 32768\nmax_response_tokens = {response}\n"
        ))
        .expect("config must parse")
    };
    parse(24576).validate().expect("57344 fits in 58982");
    let err = parse(32768)
        .validate()
        .expect_err("65536 must not fit in 58982");
    let message = err.to_string();
    assert!(message.contains("chat.max_response_tokens"), "{message}");
}

#[test]
fn mcp_server_names_that_could_share_a_tool_name_are_rejected() {
    for name in ["a__b", "a_", "a___b"] {
        let cfg: Config = toml::from_str(&format!(
            "[mcp]\nenabled = true\n[[mcp.servers]]\nname = \"{name}\"\n\
             command = \"npx\"\n"
        ))
        .expect("config must parse");
        let err = cfg
            .validate()
            .expect_err("a name that can end early in `mcp__<name>__<tool>` must not validate");
        let message = err.to_string();
        assert!(message.contains("mcp.servers[0].name"), "{name}: {message}");
    }
    let cfg: Config = toml::from_str(
        "[mcp]\nenabled = true\n[[mcp.servers]]\nname = \"google_calendar-v2\"\n\
         command = \"npx\"\n",
    )
    .expect("config must parse");
    cfg.validate()
        .expect("single inner underscores are unambiguous");
}

fn parse_reporting_unknown_keys(toml_src: &str) -> (Config, Vec<String>) {
    let cfg: Config = toml::from_str(toml_src).expect("config must parse");
    let raw: toml::Table = toml::from_str(toml_src).expect("config must be valid TOML");
    let unknown = cfg.unknown_keys(&raw).expect("config must serialize");
    (cfg, unknown)
}
