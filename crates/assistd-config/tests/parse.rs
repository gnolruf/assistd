//! Every section is optional, and a key the schema doesn't know is
//! skipped and reported rather than rejected.

use assistd_config::{Config, fixtures};

#[test]
fn empty_toml_yields_the_code_defaults() {
    let cfg: Config = toml::from_str("").expect("an empty config must parse");
    assert_eq!(cfg, Config::default());
}

#[test]
fn every_section_may_be_declared_empty() {
    let toml_src = "\
[model]
[llama_server]
[chat]
[voice]
[voice.transcription]
[voice.continuous]
[voice.synthesis]
[compositor]
[sleep]
[presence]
[daemon]
[tools]
[tools.output]
[tools.bash]
[tools.write]
[tools.screenshot]
[memory]
[embedding]
[mcp]
[tray]
[tray.popup]
[tray.popup.wake_on]
";
    let cfg: Config = toml::from_str(toml_src).expect("empty sections must parse");
    assert_eq!(cfg, Config::default());
}

#[test]
fn deleted_sections_are_ignored_and_reported() {
    let (cfg, unknown) = parse_reporting_unknown_keys("[remote]\nport = 8384\n\n[timeouts]\n");
    assert_eq!(cfg, Config::default());
    assert_eq!(unknown, ["remote", "timeouts"]);
}

#[test]
fn deleted_keys_in_known_sections_are_reported_with_their_path() {
    let (_, unknown) = parse_reporting_unknown_keys(
        "[chat]\nmax_summary_tokens = 1200\n\n[sleep]\nsuspend = false\n",
    );
    assert_eq!(unknown, ["chat.max_summary_tokens", "sleep.suspend"]);
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
transport = \"stdio\"
command = \"npx\"

[[mcp.servers]]
name = \"b\"
transport = \"sse\"
url = \"https://example.com/sse\"
sse_read_timeout_secs = 300
";
    let (cfg, unknown) = parse_reporting_unknown_keys(toml_src);
    assert_eq!(cfg.mcp.servers.len(), 2);
    assert_eq!(unknown, ["mcp.servers[1].sse_read_timeout_secs"]);
}

#[test]
fn map_valued_keys_are_not_reported() {
    let toml_src = "\
[[mcp.servers]]
name = \"a\"
transport = \"stdio\"
command = \"npx\"
env = { API_KEY = \"x\" }
";
    let (_, unknown) = parse_reporting_unknown_keys(toml_src);
    assert!(unknown.is_empty(), "{unknown:?}");
}

#[test]
fn a_wrong_type_is_still_a_parse_error() {
    let err = toml::from_str::<Config>("[chat]\ntemperature = \"hot\"\n")
        .expect_err("a mistyped known key must not parse");
    assert!(err.to_string().contains("temperature"), "{err}");
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
fn minimal_fixture_parses_and_validates() {
    fixtures::minimal()
        .validate()
        .expect("minimal fixture must validate");
}

fn parse_reporting_unknown_keys(toml_src: &str) -> (Config, Vec<String>) {
    let cfg: Config = toml::from_str(toml_src).expect("config must parse");
    let raw: toml::Table = toml::from_str(toml_src).expect("config must be valid TOML");
    let unknown = cfg.unknown_keys(&raw).expect("config must serialize");
    (cfg, unknown)
}
