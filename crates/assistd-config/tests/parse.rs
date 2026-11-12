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

[[mcp.servers]]
name = \"b\"
url = \"https://h/mcp\"
headers = { Authorization = \"Bearer x\" }
auth = \"none\"
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
fn non_loopback_server_hosts_do_not_parse() {
    for source in [
        "[model]\nhost = \"0.0.0.0\"\n",
        "[embedding]\nenabled = false\nhost = \"::\"\n",
        "[search]\nhost = \"192.168.1.2\"\n",
    ] {
        let err = toml::from_str::<Config>(source).expect_err("a wildcard host must not parse");
        assert!(err.to_string().contains("loopback"), "{source}: {err}");
    }
}

#[test]
fn relative_writable_paths_do_not_parse() {
    for entry in [".", "docs", "~alice/notes", "./tmp"] {
        let source = format!("[tools.write]\nwritable_paths = [\"/tmp\", \"{entry}\"]\n");
        toml::from_str::<Config>(&source)
            .expect_err("a relative writable_paths entry must not parse");
    }
    toml::from_str::<Config>("[tools.write]\nwritable_paths = []\n")
        .expect_err("an empty writable_paths must not parse");
}

#[test]
fn a_key_outside_its_range_does_not_parse_even_when_its_feature_is_off() {
    for source in [
        "[chat]\ntemperature = 2.5\n",
        "[chat]\npresence_penalty = nan\n",
        "[chat]\nreasoning_effort = \"\"\n",
        "[chat.themes.nord]\n",
        "[voice]\nenabled = false\n[voice.wakeword]\nphrase = \"hey computer 2\"\n",
        "[voice.wakeword]\ncommand_timeout_ms = 499\n",
        "[voice.transcription]\nmodel = \"owner/repo:decoder.onnx\"\n",
        "[voice.synthesis]\nlength_scale = 0.0\n",
        "[voice.synthesis]\nmodel_cache_dir = \"cache\"\n",
        "[embedding]\nenabled = false\ntop_k = 21\n",
        "[embedding]\nmodel = \"owner/repo@main:Q4_K_M\"\n",
        "[search]\nenabled = false\nengines = []\n",
        "[notifications]\nenabled = false\nauto_hide_ms = 100\n",
        "[tasks]\nkept_runs = 0\n",
    ] {
        assert!(
            toml::from_str::<Config>(source).is_err(),
            "{source:?} must not parse"
        );
    }
}

#[test]
fn a_tool_timeout_the_tool_call_deadline_would_cut_short_is_rejected() {
    let validate = |source: &str| -> Result<(), String> {
        let cfg: Config = toml::from_str(source).expect("config must parse");
        cfg.validate().map_err(|err| err.to_string())
    };
    validate("[tools.bash]\ntimeout_secs = 180\n").expect("180 s leaves room for a prompt");
    let bash = validate("[tools.bash]\ntimeout_secs = 181\n").expect_err("181 s is cut short");
    assert!(bash.contains("tools.bash.timeout_secs"), "{bash}");
    let server =
        "[[mcp.servers]]\nname = \"slow\"\ncommand = \"npx\"\nrequest_timeout_secs = 600\n";
    let mcp =
        validate(&format!("[mcp]\nenabled = true\n{server}")).expect_err("600 s is cut short");
    assert!(mcp.contains("mcp.servers[0].request_timeout_secs"), "{mcp}");
    validate(&format!("[mcp]\nenabled = false\n{server}"))
        .expect("a disabled server is not called");
}

#[test]
fn tool_dirs_assistd_prunes_must_not_hold_home_or_its_config() {
    let home = dirs::home_dir().expect("a home directory");
    let config = dirs::config_dir().expect("a config directory");
    let parse = |key: &str, dir: &str| -> Result<(), String> {
        let table = if key == "dir" { "scratch" } else { "output" };
        let cfg: Config = toml::from_str(&format!("[tools.{table}]\n{key} = \"{dir}\"\n"))
            .expect("config must parse");
        cfg.validate().map_err(|err| err.to_string())
    };
    for key in ["dir", "overflow_dir"] {
        for dir in [
            "/".to_string(),
            "~".to_string(),
            "~/".to_string(),
            home.parent()
                .expect("home has a parent")
                .display()
                .to_string(),
            config.display().to_string(),
        ] {
            let err = parse(key, &dir).expect_err(&format!("{key} = {dir:?} must not validate"));
            assert!(
                err.contains("a directory of its own"),
                "{key} = {dir:?}: {err}"
            );
        }
        parse(key, "~/.cache/assistd/tools").expect("a dedicated directory is valid");
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
    let parse = |name: &str| {
        toml::from_str::<Config>(&format!(
            "[mcp]\nenabled = false\n[[mcp.servers]]\nname = \"{name}\"\n\
             command = \"npx\"\n"
        ))
    };
    for name in ["a__b", "a_", "a___b", ""] {
        let err = parse(name)
            .expect_err("a name that can end early in `mcp__<name>__<tool>` must not parse");
        assert!(err.to_string().contains("MCP server name"), "{name}: {err}");
    }
    parse("google_calendar-v2").expect("single inner underscores are unambiguous");
}

fn parse_reporting_unknown_keys(toml_src: &str) -> (Config, Vec<String>) {
    let cfg: Config = toml::from_str(toml_src).expect("config must parse");
    let raw: toml::Table = toml::from_str(toml_src).expect("config must be valid TOML");
    let unknown = cfg.unknown_keys(&raw).expect("config must serialize");
    (cfg, unknown)
}
