//! Every section is optional, and anything not in the schema is a hard
//! error.

use assistd_config::Config;

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
fn deleted_sections_are_rejected() {
    for section in ["[remote]", "[timeouts]"] {
        let err = toml::from_str::<Config>(section)
            .err()
            .unwrap_or_else(|| panic!("{section} must not parse"));
        assert!(
            err.to_string().contains(section.trim_matches(['[', ']'])),
            "{err}"
        );
    }
}

#[test]
fn unknown_section_is_rejected() {
    let err = toml::from_str::<Config>("[agent]\nmax_iterations = 50\n")
        .expect_err("an unknown section must not parse");
    let msg = err.to_string();
    assert!(msg.contains("agent"), "{msg}");
}

#[test]
fn misspelled_key_is_rejected_rather_than_defaulted() {
    let err = toml::from_str::<Config>("[tools.bash]\ndenylst = []\n")
        .expect_err("a misspelled key must not parse");
    let msg = err.to_string();
    assert!(msg.contains("denylst"), "{msg}");
}

#[test]
fn unknown_key_error_points_at_the_offending_line() {
    let toml_src = "\
[model]
name = \"test/model-GGUF:Q4_K_M\"

[chat]
temperture = 0.7
";
    let err = toml::from_str::<Config>(toml_src).expect_err("a typo must not parse");
    let span = err.span().expect("toml errors carry a span");
    let highlighted = &toml_src[span.clone()];
    assert!(
        highlighted.contains("temperture"),
        "span {span:?} should cover the typo, got {highlighted:?}"
    );
}

#[test]
fn written_default_round_trips_under_strict_parsing() {
    let serialized = toml::to_string_pretty(&Config::default()).expect("serialize default");
    let back: Config = toml::from_str(&serialized).expect("written default must re-parse");
    assert_eq!(back, Config::default());
}

#[test]
fn defaults_validate() {
    Config::default()
        .validate()
        .expect("the code defaults must be a valid configuration");
}
