use super::*;

fn parse(source: &str) -> Result<Vec<String>, CustomArgsError> {
    source
        .parse::<CustomArgs>()
        .map(|custom_args| custom_args.as_slice().to_vec())
}

#[test]
fn empty_string_yields_no_args() {
    assert_eq!(parse("").expect("empty parses"), Vec::<String>::new());
    assert_eq!(
        parse("  \n\t ").expect("blank parses"),
        Vec::<String>::new()
    );
}

#[test]
fn words_split_on_whitespace_including_newlines() {
    assert_eq!(
        parse("--flash-attn on\n--threads 8").expect("parses"),
        ["--flash-attn", "on", "--threads", "8"]
    );
}

#[test]
fn quotes_keep_a_regex_in_one_word() {
    assert_eq!(
        parse(r"-ot '\.ffn_(up|down|gate)_exps\.=CPU'").expect("parses"),
        ["-ot", r"\.ffn_(up|down|gate)_exps\.=CPU"]
    );
}

#[test]
fn shell_syntax_stays_literal() {
    assert_eq!(
        parse("--log-file $HOME/x;rm -rf ~ `id` $(id) *").expect("parses"),
        ["--log-file", "$HOME/x;rm", "-rf", "~", "`id`", "$(id)", "*"]
    );
}

#[test]
fn unterminated_quote_is_rejected() {
    assert!(matches!(
        parse("-ot 'exps=CPU"),
        Err(CustomArgsError::Unparseable)
    ));
}

#[test]
fn control_characters_inside_quotes_are_rejected() {
    for source in [
        "--alias 'a\nb'",
        "--alias \"a\u{1b}[2Jb\"",
        "--alias 'a\tb'",
    ] {
        assert!(
            matches!(parse(source), Err(CustomArgsError::ControlCharacter(_))),
            "{source:?}"
        );
    }
}

#[test]
fn managed_flags_are_rejected_in_both_spellings() {
    for source in ["--host 0.0.0.0", "--host=0.0.0.0", "-c 4096", "-hf a/b:Q4"] {
        assert!(
            matches!(parse(source), Err(CustomArgsError::Managed(_))),
            "{source:?}"
        );
    }
}

#[test]
fn exposing_flags_are_rejected_in_both_spellings() {
    for source in [
        "--tools all",
        "--tools=all",
        "--path /",
        "--rpc 10.0.0.2:50052",
    ] {
        assert!(
            matches!(parse(source), Err(CustomArgsError::Exposing(_))),
            "{source:?}"
        );
    }
}

#[test]
fn a_rejected_flag_hidden_after_valid_args_is_still_found() {
    assert!(matches!(
        parse("--flash-attn on --threads 8 --props"),
        Err(CustomArgsError::Exposing(flag)) if flag == "--props"
    ));
}

#[test]
fn serializing_returns_the_source_string() {
    let source = "--flash-attn on  -ot 'exps=CPU'";
    let custom_args: CustomArgs = source.parse().expect("parses");
    assert_eq!(String::from(custom_args), source);
}
