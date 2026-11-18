//! Extra command-line arguments handed to a model server, split with POSIX
//! shell quoting but never run through a shell.

use std::fmt::Debug;
use std::marker::PhantomData;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::errors::CustomArgsError;

/// Flags assistd sets on every server it launches, or whose server defaults
/// its clients and supervisor rely on.
const SHARED_MANAGED_FLAGS: &[&str] = &[
    "--host",
    "--port",
    "-ngl",
    "--gpu-layers",
    "--n-gpu-layers",
    "-m",
    "--model",
    "-mu",
    "--model-url",
    "-hf",
    "-hfr",
    "--hf-repo",
    "-hff",
    "--hf-file",
    "--models-dir",
    "--models-preset",
    "--api-prefix",
    "--api-key",
    "--api-key-file",
    "--slots",
    "--no-slots",
    "--ssl-key-file",
    "--ssl-cert-file",
];

/// Flags that let anything holding the server's key read files,
/// run tools, change server state, or send model data off the machine.
const EXPOSING_FLAGS: &[&str] = &[
    "--tools",
    "--tools-runtime",
    "-ag",
    "--agent",
    "--mcp-servers-config",
    "--mcp-servers-json",
    "--ui-mcp-proxy",
    "--webui-mcp-proxy",
    "--connect",
    "--connect-code",
    "--log-prompts-dir",
    "--path",
    "--media-path",
    "--slot-save-path",
    "--props",
    "--rpc",
    "--reuse-port",
];

/// A server assistd launches, naming the flags it manages beyond the shared ones.
pub trait ServerKind: Debug + Clone + Default + PartialEq {
    const MANAGED_FLAGS: &'static [&'static str];
}

/// The server of the model agents run on.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelServer;

impl ServerKind for ModelServer {
    const MANAGED_FLAGS: &'static [&'static str] = &[
        "-c",
        "--ctx-size",
        "--jinja",
        "--no-jinja",
        "--embedding",
        "--embeddings",
        "--slot-save-path",
    ];
}

/// The embedding model server.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EmbeddingServer;

impl ServerKind for EmbeddingServer {
    const MANAGED_FLAGS: &'static [&'static str] = &[
        "-mm",
        "--mmproj",
        "-mmu",
        "--mmproj-url",
        "--mmproj-auto",
        "--no-mmproj",
        "--no-mmproj-auto",
    ];
}

/// Server arguments written as one string, e.g. `--flash-attn on -ot 'exps=CPU'`.
/// No shell runs, so `$VAR`, `~` and globs stay literal; a word starting with
/// `#` begins a comment.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String", bound = "S: ServerKind")]
pub struct CustomArgs<S> {
    source: String,
    args: Vec<String>,
    server: PhantomData<S>,
}

impl<S> CustomArgs<S> {
    /// The arguments in order, one element per shell word.
    pub fn as_slice(&self) -> &[String] {
        &self.args
    }
}

impl<S: ServerKind> FromStr for CustomArgs<S> {
    type Err = CustomArgsError;

    fn from_str(source: &str) -> Result<Self, Self::Err> {
        let args = shlex::split(source).ok_or(CustomArgsError::Unparseable)?;
        args.iter()
            .try_for_each(|arg| check_arg(arg, S::MANAGED_FLAGS))?;
        Ok(Self {
            source: source.to_owned(),
            args,
            server: PhantomData,
        })
    }
}

impl<S: ServerKind> TryFrom<String> for CustomArgs<S> {
    type Error = CustomArgsError;

    fn try_from(source: String) -> Result<Self, Self::Error> {
        source.parse()
    }
}

impl<S> From<CustomArgs<S>> for String {
    fn from(custom_args: CustomArgs<S>) -> Self {
        custom_args.source
    }
}

fn check_arg(arg: &str, server_managed_flags: &[&str]) -> Result<(), CustomArgsError> {
    if arg.chars().any(char::is_control) {
        return Err(CustomArgsError::ControlCharacter(arg.to_owned()));
    }
    let flag = arg.split_once('=').map_or(arg, |(flag, _)| flag);
    if SHARED_MANAGED_FLAGS.contains(&flag) || server_managed_flags.contains(&flag) {
        Err(CustomArgsError::Managed(flag.to_owned()))
    } else if EXPOSING_FLAGS.contains(&flag) {
        Err(CustomArgsError::Exposing(flag.to_owned()))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_for<S: ServerKind>(source: &str) -> Result<Vec<String>, CustomArgsError> {
        source
            .parse::<CustomArgs<S>>()
            .map(|custom_args| custom_args.as_slice().to_vec())
    }

    fn parse(source: &str) -> Result<Vec<String>, CustomArgsError> {
        parse_for::<ModelServer>(source)
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
        for source in [
            "--host 0.0.0.0",
            "--host=0.0.0.0",
            "-c 4096",
            "-hf a/b:Q4",
            "--slot-save-path /tmp/slots",
            "--slot-save-path=/tmp/slots",
        ] {
            assert!(
                matches!(parse(source), Err(CustomArgsError::Managed(_))),
                "{source:?}"
            );
        }
    }

    #[test]
    fn shared_and_exposing_flags_are_rejected_on_the_embedding_server_too() {
        for source in ["--host 0.0.0.0", "--hf-repo a/b:Q4", "--tools all"] {
            assert!(parse_for::<EmbeddingServer>(source).is_err(), "{source:?}");
        }
    }

    #[test]
    fn server_specific_flags_are_managed_only_on_their_server() {
        assert!(matches!(
            parse("--embedding"),
            Err(CustomArgsError::Managed(_))
        ));
        assert!(matches!(
            parse_for::<EmbeddingServer>("--mmproj-auto"),
            Err(CustomArgsError::Managed(_))
        ));
        parse_for::<EmbeddingServer>("--pooling cls -c 8192 --ubatch-size 2048")
            .expect("pooling and tuning are allowed");
        parse("--no-mmproj").expect("the projector is the chat server's to choose");
        assert!(matches!(
            parse_for::<EmbeddingServer>("--slot-save-path /tmp/slots"),
            Err(CustomArgsError::Exposing(_))
        ));
    }

    #[test]
    fn exposing_flags_are_rejected_in_both_spellings() {
        for source in [
            "--tools all",
            "--tools=all",
            "--path /",
            "--rpc 10.0.0.2:50052",
            "--connect",
            "-ag",
            "--mcp-servers-json={}",
            "--log-prompts-dir /tmp/prompts",
        ] {
            assert!(
                matches!(parse(source), Err(CustomArgsError::Exposing(_))),
                "{source:?}"
            );
        }
    }

    #[test]
    fn serializing_returns_the_source_string() {
        let source = "--flash-attn on  -ot 'exps=CPU'";
        let custom_args: CustomArgs<ModelServer> = source.parse().expect("parses");
        assert_eq!(String::from(custom_args), source);
    }
}
