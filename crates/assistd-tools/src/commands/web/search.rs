//! `web search`: query the supervised SearXNG instance and print the hits
//! as TSV.

use std::fmt::{Display, Write};
use std::num::NonZeroU32;

use assistd_search::{SearchError, SearchQuery, SearchResults, SearxngClient};

use crate::command::{CommandOutput, Hint, error_line};

use super::NAME;

const DEFAULT_HITS: usize = 8;

const MAX_HITS: usize = 20;

const USAGE: &str = "web search [-n N] [--lang <tag>] <query...>";

const FETCH_KNOWN: &str = "web fetch <URL> with a URL you already know";

const HELP: &str = "usage: web search [-n N] [--lang <tag>] <query...>\n\
    \n\
    Search the web. The words after the \
    options form one query, so quoting is not needed: \
    web search sway 1.11 release notes\n\
    \n\
    Prints one hit per line as TSV: rank, title, url, snippet. Instant \
    answers, when an engine gives one, come first as `answer<TAB>text`. \
    Pick the most relevant url and read it with `web fetch <URL>`.\n\
    \n\
    Options:\n  \
      -n N              at most N hits (default 8, max 20); -n 5 suits simple questions\n  \
      --lang <tag>      result language, e.g. en-US\n\
    \n\
    No results exits 0 with empty stdout and a note on stderr. Exit 1 when \
    search is disabled, the search server is not running, or the search \
    fails; exit 2 on usage errors.\n";

#[derive(Debug, PartialEq, Eq)]
struct SearchArgs<'a> {
    limit: usize,
    language: Option<&'a str>,
    text: String,
}

impl<'a> SearchArgs<'a> {
    fn parse(args: &'a [String]) -> Result<Self, String> {
        let mut limit = DEFAULT_HITS;
        let mut language = None;
        let mut rest = args;
        loop {
            match rest {
                [flag, value, tail @ ..] if flag == "-n" => {
                    limit = parse_limit(value)?;
                    rest = tail;
                }
                [flag, value, tail @ ..] if flag == "--lang" => {
                    language = Some(value.as_str());
                    rest = tail;
                }
                [flag] if matches!(flag.as_str(), "-n" | "--lang") => {
                    return Err(format!("{flag} needs a value"));
                }
                [flag, tail @ ..] if flag == "--" => {
                    rest = tail;
                    break;
                }
                _ => break,
            }
        }
        if rest.is_empty() {
            return Err("no query words".to_string());
        }
        Ok(Self {
            limit,
            language,
            text: rest.join(" "),
        })
    }
}

pub(super) async fn run(client: Option<&SearxngClient>, args: &[String]) -> CommandOutput {
    if args.is_empty() {
        return CommandOutput::usage(HELP.to_string());
    }
    let Some(client) = client else {
        return search_failed("search is disabled ([search] enabled = false)", FETCH_KNOWN);
    };
    let args = match SearchArgs::parse(args) {
        Ok(args) => args,
        Err(what) => {
            return CommandOutput::usage_error(NAME, format_args!("search: {what}"), USAGE);
        }
    };
    let query = SearchQuery {
        text: &args.text,
        language: args.language,
        page: NonZeroU32::MIN,
    };
    match client.search(&query).await {
        Ok(results) => listed(&results, args.limit),
        Err(SearchError::NotServing) => {
            search_failed("the search server is not running yet", "again in a moment")
        }
        Err(SearchError::Unavailable(reason)) => {
            search_failed(format_args!("search is unavailable: {reason}"), FETCH_KNOWN)
        }
        Err(e) => search_failed(e, "re-running the search"),
    }
}

fn parse_limit(raw: &str) -> Result<usize, String> {
    raw.parse()
        .ok()
        .filter(|limit| (1..=MAX_HITS).contains(limit))
        .ok_or_else(|| format!("-n takes a count from 1 to {MAX_HITS}, got '{raw}'"))
}

/// Answers, then the first `limit` hits, as TSV on stdout. Engines that did
/// not answer are logged, and reach the model only to explain an empty result.
fn listed(results: &SearchResults, limit: usize) -> CommandOutput {
    let stdout = tsv(results, limit);
    let unresponsive = unresponsive_engines(results);
    if let Some(engines) = &unresponsive {
        tracing::warn!("web search: engines that did not answer: {engines}");
    }
    let stderr = if stdout.is_empty() {
        let why = unresponsive
            .map(|engines| format!("; engines that did not answer: {engines}"))
            .unwrap_or_default();
        format!(
            "[note] {NAME}: no results{why}. {}: fewer or different words\n",
            Hint::Try
        )
    } else {
        String::new()
    };
    CommandOutput {
        stdout: stdout.into_bytes(),
        stderr: stderr.into_bytes(),
        ..CommandOutput::default()
    }
}

fn tsv(results: &SearchResults, limit: usize) -> String {
    let mut out = String::new();
    for answer in &results.answers {
        let answer = answer.split_whitespace().collect::<Vec<_>>().join(" ");
        let _ = writeln!(out, "answer\t{answer}");
    }
    for (rank, hit) in results.results.iter().take(limit).enumerate() {
        let _ = writeln!(
            out,
            "{}\t{}\t{}\t{}",
            rank + 1,
            hit.title,
            hit.url,
            hit.content
        );
    }
    out
}

fn unresponsive_engines(results: &SearchResults) -> Option<String> {
    (!results.unresponsive_engines.is_empty()).then(|| {
        results
            .unresponsive_engines
            .iter()
            .map(|(engine, reason)| format!("{engine} ({reason})"))
            .collect::<Vec<_>>()
            .join(", ")
    })
}

fn search_failed(what: impl Display, recovery: &str) -> CommandOutput {
    CommandOutput::failed(1, error_line(NAME, what, Hint::Try, recovery).into_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn results_print_as_one_tsv_line_each() {
        let results: SearchResults = serde_json::from_str(
            r#"{
                "answers": ["sway 1.11\nreleased"],
                "results": [
                    {"url": "https://a.example/", "title": "A\ttitle", "content": "line one\n\tline two"},
                    {"url": "https://b.example/", "title": "B", "content": ""},
                    {"url": "https://c.example/", "title": "C", "content": "dropped"}
                ],
                "unresponsive_engines": [["brave", "too many requests"]]
            }"#,
        )
        .expect("results decode");
        let out = listed(&results, 2);
        assert_eq!(out.exit_code, 0);
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            "answer\tsway 1.11 released\n\
             1\tA title\thttps://a.example/\tline one line two\n\
             2\tB\thttps://b.example/\t\n"
        );
        assert!(out.stderr.is_empty(), "{:?}", out.stderr);
    }
}
