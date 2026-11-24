//! `search`: rank past sessions, saved memories, indexed documents and the files and
//! pages sessions touched through the daemon, and print the hits as the model's `search` command lists them.

use std::io::{self, Write};

use anyhow::Result;
use assistd_ipc::{EventBody, Request, RequestBody, SearchKind};
use assistd_tools::commands::search::{MAX_HITS, hit_line};
use assistd_utils::terminal_text::escape_controls_single_line;
use clap::Args;
use clap::builder::RangedI64ValueParser;
use uuid::Uuid;

use super::run_one_shot;

#[derive(Args)]
pub(crate) struct SearchArgs {
    /// Cap on the hits listed; the daemon's `embedding.top_k` when omitted.
    #[arg(short = 'n', long, value_parser = hit_count())]
    limit: Option<u32>,
    /// Only these kinds, separated by commas: sessions, memories, documents, accesses
    /// (the files and pages sessions touched). Every kind turned on in the config file
    /// when omitted.
    #[arg(long = "in", value_name = "KINDS", value_delimiter = ',', value_parser = parse_kind)]
    kinds: Vec<SearchKind>,
    /// The words of the query; they are joined, so quoting is not needed.
    #[arg(required = true)]
    query: Vec<String>,
}

pub(crate) async fn run(args: SearchArgs) -> Result<()> {
    let limit = args.limit.unwrap_or_default();
    print_hits(request(args.query.join(" "), limit, args.kinds)).await
}

/// A search for `query` over `kinds`, every kind turned on when there is none, for
/// `limit` hits or the daemon's default when it is zero.
pub(crate) fn request(query: String, limit: u32, kinds: Vec<SearchKind>) -> Request {
    let body = RequestBody::Search {
        query,
        limit,
        kinds,
    };
    Request::new(Uuid::new_v4().to_string(), body)
}

/// Send the search `request` and print one tab-separated line per hit, best first:
/// kind, place, similarity, other holders when there are any, passage. Exits 1 when
/// nothing matched.
pub(crate) async fn print_hits(request: Request) -> Result<()> {
    let mut any_hit = false;
    run_one_shot(request, |event| {
        if let EventBody::SearchHit {
            place,
            content,
            similarity,
            held_elsewhere,
        } = &event.body
        {
            any_hit = true;
            let line = hit_line(place, *similarity, *held_elsewhere, content);
            writeln!(io::stdout(), "{}", terminal_fields(&line))?;
        }
        Ok(())
    })
    .await?;
    if !any_hit {
        writeln!(
            io::stderr(),
            "nothing matched: text still waiting for a vector is found by its exact words \
             only, and `assistd documents status` shows what is indexed"
        )?;
        std::process::exit(1);
    }
    Ok(())
}

/// The value parser of a hit count, from one to the most a search lists.
pub(crate) fn hit_count() -> RangedI64ValueParser<u32> {
    clap::value_parser!(u32).range(1..=i64::from(MAX_HITS))
}

fn parse_kind(name: &str) -> Result<SearchKind, String> {
    SearchKind::named(name)
        .ok_or_else(|| format!("'{name}' is not one of sessions, memories, documents, accesses"))
}

/// The tab-separated fields of `line`, each made safe to print on a terminal.
fn terminal_fields(line: &str) -> String {
    let fields: Vec<_> = line
        .trim_end_matches('\n')
        .split('\t')
        .map(escape_controls_single_line)
        .collect();
    fields.join("\t")
}
