//! `memory` subcommands. Everything goes through the daemon, which owns the
//! SQLite writer; the CLI never opens the file itself.

use std::io::{self, Write};

use anyhow::Result;
use assistd_ipc::{EventBody, Request, RequestBody, SearchKind};
use assistd_utils::terminal_text::{escape_controls, escape_controls_single_line};
use clap::{Args, Subcommand};
use uuid::Uuid;

use super::run_one_shot;
use super::search::{self, hit_count, print_hits};

#[derive(Args)]
pub(crate) struct MemoryArgs {
    #[command(subcommand)]
    pub action: MemoryAction,
}

#[derive(Subcommand)]
pub(crate) enum MemoryAction {
    /// Rank the messages of saved sessions by meaning and by the query's exact words:
    /// `assistd search --in sessions`.
    Reminisce {
        query: String,
        /// Cap on number of hits returned.
        #[arg(long, default_value = "5", value_parser = hit_count())]
        limit: u32,
    },
    /// Persist a value under `key`. Overwrites any prior value.
    Save { key: String, value: String },
    /// Read the value previously stored at `key`.
    Load { key: String },
    /// List all stored memories (id, key, value) whose key starts with
    /// `prefix`. Empty prefix lists every memory. Output is one
    /// tab-separated row per memory in lexicographic key order.
    List {
        #[arg(default_value = "")]
        prefix: String,
    },
    /// Forget the memory with row id `id`. Prints `forgot id=N key=...`
    /// on success; exits 2 with `no memory with id=N` on miss.
    Forget { id: i64 },
    /// Remove `key` from the store. No-op if absent.
    Delete { key: String },
    /// Have the daemon retry the memories and chunks it marked as impossible to
    /// embed, then wait until nothing is left to embed. Prints how many rows still
    /// wait; interrupting stops the report, not the embedding.
    Reindex {
        /// Print nothing while waiting. Useful in scripts.
        #[arg(long)]
        quiet: bool,
    },
}

impl MemoryAction {
    fn into_request(self, id: String) -> Request {
        match self {
            MemoryAction::Reminisce { query, limit } => {
                search::request(query, limit, vec![SearchKind::Sessions])
            }
            MemoryAction::Save { key, value } => {
                Request::new(id, RequestBody::MemorySave { key, value })
            }
            MemoryAction::Load { key } => Request::new(id, RequestBody::MemoryLoad { key }),
            MemoryAction::List { prefix } => {
                Request::new(id, RequestBody::MemoryListAll { prefix, limit: 0 })
            }
            MemoryAction::Forget { id: memory_id } => {
                Request::new(id, RequestBody::MemoryForget { memory_id })
            }
            MemoryAction::Delete { key } => Request::new(id, RequestBody::MemoryDelete { key }),
            MemoryAction::Reindex { .. } => Request::new(id, RequestBody::MemoryReindex),
        }
    }
}

pub(crate) async fn run(args: MemoryArgs) -> Result<()> {
    let forget_target = match &args.action {
        MemoryAction::Forget { id } => Some(*id),
        _ => None,
    };
    let reindex_quiet = matches!(&args.action, MemoryAction::Reindex { quiet: true });
    let searching = matches!(&args.action, MemoryAction::Reminisce { .. });

    let req = args.action.into_request(Uuid::new_v4().to_string());
    if searching {
        return print_hits(req).await;
    }
    let mut reindex_reported = false;
    run_one_shot(req, |event| {
        match &event.body {
            EventBody::MemoryValue { key, value } => match value {
                Some(v) => writeln!(
                    io::stdout(),
                    "{}\t{}",
                    escape_controls_single_line(key),
                    escape_controls(v)
                )?,
                None => {
                    writeln!(io::stderr(), "(no value for key {key:?})")?;
                    std::process::exit(2);
                }
            },
            EventBody::MemoryRow {
                memory_id,
                key,
                value,
            } => {
                let key = escape_controls_single_line(key);
                let flattened = value.replace('\n', " ");
                let single_line = escape_controls_single_line(&flattened);
                writeln!(io::stdout(), "{memory_id}\t{key}\t{single_line}")?;
            }
            EventBody::MemoryForgetResult { deleted: true, key } => {
                let id = forget_target.unwrap_or(0);
                match key {
                    Some(k) => writeln!(
                        io::stdout(),
                        "forgot id={id} key={}",
                        escape_controls_single_line(k)
                    )?,
                    None => writeln!(io::stdout(), "forgot id={id}")?,
                }
            }
            EventBody::MemoryForgetResult { deleted: false, .. } => {
                let id = forget_target.unwrap_or(0);
                writeln!(io::stderr(), "no memory with id={id}")?;
                std::process::exit(2);
            }
            EventBody::ReindexProgress {
                message_chunks,
                document_chunks,
                memories,
                accesses,
            } if !reindex_quiet => {
                reindex_reported = true;
                let waiting = [*message_chunks, *document_chunks, *memories, *accesses];
                print_reindex_progress(waiting);
            }
            EventBody::Done if reindex_reported => {
                writeln!(io::stderr())?;
            }
            _ => {}
        }
        Ok(())
    })
    .await
}

/// Overwrite the stderr progress line in place with the rows still waiting: message
/// chunks, document chunks, memories and accesses.
fn print_reindex_progress([message_chunks, document_chunks, memories, accesses]: [u64; 4]) {
    let mut err = io::stderr();
    let _ = write!(
        err,
        "\rreindex: {message_chunks} message chunks, {document_chunks} document chunks, \
         {memories} memories, {accesses} accesses waiting\x1b[K"
    );
    let _ = err.flush();
}
