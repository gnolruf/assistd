//! `memory` subcommands. Everything goes through the daemon; the CLI
//! never opens the SQLite file itself, because the daemon owns the
//! writer.

use anyhow::Result;
use assistd_ipc::{Event, ReindexKind, Request};
use clap::{Args, Subcommand};
use uuid::Uuid;

use crate::ipc_helper::run_one_shot;

#[derive(Args)]
pub struct MemoryArgs {
    #[command(subcommand)]
    pub action: MemoryAction,
}

#[derive(Subcommand)]
pub enum MemoryAction {
    /// Semantic search over persisted conversation content. Embeds the
    /// query and ranks past messages by cosine similarity, so
    /// paraphrased phrasings still hit. Requires the embedding
    /// subsystem; with embeddings disabled the daemon emits zero hits
    /// and a clean Done.
    Reminisce {
        /// Natural-language query. The daemon embeds this and finds
        /// the top-`limit` most semantically similar past messages.
        query: String,
        /// Cap on number of hits returned.
        #[arg(long, default_value = "5")]
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
    /// Re-embed every memory and conversation chunk that has no
    /// embedding under the daemon's currently-configured model.
    /// Recovers from a model swap or from runs where the embedding
    /// subsystem was unavailable. Prints one progress line per kind
    /// (chunks, memories) updated as each item completes.
    Reindex {
        /// Suppress the per-item progress lines; print only the final
        /// summary. Useful in scripts.
        #[arg(long)]
        quiet: bool,
    },
}

impl MemoryAction {
    fn into_request(self, id: String) -> Request {
        match self {
            MemoryAction::Reminisce { query, limit } => {
                Request::MemorySemanticSearch { id, query, limit }
            }
            MemoryAction::Save { key, value } => Request::MemorySave { id, key, value },
            MemoryAction::Load { key } => Request::MemoryLoad { id, key },
            MemoryAction::List { prefix } => Request::MemoryListAll {
                id,
                prefix,
                limit: 0,
            },
            MemoryAction::Forget { id: memory_id } => Request::MemoryForget { id, memory_id },
            MemoryAction::Delete { key } => Request::MemoryDelete { id, key },
            MemoryAction::Reindex { .. } => Request::MemoryReindex { id },
        }
    }
}

pub async fn run(args: MemoryArgs) -> Result<()> {
    let forget_target = match &args.action {
        MemoryAction::Forget { id } => Some(*id),
        _ => None,
    };
    let reindex_quiet = matches!(&args.action, MemoryAction::Reindex { quiet: true });

    let req = args.action.into_request(Uuid::new_v4().to_string());
    let mut last_reindex_kind: Option<ReindexKind> = None;
    run_one_shot(req, |event| {
        match event {
            Event::SemanticHit {
                conversation_id,
                session_id,
                timestamp,
                role,
                content,
                similarity,
                ..
            } => {
                let session_short = session_id.chars().take(8).collect::<String>();
                let single_line = content.replace('\n', " ");
                println!(
                    "{timestamp}  {role:9}  conv={conversation_id:<6}  sess={session_short}  sim={similarity:.2}  {single_line}"
                );
            }
            Event::MemoryValue { key, value, .. } => match value {
                Some(v) => println!("{key}\t{v}"),
                None => {
                    eprintln!("(no value for key {key:?})");
                    std::process::exit(2);
                }
            },
            Event::MemoryKeys { keys, .. } => {
                for k in keys {
                    println!("{k}");
                }
            }
            Event::MemoryRow {
                memory_id,
                key,
                value,
                ..
            } => {
                let single_line = value.replace('\n', " ");
                println!("{memory_id}\t{key}\t{single_line}");
            }
            Event::MemoryForgetResult { deleted: true, key, .. } => {
                let id = forget_target.unwrap_or(0);
                match key {
                    Some(k) => println!("forgot id={id} key={k}"),
                    None => println!("forgot id={id}"),
                }
            }
            Event::MemoryForgetResult { deleted: false, .. } => {
                let id = forget_target.unwrap_or(0);
                eprintln!("no memory with id={id}");
                std::process::exit(2);
            }
            Event::ReindexProgress {
                kind, done, total, ..
            } if !reindex_quiet => {
                use std::io::Write;
                let mut err = std::io::stderr();
                let kind_changed = last_reindex_kind != Some(*kind);
                if kind_changed && last_reindex_kind.is_some() {
                    let _ = writeln!(err);
                }
                last_reindex_kind = Some(*kind);
                let _ = write!(err, "\rreindex {kind}: {done}/{total}");
                let _ = err.flush();
            }
            Event::Done { .. } if last_reindex_kind.is_some() && !reindex_quiet => {
                eprintln!();
            }
            _ => {}
        }
        Ok(())
    })
    .await
}
