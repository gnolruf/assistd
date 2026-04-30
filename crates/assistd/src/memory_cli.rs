//! Client for `assistd memory <action>` subcommands.
//!
//! Mirrors the shape of [`crate::presence::run`]: opens the daemon's
//! Unix socket, sends one [`Request::Memory*`], prints incoming events
//! to stdout, and exits on `Event::Done` / `Event::Error`. All routing
//! goes through the daemon — there's no direct-SQLite fallback (the
//! daemon owns the writer, and a CLI grabbing the file lock while it's
//! mid-write would risk corruption).

use anyhow::Result;
use assistd_ipc::{Event, IpcClient, Request};
use clap::{Args, Subcommand};
use uuid::Uuid;

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
    // Capture the forget target id before `into_request` consumes the
    // action — needed so the `MemoryForgetResult` printer can echo it
    // back on the miss path (`no memory with id=N`), where the daemon
    // doesn't carry a key.
    let forget_target = match &args.action {
        MemoryAction::Forget { id } => Some(*id),
        _ => None,
    };
    // Same idea for the reindex progress printer: capture `--quiet`
    // before `into_request` moves the action.
    let reindex_quiet = matches!(&args.action, MemoryAction::Reindex { quiet: true });

    let req = args.action.into_request(Uuid::new_v4().to_string());
    let mut stream = IpcClient::new()
        .one_shot(req)
        .await
        .map_err(crate::ipc_helper::map_not_reachable)?;

    // Track the last reindex kind so a kind change emits a newline
    // before the new kind's progress line — keeps the chunks → memories
    // transition from clobbering the chunks total under `\r` rewrites.
    let mut last_reindex_kind: Option<String> = None;
    loop {
        let event = match stream.next_event().await? {
            Some(ev) => ev,
            None => anyhow::bail!("daemon closed the connection without sending a terminal event"),
        };

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
                // One line per hit: timestamp, role, conversation row
                // id, similarity score, full message content. Short
                // session-id prefix gives users a stable handle for
                // follow-up queries without wasting columns. `content`
                // is the full message text (single line — collapse
                // newlines so columns stay aligned).
                let session_short = session_id.chars().take(8).collect::<String>();
                let single_line = content.replace('\n', " ");
                println!(
                    "{timestamp}  {role:9}  conv={conversation_id:<6}  sess={session_short}  sim={:.2}  {single_line}",
                    similarity
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
                // Tab-separated id, key, value. Collapse newlines in
                // value so a multi-line memory doesn't break the row
                // boundary in shell pipelines.
                let single_line = value.replace('\n', " ");
                println!("{memory_id}\t{key}\t{single_line}");
            }
            Event::MemoryForgetResult {
                deleted: true,
                key: Some(k),
                ..
            } => {
                let id = forget_target.unwrap_or(0);
                println!("forgot id={id} key={k}");
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
                let kind_changed = last_reindex_kind.as_deref() != Some(kind.as_str());
                if kind_changed && last_reindex_kind.is_some() {
                    let _ = writeln!(err);
                }
                last_reindex_kind = Some(kind.clone());
                // `\r` rewrites the same line for successive ticks
                // within one kind; the newline above moves to a
                // fresh line on transitions.
                let _ = write!(err, "\rreindex {kind}: {done}/{total}");
                let _ = err.flush();
            }
            // `MemoryForgetResult { deleted: true, key: None }` is not
            // produced by the daemon (a delete-by-id always has a key
            // when it hits a row), but the type allows it; treat it as
            // a successful no-op so we don't trip the missing-id exit.
            Event::MemoryForgetResult {
                deleted: true,
                key: None,
                ..
            } => {
                let id = forget_target.unwrap_or(0);
                println!("forgot id={id}");
            }
            Event::Done { .. } => {
                if last_reindex_kind.is_some() && !reindex_quiet {
                    eprintln!();
                }
                return Ok(());
            }
            Event::Error { message, .. } => {
                eprintln!("daemon error: {message}");
                std::process::exit(1);
            }
            // Other event types are not expected for memory commands —
            // ignore defensively in case a future protocol addition
            // funnels them onto this stream.
            _ => {}
        }
    }
}
