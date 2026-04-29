//! Client for `assistd memory <action>` subcommands.
//!
//! Mirrors the shape of [`crate::presence::run`]: opens the daemon's
//! Unix socket, sends one [`Request::Memory*`], prints incoming events
//! to stdout, and exits on `Event::Done` / `Event::Error`. All routing
//! goes through the daemon — there's no direct-SQLite fallback (the
//! daemon owns the writer, and a CLI grabbing the file lock while it's
//! mid-write would risk corruption).

use anyhow::{Context, Result};
use assistd_ipc::{Event, Request, socket_path};
use clap::{Args, Subcommand, ValueEnum};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use uuid::Uuid;

#[derive(Args)]
pub struct MemoryArgs {
    #[command(subcommand)]
    pub action: MemoryAction,
}

#[derive(Subcommand)]
pub enum MemoryAction {
    /// Search persisted conversation content. Two modes:
    /// - `semantic` (default): cosine-similarity search over embedded
    ///   chunks. Robust to paraphrase. Requires the embedding
    ///   subsystem.
    /// - `fts`: SQLite FTS5 full-text search. Supports phrase queries
    ///   (`"foo bar"`), boolean ops (`foo AND bar`), and prefix matches
    ///   (`foo*`).
    Search {
        /// Query string. Interpretation depends on `--mode`.
        query: String,
        /// Cap on number of hits returned.
        #[arg(long, default_value = "5")]
        limit: u32,
        /// Search mode. Defaults to `semantic` (the most-relevant
        /// ranking; pass `--mode fts` for keyword/exact-phrase work).
        #[arg(long, value_enum, default_value_t = SearchMode::Semantic)]
        mode: SearchMode,
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
}

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum SearchMode {
    /// SQLite FTS5 keyword search. Fast, exact, deterministic.
    Fts,
    /// Cosine-similarity over embeddings. Paraphrase-tolerant.
    Semantic,
}

impl MemoryAction {
    fn into_request(self, id: String) -> Request {
        match self {
            MemoryAction::Search {
                query,
                limit,
                mode: SearchMode::Fts,
            } => Request::MemorySearch { id, query, limit },
            MemoryAction::Search {
                query,
                limit,
                mode: SearchMode::Semantic,
            } => Request::MemorySemanticSearch { id, query, limit },
            MemoryAction::Save { key, value } => Request::MemorySave { id, key, value },
            MemoryAction::Load { key } => Request::MemoryLoad { id, key },
            MemoryAction::List { prefix } => Request::MemoryListAll {
                id,
                prefix,
                limit: 0,
            },
            MemoryAction::Forget { id: memory_id } => Request::MemoryForget { id, memory_id },
            MemoryAction::Delete { key } => Request::MemoryDelete { id, key },
        }
    }
}

pub async fn run(args: MemoryArgs) -> Result<()> {
    let path = socket_path();
    let stream = UnixStream::connect(&path).await.with_context(|| {
        format!(
            "assistd daemon is not running (could not connect to {})",
            path.display()
        )
    })?;

    // Capture the forget target id before `into_request` consumes the
    // action — needed so the `MemoryForgetResult` printer can echo it
    // back on the miss path (`no memory with id=N`), where the daemon
    // doesn't carry a key.
    let forget_target = match &args.action {
        MemoryAction::Forget { id } => Some(*id),
        _ => None,
    };

    let (read_half, mut write_half) = stream.into_split();
    let req = args.action.into_request(Uuid::new_v4().to_string());
    let mut body = serde_json::to_string(&req)?;
    body.push('\n');
    write_half.write_all(body.as_bytes()).await?;
    write_half.shutdown().await?;

    let mut reader = BufReader::new(read_half);
    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line).await?;
        if n == 0 {
            anyhow::bail!("daemon closed the connection without sending a terminal event");
        }
        let event: Event = serde_json::from_str(line.trim())
            .with_context(|| format!("invalid JSON from daemon: {}", line.trim()))?;

        match event {
            Event::MemoryHit {
                conversation_id,
                session_id,
                timestamp,
                role,
                snippet,
                ..
            } => {
                // One line per hit: timestamp, role, conversation row
                // id, snippet (with FTS5 `<mark>` highlights). Short
                // session-id prefix gives users a stable handle for
                // follow-up queries without wasting columns.
                let session_short = session_id.chars().take(8).collect::<String>();
                println!(
                    "{timestamp}  {role:9}  conv={conversation_id:<6}  sess={session_short}  {snippet}"
                );
            }
            Event::SemanticHit {
                conversation_id,
                session_id,
                timestamp,
                role,
                content,
                similarity,
                ..
            } => {
                // Same column layout as MemoryHit but with similarity
                // score instead of `<mark>` highlighting. `content` is
                // the full message text (single line — collapse newlines
                // so columns stay aligned).
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
            Event::Done { .. } => return Ok(()),
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
