//! `documents` subcommands: report the document index, start a pass over the indexed
//! folders and follow it, or search the indexed documents, all through the daemon.

use std::io::{self, Write};

use anyhow::Result;
use assistd_ipc::{
    ComponentReadiness, DocumentPass, DocumentPassCounts, DocumentPassEnd, DocumentRootStatus,
    EventBody, Request, RequestBody, SkippedFiles,
};
use assistd_utils::terminal_text::escape_controls_single_line;
use assistd_utils::text::{collapse_whitespace, similarity_percent};
use chrono::{DateTime, Local, Utc};
use clap::{Args, Subcommand};
use uuid::Uuid;

use super::run_one_shot;

const PASS_TIME_FORMAT: &str = "%Y-%m-%d %H:%M %Z";

const INDEXED_DATE_FORMAT: &str = "%Y-%m-%d";

#[derive(Args)]
pub(crate) struct DocumentsArgs {
    #[command(subcommand)]
    pub action: DocumentsAction,
}

#[derive(Subcommand)]
pub(crate) enum DocumentsAction {
    /// Whether indexing and embedding run and how full the index is, then for each
    /// indexed folder: whether it can be read and is watched, what is stored from it and
    /// still waiting for a vector, and what the last pass over it did and skipped.
    Status,
    /// Walk every indexed folder now, indexing changed files and dropping removed
    /// ones. Prints a progress line as files are handled, then what the pass over
    /// each folder did.
    Reindex {
        /// Suppress the progress lines; print only what each pass did.
        #[arg(long)]
        quiet: bool,
    },
    /// Rank the indexed documents by meaning and by the query's exact words, as the
    /// model's `docs search` does. One tab-separated row per document, best first: path,
    /// similarity, page, byte range within the page, index date, best-matching passage.
    Search {
        /// Cap on the documents listed; the daemon's `embedding.top_k` when omitted.
        #[arg(short = 'n', long, value_parser = clap::value_parser!(u32).range(1..=20))]
        limit: Option<u32>,
        /// The words of the query; they are joined, so quoting is not needed.
        #[arg(required = true)]
        query: Vec<String>,
    },
}

pub(crate) async fn run(args: DocumentsArgs) -> Result<()> {
    match args.action {
        DocumentsAction::Status => status().await,
        DocumentsAction::Reindex { quiet } => reindex(quiet).await,
        DocumentsAction::Search { limit, query } => {
            search(query.join(" "), limit.unwrap_or_default()).await
        }
    }
}

async fn status() -> Result<()> {
    let request = Request::new(Uuid::new_v4().to_string(), RequestBody::DocumentsStatus);
    run_one_shot(request, |event| {
        let mut out = io::stdout();
        match &event.body {
            EventBody::DocumentIndex {
                indexing,
                embedding,
                chunks,
                chunk_limit,
                running_pass,
            } => {
                writeln!(out, "indexing: {}", describe_readiness(indexing))?;
                writeln!(out, "embedding: {}", describe_readiness(embedding))?;
                writeln!(out, "chunks: {chunks} of {chunk_limit}")?;
                if chunks >= chunk_limit {
                    writeln!(out, "the index is full: files not yet stored are skipped")?;
                }
                if let Some(root) = running_pass {
                    let root = escape_controls_single_line(root);
                    writeln!(out, "a pass is running over {root}")?;
                }
            }
            EventBody::DocumentRoot { status } => write!(out, "\n{}", describe_root(status))?,
            _ => {}
        }
        Ok(())
    })
    .await
}

async fn reindex(quiet: bool) -> Result<()> {
    let request = Request::new(Uuid::new_v4().to_string(), RequestBody::DocumentsReindex);
    let mut progress_root: Option<String> = None;
    run_one_shot(request, |event| {
        match &event.body {
            EventBody::DocumentPassProgress { root, counts } if !quiet => {
                print_pass_progress(root, counts, &mut progress_root);
            }
            EventBody::DocumentPassEnded { root, pass } => {
                if progress_root.take().is_some() {
                    writeln!(io::stderr())?;
                }
                let root = escape_controls_single_line(root);
                let mut out = io::stdout();
                writeln!(out, "{root}: {}", describe_pass(pass))?;
                if let Some(skipped) = describe_skipped(&pass.counts.skipped) {
                    writeln!(out, "  skipped: {skipped}")?;
                }
            }
            _ => {}
        }
        Ok(())
    })
    .await
}

async fn search(query: String, limit: u32) -> Result<()> {
    let request = Request::new(
        Uuid::new_v4().to_string(),
        RequestBody::DocumentsSearch { query, limit },
    );
    let mut any_hit = false;
    run_one_shot(request, |event| {
        match &event.body {
            EventBody::DocumentHit {
                path,
                page,
                page_count,
                byte_start,
                byte_end,
                indexed_at,
                content,
                similarity,
            } => {
                any_hit = true;
                writeln!(
                    io::stdout(),
                    "{}\tsim={}\tpage {page} of {page_count}\tbytes {byte_start}..{byte_end}\t\
                     indexed {}\t{}",
                    escape_controls_single_line(path),
                    similarity_percent(*similarity),
                    indexed_at.format(INDEXED_DATE_FORMAT),
                    escape_controls_single_line(&collapse_whitespace(content)),
                )?;
            }
            EventBody::Done if !any_hit => writeln!(
                io::stderr(),
                "no document matched; `assistd documents status` shows what is indexed and \
                 still waiting for a vector"
            )?,
            _ => {}
        }
        Ok(())
    })
    .await
}

/// Overwrite the stderr progress line in place, starting a new line when the pass
/// moves on to another root.
fn print_pass_progress(root: &str, counts: &DocumentPassCounts, last_root: &mut Option<String>) {
    let mut err = io::stderr();
    if last_root.as_deref() != Some(root) {
        if last_root.is_some() {
            let _ = writeln!(err);
        }
        *last_root = Some(root.to_owned());
    }
    let _ = write!(
        err,
        "\rindexing {}: {} indexed, {} unchanged, {} skipped",
        escape_controls_single_line(root),
        counts.indexed,
        counts.unchanged,
        counts.skipped_files(),
    );
    let _ = err.flush();
}

fn describe_readiness(readiness: &ComponentReadiness) -> String {
    escape_controls_single_line(&readiness.to_string()).into_owned()
}

/// The lines `status` prints for one indexed folder.
fn describe_root(status: &DocumentRootStatus) -> String {
    let folder = match &status.unavailable {
        None => "available".to_owned(),
        Some(reason) => format!("unavailable ({})", escape_controls_single_line(reason)),
    };
    let unwatched = status
        .unwatched
        .as_ref()
        .map(|reason| {
            let reason = escape_controls_single_line(reason);
            format!("  changes: wait for the next pass ({reason})\n")
        })
        .unwrap_or_default();
    let last_pass = status
        .last_pass
        .as_ref()
        .map_or_else(|| "none since the daemon started".to_owned(), describe_pass);
    let skipped = status
        .last_pass
        .as_ref()
        .and_then(|pass| describe_skipped(&pass.counts.skipped))
        .map(|skipped| format!("  skipped: {skipped}\n"))
        .unwrap_or_default();
    format!(
        "{}\n  folder: {folder}\n{unwatched}  stored: {} documents in {} chunks, {} waiting for a \
         vector\n  last pass: {last_pass}\n{skipped}",
        escape_controls_single_line(&status.root),
        status.documents,
        status.chunks,
        status.unembedded_chunks,
    )
}

/// How `pass` ended, when in local time, and what it did.
fn describe_pass(pass: &DocumentPass) -> String {
    let ended_at = local_time(pass.ended_at);
    let counts = &pass.counts;
    let handled = format!(
        "{} indexed, {} unchanged, {} skipped",
        counts.indexed,
        counts.unchanged,
        counts.skipped_files()
    );
    match &pass.end {
        DocumentPassEnd::Finished => {
            format!("finished {ended_at}: {handled}, {} removed", counts.removed)
        }
        DocumentPassEnd::RootUnavailable { reason } => format!(
            "left the folder alone {ended_at} and kept its documents: {}",
            escape_controls_single_line(reason)
        ),
        DocumentPassEnd::Incomplete { reason } => format!(
            "stopped early {ended_at} and removed no document: {} ({handled})",
            escape_controls_single_line(reason)
        ),
    }
}

/// `4 too large, 2 unsupported type`; `None` when nothing was skipped.
fn describe_skipped(skipped: &[SkippedFiles]) -> Option<String> {
    let reasons: Vec<String> = skipped
        .iter()
        .map(|skipped| format!("{} {}", skipped.files, skipped.reason))
        .collect();
    (!reasons.is_empty()).then(|| reasons.join(", "))
}

fn local_time(at: DateTime<Utc>) -> String {
    at.with_timezone(&Local)
        .format(PASS_TIME_FORMAT)
        .to_string()
}
