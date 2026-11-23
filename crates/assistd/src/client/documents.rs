//! `documents` subcommands: report the document index, start a pass over the indexed
//! folders and follow it, or search the indexed documents, all through the daemon.

use std::io::{self, Write};

use anyhow::Result;
use assistd_ipc::{
    ComponentReadiness, DocumentPass, DocumentPassCounts, DocumentPassEnd, DocumentRootStatus,
    EventBody, PageTranscriptionStatus, Request, RequestBody, SearchKind, SkippedFiles,
};
use assistd_utils::terminal_text::escape_controls_single_line;
use chrono::{DateTime, Local, Utc};
use clap::{Args, Subcommand};
use uuid::Uuid;

use super::run_one_shot;
use super::search::{self, hit_count, print_hits};

const PASS_TIME_FORMAT: &str = "%Y-%m-%d %H:%M %Z";

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
    /// Rank the indexed documents by meaning and by the query's exact words:
    /// `assistd search --in documents`.
    Search {
        /// Cap on the documents listed; the daemon's `embedding.top_k` when omitted.
        #[arg(short = 'n', long, value_parser = hit_count())]
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
            let limit = limit.unwrap_or_default();
            print_hits(search::request(
                query.join(" "),
                limit,
                vec![SearchKind::Documents],
            ))
            .await
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
                transcription,
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
                write!(out, "{}", describe_transcription(transcription))?;
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

/// The lines `status` prints on the pages that have no text.
fn describe_transcription(transcription: &PageTranscriptionStatus) -> String {
    let stopped = transcription.stopped.as_ref().map_or_default(|why| {
        let why = escape_controls_single_line(why);
        format!("the last transcription run stopped early: {why}\n")
    });
    format!(
        "page transcription: {}\npages without text: {} waiting for transcription, {} failed\n\
         {stopped}",
        transcription.state, transcription.waiting_pages, transcription.failed_pages,
    )
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
    let unwatched = status.unwatched.as_ref().map_or_default(|reason| {
        let reason = escape_controls_single_line(reason);
        format!("  changes: wait for the next pass ({reason})\n")
    });
    let last_pass = status
        .last_pass
        .as_ref()
        .map_or_else(|| "none since the daemon started".to_owned(), describe_pass);
    let skipped = status
        .last_pass
        .as_ref()
        .and_then(|pass| describe_skipped(&pass.counts.skipped))
        .map_or_default(|skipped| format!("  skipped: {skipped}\n"));
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
