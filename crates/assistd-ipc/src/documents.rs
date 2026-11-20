//! The document index on the wire: what a pass over an index root did, where each root
//! stands, what the index is doing now, and where page transcription stands.

use std::fmt;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Why a file found under an index root is not in the index.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DocumentSkipReason {
    /// Reading it would need the user's confirmation.
    Sensitive,
    TooLarge,
    Unsupported,
    Binary,
    NotDecodable,
    PasswordProtected,
    NoTextLayer,
    Unreadable,
    ExtractorFailed,
    TimedOut,
    /// The index already held as many chunks as it may.
    IndexFull,
}

impl fmt::Display for DocumentSkipReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            DocumentSkipReason::Sensitive => "sensitive",
            DocumentSkipReason::TooLarge => "too large",
            DocumentSkipReason::Unsupported => "unsupported type",
            DocumentSkipReason::Binary => "binary",
            DocumentSkipReason::NotDecodable => "not decodable as text",
            DocumentSkipReason::PasswordProtected => "password protected",
            DocumentSkipReason::NoTextLayer => "no text layer",
            DocumentSkipReason::Unreadable => "unreadable",
            DocumentSkipReason::ExtractorFailed => "extraction failed",
            DocumentSkipReason::TimedOut => "extraction timed out",
            DocumentSkipReason::IndexFull => "index full",
        })
    }
}

/// How many files a pass left out of the index for one reason.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct SkippedFiles {
    pub reason: DocumentSkipReason,
    pub files: u64,
}

/// How many files a pass over an index root indexed, found unchanged, removed and skipped.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct DocumentPassCounts {
    /// Files whose text was extracted and stored.
    pub indexed: u64,
    /// Files whose stored stamp still matched, so they were not read.
    pub unchanged: u64,
    /// Documents removed because the pass no longer found their file.
    pub removed: u64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skipped: Vec<SkippedFiles>,
}

impl DocumentPassCounts {
    /// Files skipped for any reason.
    pub fn skipped_files(&self) -> u64 {
        self.skipped.iter().map(|skipped| skipped.files).sum()
    }
}

/// How a pass over an index root ended.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum DocumentPassEnd {
    /// The whole root was walked, and documents no longer found were removed.
    Finished,
    /// The root was missing, unreadable or empty, so its documents were kept.
    RootUnavailable { reason: String },
    /// The pass stopped early or could not list part of the root, so no document was
    /// removed.
    Incomplete { reason: String },
}

/// What a pass over an index root did, and when and how it ended.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DocumentPass {
    pub ended_at: DateTime<Utc>,
    pub end: DocumentPassEnd,
    pub counts: DocumentPassCounts,
}

/// Where one index root stands.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DocumentRootStatus {
    pub root: String,
    /// Why its files cannot be listed now; `None` while they can.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unavailable: Option<String>,
    /// Why changes under it wait for the next pass instead of being indexed as they
    /// happen; `None` unless its watch failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unwatched: Option<String>,
    /// Documents stored from it.
    pub documents: u64,
    /// Chunks those documents are cut into.
    pub chunks: u64,
    /// Those of its chunks neither embedded nor marked unembeddable.
    pub unembedded_chunks: u64,
    /// The last pass over it since the daemon started; `None` before one has ended.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_pass: Option<DocumentPass>,
}

/// When the pages without text are transcribed.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PageTranscriptionState {
    /// Never: `documents.ocr.model` is unset.
    NoModel,
    /// Never on its own: `sleep.idle_to_drowsy_mins` is `0`, so the daemon does not drowse.
    NeverDrowsy,
    /// Whenever the daemon is drowsy.
    WhileDrowsy,
    /// Now.
    Running,
}

impl fmt::Display for PageTranscriptionState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            PageTranscriptionState::NoModel => "off (documents.ocr.model is unset)",
            PageTranscriptionState::NeverDrowsy => {
                "never runs: pages are read while the daemon is drowsy, and \
                 sleep.idle_to_drowsy_mins = 0 keeps it from drowsing"
            }
            PageTranscriptionState::WhileDrowsy => "runs while the daemon is drowsy",
            PageTranscriptionState::Running => "running",
        })
    }
}

/// Where transcribing the pages that have no text stands.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PageTranscriptionStatus {
    pub state: PageTranscriptionState,
    /// Pages waiting for the OCR model.
    pub waiting_pages: u64,
    /// Pages the model could not read; they are tried again when their file
    /// changes or the daemon restarts.
    pub failed_pages: u64,
    /// Why the last run stopped with pages still waiting; `None` unless it did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stopped: Option<String>,
}

/// What the document index is doing.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct DocumentActivity {
    /// The root a pass is walking; `None` between passes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub indexing: Option<String>,
    /// Document chunks waiting for the embedder; zero while there is none.
    pub unembedded_chunks: u64,
}

impl DocumentActivity {
    /// True when no pass runs and no chunk waits to be embedded.
    pub fn is_idle(&self) -> bool {
        self.indexing.is_none() && self.unembedded_chunks == 0
    }
}
