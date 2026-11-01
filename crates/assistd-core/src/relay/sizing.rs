//! How much text a "read" relay's inputs hold, measured by file size before
//! its sub-agent starts, so work too large for one sub-agent is split instead.

use std::fs::{self, Metadata};
use std::path::{Path, PathBuf};

/// Bytes of text per token in the estimate, as the chat history counts them.
const BYTES_PER_TOKEN: u64 = 4;

/// Parts named in a refusal, largest first.
const LARGEST_PARTS: usize = 8;

/// Files and directories one measurement visits before it gives up counting.
const MAX_ENTRIES: usize = 200_000;

/// An estimate of the tokens in an input; `capped` when counting stopped
/// at the limit, so the input holds at least that many.
#[derive(Debug, Clone, Copy)]
struct Estimate {
    tokens: u64,
    capped: bool,
}

/// A running count of bytes and entries, stopped at a byte limit.
struct Tally {
    bytes: u64,
    limit: u64,
    entries: usize,
}

/// Refuse `inputs` when they hold more than `limit` tokens of files, with a
/// message naming their largest parts so the caller can split them.
pub(super) fn check_inputs(inputs: &[PathBuf], limit: u64, target: u64) -> Result<(), String> {
    let total = estimate(inputs, limit);
    if !total.capped && total.tokens <= limit {
        return Ok(());
    }
    let parts = largest_parts(inputs, limit)
        .iter()
        .map(|(name, size)| format!("{name} {}", size.render()))
        .collect::<Vec<_>>()
        .join(", ");
    Err(format!(
        "its inputs hold {} of text, more than one sub-agent can get through (the limit is \
         about {}). Split the work into several relays of about {} or less each, \
         make those calls together, and combine their reports. Largest parts: {parts}. Leave \
         out build output and dependencies. If the task only searches the inputs instead of \
         reading them through, set \"search_only\": true on this step.",
        total.render(),
        thousands(limit),
        thousands(target)
    ))
}

impl Estimate {
    fn render(self) -> String {
        match (self.capped, self.tokens < 1000) {
            (true, _) => format!("over {}", thousands(self.tokens)),
            (false, true) => "under 1k tokens".to_string(),
            (false, false) => format!("about {}", thousands(self.tokens)),
        }
    }
}

impl Tally {
    fn new(limit_tokens: u64) -> Self {
        Self {
            bytes: 0,
            limit: limit_tokens.saturating_mul(BYTES_PER_TOKEN),
            entries: 0,
        }
    }

    /// Whether counting may go on: under the byte limit and the entry cap.
    fn open(&self) -> bool {
        self.bytes <= self.limit && self.entries < MAX_ENTRIES
    }

    /// Count the file or directory tree at `path`, described by `meta`,
    /// without following symlinks inside it.
    fn count(&mut self, path: &Path, meta: &Metadata) {
        self.entries += 1;
        if meta.is_file() {
            self.bytes = self.bytes.saturating_add(meta.len());
            return;
        }
        if !meta.is_dir() {
            return;
        }
        let Ok(entries) = fs::read_dir(path) else {
            return;
        };
        for entry in entries.flatten() {
            if !self.open() {
                return;
            }
            if let Ok(meta) = entry.metadata() {
                self.count(&entry.path(), &meta);
            }
        }
    }

    fn estimate(&self) -> Estimate {
        Estimate {
            tokens: self.bytes / BYTES_PER_TOKEN,
            capped: !self.open(),
        }
    }
}

/// Tokens in all of `inputs`, counted up to just past `limit`.
fn estimate(inputs: &[PathBuf], limit: u64) -> Estimate {
    let mut tally = Tally::new(limit);
    for input in inputs {
        if let Ok(meta) = fs::metadata(input) {
            tally.count(input, &meta);
        }
    }
    tally.estimate()
}

/// Each input, or each entry of an input directory, with its estimate,
/// largest first.
fn largest_parts(inputs: &[PathBuf], limit: u64) -> Vec<(String, Estimate)> {
    let mut parts: Vec<(String, Estimate)> = inputs
        .iter()
        .flat_map(|input| parts_of(input))
        .map(|(name, path)| (name, estimate(&[path], limit)))
        .collect();
    parts.sort_by_key(|(_, size)| std::cmp::Reverse(size.tokens));
    parts.truncate(LARGEST_PARTS);
    parts
}

/// The entries of `input` named under it when it is a directory, else
/// `input` itself.
fn parts_of(input: &Path) -> Vec<(String, PathBuf)> {
    let name = input.file_name().map_or_else(
        || input.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    );
    let Ok(entries) = fs::read_dir(input) else {
        return vec![(name, input.to_path_buf())];
    };
    entries
        .flatten()
        .map(|entry| {
            let part = format!("{name}/{}", entry.file_name().to_string_lossy());
            (part, entry.path())
        })
        .collect()
}

/// `tokens` rounded to the nearest thousand, as `12k tokens`.
fn thousands(tokens: u64) -> String {
    format!("{}k tokens", (tokens + 500) / 1000)
}
