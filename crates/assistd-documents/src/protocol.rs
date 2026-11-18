//! Line-delimited JSON spoken between the daemon and its extraction worker over
//! the worker's stdin and stdout.

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use thiserror::Error;

/// Most bytes of text one file yields; a longer text is cut there.
pub const MAX_TEXT_BYTES: usize = 4 * 1024 * 1024;

/// Most bytes of one line the daemon reads from the worker: [`MAX_TEXT_BYTES`]
/// with every byte escaped to JSON's six, and room for the rest of the message.
pub(crate) const MAX_REPLY_LINE_BYTES: usize = 6 * MAX_TEXT_BYTES + 1024;

/// A line from the other side that is not a message this build knows.
#[derive(Debug, Error)]
#[error("malformed extraction-worker message: {0}")]
pub(crate) struct ProtocolError(#[from] serde_json::Error);

/// Daemon → worker: extract the text of one file, answered with a
/// [`FromWorker::Reply`] of the same id.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Request {
    pub(crate) id: u64,
    pub(crate) path: String,
    /// The largest file read, in bytes; a larger one is skipped.
    pub(crate) max_file_bytes: u64,
}

/// Worker → daemon.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum FromWorker {
    /// The first line, sent once the worker can take requests.
    Ready,
    Reply {
        id: u64,
        extraction: Extraction,
    },
}

/// What the worker made of one file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Extraction {
    Text {
        /// At most [`MAX_TEXT_BYTES`] long.
        text: String,
        /// Whether the file held more text than `text` carries.
        truncated: bool,
    },
    Skipped {
        reason: SkipReason,
    },
}

impl Extraction {
    /// `text` cut to [`MAX_TEXT_BYTES`] at a character boundary.
    pub(crate) fn of_text(mut text: String) -> Self {
        let truncated = text.len() > MAX_TEXT_BYTES;
        if truncated {
            text.truncate(text.floor_char_boundary(MAX_TEXT_BYTES));
        }
        Self::Text { text, truncated }
    }

    /// This extraction with its text held to [`MAX_TEXT_BYTES`], whatever the
    /// worker sent.
    pub(crate) fn capped(self) -> Self {
        match self {
            Self::Text { text, .. } if text.len() > MAX_TEXT_BYTES => Self::of_text(text),
            extraction => extraction,
        }
    }
}

/// Why a file has no text to index.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Error)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SkipReason {
    /// A recognised format no extractor reads.
    #[error("unsupported file type {mime}")]
    Unsupported { mime: String },
    #[error("binary file")]
    Binary,
    #[error("larger than the file-size limit")]
    TooLarge,
    #[error("not valid text in its encoding")]
    NotDecodable,
    #[error("cannot be read: {message}")]
    Unreadable { message: String },
    #[error("extraction failed: {message}")]
    ExtractorFailed { message: String },
    #[error("extraction timed out")]
    TimedOut,
}

/// `message` as one line, newline included.
pub(crate) fn encode(message: &impl Serialize) -> String {
    let mut line =
        serde_json::to_string(message).expect("extraction-worker messages always serialize");
    line.push('\n');
    line
}

pub(crate) fn decode<T: DeserializeOwned>(line: &[u8]) -> Result<T, ProtocolError> {
    Ok(serde_json::from_slice(line)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trips<T>(messages: &[T])
    where
        T: Serialize + DeserializeOwned + PartialEq + std::fmt::Debug,
    {
        for message in messages {
            let line = encode(message);
            assert_eq!(
                line.find('\n'),
                Some(line.len() - 1),
                "one line per message: {line}"
            );
            let decoded: T = decode(line.trim_end().as_bytes()).expect(&line);
            assert_eq!(&decoded, message, "{line}");
        }
    }

    #[test]
    fn every_message_round_trips_as_one_line() {
        round_trips(&[Request {
            id: u64::MAX,
            path: "/home/someone/notes/Line \"one\"\nline two.md".into(),
            max_file_bytes: 20 * 1024 * 1024,
        }]);
        let reasons = [
            SkipReason::Unsupported {
                mime: "application/pdf".into(),
            },
            SkipReason::Binary,
            SkipReason::TooLarge,
            SkipReason::NotDecodable,
            SkipReason::Unreadable {
                message: "No such file or directory (os error 2)".into(),
            },
            SkipReason::ExtractorFailed {
                message: "the extractor panicked".into(),
            },
            SkipReason::TimedOut,
        ];
        let mut messages: Vec<FromWorker> = reasons
            .into_iter()
            .zip(1..)
            .map(|(reason, id)| FromWorker::Reply {
                id,
                extraction: Extraction::Skipped { reason },
            })
            .collect();
        messages.extend([
            FromWorker::Ready,
            FromWorker::Reply {
                id: 0,
                extraction: Extraction::Text {
                    text: "# Title\n\nBody with a \u{1} control, a tab\tand \u{2028} a separator."
                        .into(),
                    truncated: true,
                },
            },
        ]);
        round_trips(&messages);
    }

    #[test]
    fn the_longest_text_fits_the_reply_line_however_it_escapes() {
        let reply = FromWorker::Reply {
            id: u64::MAX,
            extraction: Extraction::of_text("\u{1}".repeat(MAX_TEXT_BYTES + 1)),
        };
        assert!(encode(&reply).len() <= MAX_REPLY_LINE_BYTES);
    }
}
