//! Line-delimited JSON spoken between the daemon and its extraction worker over
//! the worker's stdin and stdout.

use std::convert::Infallible;

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use thiserror::Error;

/// Most bytes of text one file yields; a longer text is cut there.
pub const MAX_TEXT_BYTES: usize = 4 * 1024 * 1024;

/// Most pages of one file the worker reports; a longer file is cut there.
pub const MAX_PAGES: usize = 20_000;

/// Most bytes of the PNG one page is rendered to; a larger image is a failed render.
pub const MAX_PAGE_IMAGE_BYTES: usize = 12 * 1024 * 1024;

/// [`MAX_PAGE_IMAGE_BYTES`] in padded base64, four characters to three bytes.
const MAX_PAGE_IMAGE_BASE64_BYTES: usize = MAX_PAGE_IMAGE_BYTES.div_ceil(3) * 4;

/// Most bytes of one line the daemon reads from the worker: [`MAX_TEXT_BYTES`]
/// with every byte escaped to JSON's six, the quotes and comma of [`MAX_PAGES`]
/// pages, and room for the rest of the message, the file's hash included. A page image in base64 is shorter.
pub(crate) const MAX_REPLY_LINE_BYTES: usize = 6 * MAX_TEXT_BYTES + 3 * MAX_PAGES + 1024;

/// A line from the other side that is not a message this build knows.
#[derive(Debug, Error)]
#[error("malformed extraction-worker message: {0}")]
pub(crate) struct ProtocolError(#[from] serde_json::Error);

/// Daemon → worker: one task on one file, answered with the [`FromWorker`]
/// message of the same id that the task names.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Request {
    pub(crate) id: u64,
    pub(crate) path: String,
    /// The largest file read, in bytes; a larger one is skipped.
    pub(crate) max_file_bytes: u64,
    pub(crate) task: Task,
}

/// What the worker does with a request's file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum Task {
    /// Extract its text, answered with a [`FromWorker::Reply`].
    Extract,
    /// Render one page of a PDF, counted from zero, answered with a
    /// [`FromWorker::Rendered`].
    RenderPage { page: u32 },
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
    Rendered {
        id: u64,
        page: RenderedPage,
    },
}

/// What the worker made of one file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Extraction {
    /// The file's text as the pages its format divides it into: one page for a
    /// format that has none.
    Pages {
        /// Each page's text in page order, empty for a page without text. At most
        /// [`MAX_PAGES`] pages and [`MAX_TEXT_BYTES`] in all.
        pages: Vec<String>,
        /// Whether the file held more pages or text than `pages` carries.
        truncated: bool,
        /// SHA-256 of the file's bytes as they were read.
        file_hash: [u8; 32],
    },
    Skipped {
        reason: SkipReason,
    },
}

impl Extraction {
    /// This extraction with its text held to [`MAX_TEXT_BYTES`] and its pages to
    /// [`MAX_PAGES`], whatever the worker sent.
    pub(crate) fn capped(self) -> Self {
        let Self::Pages {
            pages,
            truncated,
            file_hash,
        } = self
        else {
            return self;
        };
        let capped = PageTexts::of_pages(pages);
        Self::Pages {
            pages: capped.pages,
            truncated: capped.truncated || truncated,
            file_hash,
        }
    }
}

/// The text of one file as the pages its format divides it into, within the limits
/// on what the worker reports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PageTexts {
    /// At most [`MAX_PAGES`] pages and [`MAX_TEXT_BYTES`] in all.
    pub(crate) pages: Vec<String>,
    /// Whether the file held more pages or text than `pages` carries.
    pub(crate) truncated: bool,
}

impl PageTexts {
    /// `text` as the one page of a file whose format has no pages, cut to
    /// [`MAX_TEXT_BYTES`] at a character boundary.
    pub(crate) fn of_one_page(text: String) -> Self {
        Self::of_pages([text])
    }

    /// [`Self::try_of_pages`] for texts that are already in hand.
    pub(crate) fn of_pages(texts: impl IntoIterator<Item = String>) -> Self {
        let Ok(page_texts) = Self::try_of_pages(texts.into_iter().map(Ok::<_, Infallible>));
        page_texts
    }

    /// The pages of `texts` up to [`MAX_PAGES`] pages and [`MAX_TEXT_BYTES`] in
    /// all, the page that crosses the limit cut at a character boundary and left
    /// out when nothing of it fits. Stops at the first error.
    pub(crate) fn try_of_pages<E>(
        texts: impl IntoIterator<Item = Result<String, E>>,
    ) -> Result<Self, E> {
        let mut texts = texts.into_iter();
        let mut pages = Vec::new();
        let mut room = MAX_TEXT_BYTES;
        while pages.len() < MAX_PAGES && room > 0 {
            let Some(text) = texts.next() else {
                return Ok(Self {
                    pages,
                    truncated: false,
                });
            };
            let mut text = text?;
            if text.len() > room {
                text.truncate(text.floor_char_boundary(room));
                pages.extend(Some(text).filter(|cut| !cut.is_empty()));
                return Ok(Self {
                    pages,
                    truncated: true,
                });
            }
            room -= text.len();
            pages.push(text);
        }
        let truncated = texts.next().is_some();
        Ok(Self { pages, truncated })
    }
}

/// What the worker made of one page asked for as an image.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum RenderedPage {
    Image {
        /// A PNG of at most [`MAX_PAGE_IMAGE_BYTES`], in standard base64.
        png_base64: String,
    },
    /// The page shows nothing but its background.
    Blank,
    Failed {
        reason: SkipReason,
    },
}

impl RenderedPage {
    /// This page with its image held to [`MAX_PAGE_IMAGE_BYTES`], whatever the
    /// worker sent.
    pub(crate) fn capped(self) -> Self {
        match self {
            Self::Image { png_base64 } if png_base64.len() > MAX_PAGE_IMAGE_BASE64_BYTES => {
                Self::Failed {
                    reason: SkipReason::ExtractorFailed {
                        message: "the page image is larger than the worker may send".into(),
                    },
                }
            }
            page => page,
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
    /// Larger than the file-size limit, or an archive that holds or expands to
    /// more than the worker reads.
    #[error("larger than a size limit")]
    TooLarge,
    #[error("not valid text in its encoding")]
    NotDecodable,
    #[error("needs a password")]
    PasswordProtected,
    /// No page has text, and no page was transcribed.
    #[error("no text layer")]
    NoTextLayer,
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
        let request = |task| Request {
            id: u64::MAX,
            path: "/home/someone/notes/Line \"one\"\nline two.md".into(),
            max_file_bytes: 20 * 1024 * 1024,
            task,
        };
        round_trips(&[
            request(Task::Extract),
            request(Task::RenderPage { page: u32::MAX }),
        ]);
        let reasons = [
            SkipReason::Unsupported {
                mime: "application/zip".into(),
            },
            SkipReason::Binary,
            SkipReason::TooLarge,
            SkipReason::NotDecodable,
            SkipReason::PasswordProtected,
            SkipReason::NoTextLayer,
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
                extraction: Extraction::Pages {
                    pages: vec![
                        "# Title\n\nBody with a \u{1} control, a tab\tand \u{2028} a separator."
                            .into(),
                        String::new(),
                    ],
                    truncated: true,
                    file_hash: [u8::MAX; 32],
                },
            },
            FromWorker::Rendered {
                id: 0,
                page: RenderedPage::Image {
                    png_base64: "iVBORw0KGgo=".into(),
                },
            },
            FromWorker::Rendered {
                id: 0,
                page: RenderedPage::Blank,
            },
            FromWorker::Rendered {
                id: 0,
                page: RenderedPage::Failed {
                    reason: SkipReason::PasswordProtected,
                },
            },
        ]);
        round_trips(&messages);
    }

    #[test]
    fn the_longest_reply_fits_the_reply_line_however_it_escapes() {
        let reply_of = |texts: PageTexts| FromWorker::Reply {
            id: u64::MAX,
            extraction: Extraction::Pages {
                pages: texts.pages,
                truncated: texts.truncated,
                file_hash: [u8::MAX; 32],
            },
        };
        let longest = PageTexts::of_pages((0..=MAX_PAGES).map(|_| "\u{1}".repeat(1024)));
        assert!(longest.truncated);
        let kept: usize = longest.pages.iter().map(String::len).sum();
        assert_eq!(kept, MAX_TEXT_BYTES);
        let empty = PageTexts::of_pages((0..=MAX_PAGES).map(|_| String::new()));
        let image = FromWorker::Rendered {
            id: u64::MAX,
            page: RenderedPage::Image {
                png_base64: "A".repeat(MAX_PAGE_IMAGE_BASE64_BYTES),
            },
        };
        for reply in [reply_of(longest), reply_of(empty), image] {
            assert!(encode(&reply).len() <= MAX_REPLY_LINE_BYTES);
        }
    }
}
