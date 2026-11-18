//! Turns one file into text: the format its extension names, its decoding, and
//! the limits on what is read and returned.

use std::borrow::Cow;
use std::ffi::OsStr;
use std::io::{self, Read};
use std::path::Path;

use assistd_utils::binary::{binary_format, sniff_binary};
use assistd_utils::fs::open_regular;
use encoding_rs::{Encoding, UTF_8};

use crate::protocol::{Extraction, SkipReason};

const HTML_EXTENSIONS: [&str; 3] = ["html", "htm", "xhtml"];

/// Columns an HTML file's text is wrapped to.
const HTML_TEXT_WIDTH: usize = 100;

/// How a file's decoded bytes become its text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    /// Plain text, Markdown and source code, kept as they are.
    Text,
    Html,
}

impl Format {
    fn of(path: &Path) -> Self {
        let is_html = path
            .extension()
            .and_then(OsStr::to_str)
            .is_some_and(|extension| {
                HTML_EXTENSIONS
                    .iter()
                    .any(|html| extension.eq_ignore_ascii_case(html))
            });
        if is_html { Self::Html } else { Self::Text }
    }

    fn render(self, decoded: String) -> Result<String, SkipReason> {
        match self {
            Self::Text => Ok(decoded),
            Self::Html => html2text::config::plain_no_decorate()
                .allow_width_overflow()
                .no_table_borders()
                .string_from_read(decoded.as_bytes(), HTML_TEXT_WIDTH)
                .map_err(|error| SkipReason::ExtractorFailed {
                    message: error.to_string(),
                }),
        }
    }
}

/// The text of the regular file at `path`, or why it has none; a file larger
/// than `max_file_bytes` is not read. Blocks on the filesystem.
pub(crate) fn extract_text(path: &Path, max_file_bytes: u64) -> Extraction {
    let text = read_file(path, max_file_bytes)
        .and_then(|bytes| decode(&bytes))
        .and_then(|decoded| Format::of(path).render(decoded));
    match text {
        Ok(text) => Extraction::of_text(text),
        Err(reason) => Extraction::Skipped { reason },
    }
}

fn read_file(path: &Path, max_file_bytes: u64) -> Result<Vec<u8>, SkipReason> {
    let unreadable = |error: io::Error| SkipReason::Unreadable {
        message: error.to_string(),
    };
    let (file, size) = open_regular(path).map_err(unreadable)?;
    if size > max_file_bytes {
        return Err(SkipReason::TooLarge);
    }
    let mut bytes = Vec::new();
    let read = file
        .take(max_file_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(unreadable)?;
    if read as u64 > max_file_bytes {
        return Err(SkipReason::TooLarge);
    }
    Ok(bytes)
}

/// `bytes` as text in the encoding their byte-order mark names, or UTF-8 without
/// one, where a file that looks binary as `cat` judges it is refused.
fn decode(bytes: &[u8]) -> Result<String, SkipReason> {
    if let Some(mime) = binary_format(bytes) {
        return Err(SkipReason::Unsupported {
            mime: mime.to_owned(),
        });
    }
    let (encoding, mark_len) = Encoding::for_bom(bytes).unwrap_or((UTF_8, 0));
    if encoding == UTF_8 && sniff_binary(bytes).is_some() {
        return Err(SkipReason::Binary);
    }
    encoding
        .decode_without_bom_handling_and_without_replacement(&bytes[mark_len..])
        .map(Cow::into_owned)
        .ok_or(SkipReason::NotDecodable)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PNG_MAGIC: &[u8] = b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR";

    fn extraction_of(name: &str, contents: &[u8], max_file_bytes: u64) -> Extraction {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join(name);
        std::fs::write(&path, contents).expect("file");
        extract_text(&path, max_file_bytes)
    }

    fn text(text: &str) -> Extraction {
        Extraction::Text {
            text: text.into(),
            truncated: false,
        }
    }

    fn skipped(reason: SkipReason) -> Extraction {
        Extraction::Skipped { reason }
    }

    #[test]
    fn each_kind_of_file_yields_its_text_or_the_reason_it_has_none() {
        let utf16: Vec<u8> = [0xFF, 0xFE]
            .into_iter()
            .chain("naïve".encode_utf16().flat_map(u16::to_le_bytes))
            .collect();
        let unsupported = SkipReason::Unsupported {
            mime: "image/png".into(),
        };
        let cases: [(&str, &[u8], Extraction); 8] = [
            (
                "notes.md",
                "# Café\n\nBody.\n".as_bytes(),
                text("# Café\n\nBody.\n"),
            ),
            ("marked.txt", b"\xEF\xBB\xBFplain", text("plain")),
            ("wide.txt", &utf16, text("naïve")),
            ("main.rs", b"fn main() {}\n", text("fn main() {}\n")),
            ("latin1.txt", b"caf\xE9", skipped(SkipReason::NotDecodable)),
            ("odd.txt", b"\xFF\xFEa", skipped(SkipReason::NotDecodable)),
            ("blob.dat", b"ab\0cd", skipped(SkipReason::Binary)),
            ("picture.txt", PNG_MAGIC, skipped(unsupported)),
        ];
        for (name, contents, expected) in cases {
            assert_eq!(extraction_of(name, contents, 1024), expected, "{name}");
        }
    }

    #[test]
    fn html_is_rendered_to_its_text_under_markdown_headings_without_link_targets() {
        let page = b"<html><head><title>T</title><script>var hidden = 1;</script></head>\
            <body><h1>Guide</h1><p>Read <a href=\"https://example.com/x\">this</a> first.</p>\
            </body></html>";
        for name in ["page.html", "PAGE.HTM"] {
            assert_eq!(
                extraction_of(name, page, 1024),
                text("# Guide\n\nRead [this] first.\n"),
                "{name}"
            );
        }
    }

    #[test]
    fn a_file_over_the_limit_or_not_a_regular_file_is_never_read() {
        assert_eq!(
            extraction_of("big.txt", b"12345", 4),
            skipped(SkipReason::TooLarge)
        );
        assert_eq!(extraction_of("fits.txt", b"1234", 4), text("1234"));
        for unreadable in ["/dev/zero", "/nonexistent/file.txt"] {
            assert!(
                matches!(
                    extract_text(Path::new(unreadable), 4),
                    Extraction::Skipped {
                        reason: SkipReason::Unreadable { .. }
                    }
                ),
                "{unreadable}"
            );
        }
    }
}
