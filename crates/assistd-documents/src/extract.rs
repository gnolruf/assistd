//! Turns one file into text, or one of its pages into an image: the format its
//! extension names, its decoding, and the limits on what is read and returned.

use std::borrow::Cow;
use std::ffi::OsStr;
use std::io::{self, Read};
use std::path::Path;

use assistd_utils::binary::{binary_format, sniff_binary};
use assistd_utils::fs::open_regular;
use encoding_rs::{Encoding, UTF_8};

use crate::pdf;
use crate::protocol::{Extraction, RenderedPage, SkipReason};

const HTML_EXTENSIONS: [&str; 3] = ["html", "htm", "xhtml"];

const PDF_EXTENSION: &str = "pdf";

/// Columns an HTML file's text is wrapped to.
const HTML_TEXT_WIDTH: usize = 100;

/// How a file's bytes become its text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    /// Plain text, Markdown and source code, kept as they are.
    Text,
    Html,
    Pdf,
}

impl Format {
    fn of(path: &Path) -> Self {
        let extension = path.extension().and_then(OsStr::to_str).unwrap_or_default();
        if extension.eq_ignore_ascii_case(PDF_EXTENSION) {
            Self::Pdf
        } else if HTML_EXTENSIONS
            .iter()
            .any(|html| extension.eq_ignore_ascii_case(html))
        {
            Self::Html
        } else {
            Self::Text
        }
    }

    fn extract(self, bytes: Vec<u8>) -> Result<Extraction, SkipReason> {
        match self {
            Self::Text => decode(&bytes).map(Extraction::of_text),
            Self::Html => decode(&bytes)
                .and_then(|html| html_text(&html))
                .map(Extraction::of_text),
            Self::Pdf => pdf::page_texts(bytes),
        }
    }
}

/// The text of the regular file at `path`, or why it has none; a file larger
/// than `max_file_bytes` is not read. Blocks on the filesystem.
pub(crate) fn extract_text(path: &Path, max_file_bytes: u64) -> Extraction {
    read_file(path, max_file_bytes)
        .and_then(|bytes| Format::of(path).extract(bytes))
        .unwrap_or_else(|reason| Extraction::Skipped { reason })
}

/// Page `page` of the PDF at `path`, counted from zero, as an image; a file
/// larger than `max_file_bytes` is not read. Blocks on the filesystem.
pub(crate) fn render_page(path: &Path, max_file_bytes: u64, page: u32) -> RenderedPage {
    match read_file(path, max_file_bytes) {
        Ok(bytes) => pdf::render_page(bytes, page),
        Err(reason) => RenderedPage::Failed { reason },
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
    let mut bytes = Vec::with_capacity(usize::try_from(size).unwrap_or_default());
    let read = file
        .take(max_file_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(unreadable)?;
    if read as u64 > max_file_bytes {
        return Err(SkipReason::TooLarge);
    }
    Ok(bytes)
}

fn html_text(html: &str) -> Result<String, SkipReason> {
    html2text::config::plain_no_decorate()
        .allow_width_overflow()
        .no_table_borders()
        .string_from_read(html.as_bytes(), HTML_TEXT_WIDTH)
        .map_err(|error| SkipReason::ExtractorFailed {
            message: error.to_string(),
        })
}

/// `bytes` as text in the encoding their byte-order mark names, or UTF-8 without
/// one; bytes that look binary are refused.
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
    use crate::sample_pdf::{EMPTY_PAGE, IMAGE_PAGE, pdf_of, text_page};

    const PNG_MAGIC: &[u8] = b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR";

    /// The first six bytes of a PNG in base64.
    const PNG_MAGIC_BASE64: &str = "iVBORw0K";

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
        let cases: [(&str, &[u8], Extraction); 6] = [
            (
                "notes.md",
                "# Café\n\nBody.\n".as_bytes(),
                text("# Café\n\nBody.\n"),
            ),
            ("marked.txt", b"\xEF\xBB\xBFplain", text("plain")),
            ("wide.txt", &utf16, text("naïve")),
            ("latin1.txt", b"caf\xE9", skipped(SkipReason::NotDecodable)),
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
    fn a_pdf_yields_each_pages_text_and_untranscribed_is_skipped_when_no_page_has_any() {
        let scanned = extraction_of("scan.PDF", &pdf_of(&[IMAGE_PAGE, IMAGE_PAGE]), 4096);
        assert_eq!(
            scanned,
            Extraction::Pages {
                pages: vec![String::new(), String::new()],
                truncated: false,
            }
        );
        assert_eq!(
            scanned.with_pages_joined(),
            skipped(SkipReason::NoTextLayer)
        );

        let pages = [&text_page("Minutes"), IMAGE_PAGE, &text_page("Appendix")];
        let mixed = extraction_of("minutes.pdf", &pdf_of(&pages), 4096);
        let Extraction::Pages { pages, truncated } = &mixed else {
            panic!("a PDF comes back as pages: {mixed:?}");
        };
        assert!(!truncated);
        assert!(pages[0].contains("Minutes"), "{pages:?}");
        assert_eq!(pages[1], "");
        let Extraction::Text { text, .. } = mixed.with_pages_joined() else {
            panic!("pages with text join into one text");
        };
        let (minutes, appendix) = text.split_once("\n\n").expect("a break between pages");
        assert!(minutes.contains("Minutes"), "{text:?}");
        assert!(appendix.contains("Appendix"), "{text:?}");
    }

    #[test]
    fn a_pdf_page_is_rendered_to_a_png_unless_it_shows_nothing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("scan.pdf");
        std::fs::write(&path, pdf_of(&[IMAGE_PAGE, EMPTY_PAGE])).expect("file");

        let RenderedPage::Image { png_base64 } = render_page(&path, 4096, 0) else {
            panic!("a page with marks is an image");
        };
        assert!(png_base64.starts_with(PNG_MAGIC_BASE64), "{png_base64}");
        assert_eq!(render_page(&path, 4096, 1), RenderedPage::Blank);
        assert_eq!(
            render_page(&path, 16, 0),
            RenderedPage::Failed {
                reason: SkipReason::TooLarge
            }
        );
    }

    #[test]
    fn a_file_over_the_limit_or_not_a_regular_file_is_never_read() {
        assert_eq!(
            extraction_of("big.txt", b"12345", 4),
            skipped(SkipReason::TooLarge)
        );
        assert_eq!(extraction_of("fits.txt", b"1234", 4), text("1234"));
        assert!(matches!(
            extract_text(Path::new("/dev/zero"), 4),
            Extraction::Skipped {
                reason: SkipReason::Unreadable { .. }
            }
        ));
    }
}
