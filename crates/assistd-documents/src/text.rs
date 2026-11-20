//! Bytes decoded to text, and HTML and XHTML rendered to it.

use std::borrow::Cow;

use assistd_utils::binary::{binary_format, sniff_binary};
use encoding_rs::{Encoding, UTF_8};
use quick_xml::events::Event;
use quick_xml::{Reader, Writer};

use crate::protocol::SkipReason;

/// Columns an HTML file's text is wrapped to.
const HTML_TEXT_WIDTH: usize = 100;

/// The HTML elements that never have an end tag.
const VOID_ELEMENTS: [&str; 13] = [
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track",
    "wbr",
];

/// `bytes` as text in the encoding their byte-order mark names, or UTF-8 without
/// one; bytes that look binary are refused.
pub(crate) fn decode(bytes: &[u8]) -> Result<String, SkipReason> {
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

/// The text of `html`, its headings as Markdown headings and its links without
/// their targets.
pub(crate) fn html_text(html: &str) -> Result<String, SkipReason> {
    html2text::config::plain_no_decorate()
        .allow_width_overflow()
        .no_table_borders()
        .string_from_read(html.as_bytes(), HTML_TEXT_WIDTH)
        .map_err(|error| SkipReason::ExtractorFailed {
            message: error.to_string(),
        })
}

/// The text of `xhtml` as [`html_text`] gives it, an element XML leaves empty
/// with `<title/>` closed first, where an HTML parser would read on inside it.
pub(crate) fn xhtml_text(xhtml: &str) -> Result<String, SkipReason> {
    html_text(with_end_tags(xhtml).as_deref().unwrap_or(xhtml))
}

/// `xhtml` with each empty-element tag of an element HTML gives an end tag
/// written as a start and an end tag; `None` when it is not well-formed XML.
fn with_end_tags(xhtml: &str) -> Option<String> {
    let mut reader = Reader::from_str(xhtml);
    let mut html = Writer::new(Vec::with_capacity(xhtml.len()));
    loop {
        match reader.read_event().ok()? {
            Event::Eof => break,
            Event::Empty(element) if !VOID_ELEMENTS.contains(&element.local_name().as_ref()) => {
                html.write_event(Event::Start(element.borrow())).ok()?;
                html.write_event(Event::End(element.to_end())).ok()?;
            }
            event => html.write_event(event).ok()?,
        }
    }
    String::from_utf8(html.into_inner()).ok()
}
