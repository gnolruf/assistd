//! Turns one file into text, or one of its pages into an image: the format its
//! extension names and the limit on what is read.

use std::ffi::OsStr;
use std::io::{self, Read};
use std::path::Path;

use assistd_utils::fs::open_regular;

use crate::office::{self, OfficeFormat};
use crate::protocol::{Extraction, RenderedPage, SkipReason};
use crate::text::{decode, html_text};
use crate::{epub, opendocument, pdf};

/// The format each extension names, whatever its case; any other is [`Format::Text`].
const EXTENSION_FORMATS: [(&str, Format); 11] = [
    ("html", Format::Html),
    ("htm", Format::Html),
    ("xhtml", Format::Html),
    ("pdf", Format::Pdf),
    ("docx", Format::Office(OfficeFormat::Docx)),
    ("pptx", Format::Office(OfficeFormat::Pptx)),
    ("xlsx", Format::Office(OfficeFormat::Xlsx)),
    ("odt", Format::OpenDocument),
    ("ods", Format::OpenDocument),
    ("odp", Format::OpenDocument),
    ("epub", Format::Epub),
];

/// How a file's bytes become its text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    /// Plain text, Markdown and source code, kept as they are.
    Text,
    Html,
    Pdf,
    Office(OfficeFormat),
    OpenDocument,
    Epub,
}

impl Format {
    fn of(path: &Path) -> Self {
        let extension = path.extension().and_then(OsStr::to_str).unwrap_or_default();
        EXTENSION_FORMATS
            .iter()
            .find(|(known, _)| extension.eq_ignore_ascii_case(known))
            .map_or(Self::Text, |(_, format)| *format)
    }

    fn extract(self, bytes: Vec<u8>) -> Result<Extraction, SkipReason> {
        match self {
            Self::Text => decode(&bytes).map(Extraction::of_one_page),
            Self::Html => decode(&bytes)
                .and_then(|html| html_text(&html))
                .map(Extraction::of_one_page),
            Self::Pdf => pdf::page_texts(bytes),
            Self::Office(format) => office::page_texts(bytes, format),
            Self::OpenDocument => opendocument::page_texts(bytes),
            Self::Epub => epub::page_texts(bytes),
        }
    }
}

/// Whether a page of the file at `path` can be rendered to an image, which only a
/// PDF's can.
pub fn renders_pages(path: &Path) -> bool {
    Format::of(path) == Format::Pdf
}

/// The text of the regular file at `path` page by page, or why it has none; a
/// file larger than `max_file_bytes` is not read. Blocks on the filesystem.
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

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use office_oxide::docx::write::DocxWriter;
    use office_oxide::pptx::write::PptxWriter;
    use office_oxide::xlsx::write::{CellData, XlsxWriter};

    use super::*;
    use crate::archive::{MAX_ENTRIES, MAX_ENTRY_BYTES};
    use crate::protocol::MAX_TEXT_BYTES;
    use crate::sample_archive::archive_of;
    use crate::sample_pdf::{EMPTY_PAGE, IMAGE_PAGE, pdf_of, text_page};

    const MEBIBYTE: u64 = 1024 * 1024;

    const OPENDOCUMENT_MANIFEST: &str = r#"<manifest:manifest><manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/></manifest:manifest>"#;

    const ENCRYPTED_MANIFEST: &str = r#"<manifest:manifest><manifest:file-entry manifest:full-path="content.xml"><manifest:encryption-data manifest:checksum="x"/></manifest:file-entry></manifest:manifest>"#;

    const TEXT_BODY: &str = r#"<office:text>
        <text:tracked-changes><text:changed-region><text:deletion>
          <text:p>Struck out.</text:p>
        </text:deletion></text:changed-region></text:tracked-changes>
        <text:h text:outline-level="2">Minutes</text:h>
        <text:p>Present:<text:s text:c="2"/>Ann &amp; <text:span>Bo</text:span><text:note
          ><text:note-citation>1</text:note-citation><text:note-body
          ><text:p>By phone.</text:p></text:note-body></text:note>,
          then<text:line-break/>Cy.</text:p>
        <table:table>
          <table:table-row>
            <table:table-cell><text:p>Item</text:p></table:table-cell>
            <table:table-cell><text:p>Owner</text:p></table:table-cell>
          </table:table-row>
          <table:table-row>
            <table:table-cell><text:p>Budget</text:p><text:p>a|b</text:p></table:table-cell>
            <table:table-cell><text:p>Ann</text:p></table:table-cell>
          </table:table-row>
        </table:table>
      </office:text>"#;

    const SPREADSHEET_BODY: &str = r#"<office:spreadsheet>
        <table:table table:name="Costs &amp; totals">
          <table:table-row>
            <table:table-cell><text:p>Item</text:p></table:table-cell>
            <table:table-cell table:number-columns-repeated="2"/>
            <table:table-cell office:value="3"><text:p>3</text:p></table:table-cell>
            <table:table-cell table:number-columns-repeated="16380"/>
          </table:table-row>
          <table:table-row table:number-rows-repeated="2">
            <table:table-cell table:number-columns-repeated="2"><text:p>x</text:p></table:table-cell>
          </table:table-row>
          <table:table-row table:number-rows-repeated="1048573">
            <table:table-cell table:number-columns-repeated="16384"/>
          </table:table-row>
        </table:table>
        <table:table table:name="Unused">
          <table:table-row table:number-rows-repeated="1048576">
            <table:table-cell table:number-columns-repeated="16384"/>
          </table:table-row>
        </table:table>
      </office:spreadsheet>"#;

    const PRESENTATION_BODY: &str = r#"<office:presentation>
        <draw:page draw:name="page1">
          <draw:frame><draw:text-box><text:p>Plan</text:p></draw:text-box></draw:frame>
          <draw:frame><draw:text-box><text:p>Ship it</text:p></draw:text-box></draw:frame>
        </draw:page>
        <draw:page draw:name="page2"><draw:frame><draw:image/></draw:frame></draw:page>
        <draw:page draw:name="page3">
          <draw:frame><draw:text-box><text:p>Questions</text:p></draw:text-box></draw:frame>
        </draw:page>
      </office:presentation>"#;

    const REPEATED_PAST_ANY_LIMIT_BODY: &str = r#"<office:spreadsheet><table:table>
        <table:table-row table:number-rows-repeated="1073741824">
          <table:table-cell table:number-columns-repeated="16384"><text:p>again</text:p></table:table-cell>
        </table:table-row>
      </table:table></office:spreadsheet>"#;

    const EPUB_CONTAINER: &str = r#"<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
        <rootfiles><rootfile full-path="OEBPS/book.opf"/></rootfiles></container>"#;

    const EPUB_PACKAGE: &str = r#"<package xmlns="http://www.idpf.org/2007/opf">
        <manifest>
          <item id="two" href="text/part%202.xhtml" media-type="application/xhtml+xml"/>
          <item id="one" href="text/one.xhtml" media-type="application/xhtml+xml"/>
          <item id="art" href="art.svg" media-type="image/svg+xml"/>
          <item id="end" href="../end.xhtml" media-type="application/xhtml+xml"/>
        </manifest>
        <spine><itemref idref="art"/><itemref idref="one"/><itemref idref="two"/>
          <itemref idref="end"/></spine>
      </package>"#;

    const EPUB_FIRST_CHAPTER: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
      <html xmlns="http://www.w3.org/1999/xhtml"><head><title/><script src="reader.js"/></head>
      <body><h1>One</h1><p>First<br/>chapter.</p></body></html>"#;

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
        pages(&[text])
    }

    fn skipped(reason: SkipReason) -> Extraction {
        Extraction::Skipped { reason }
    }

    fn pages(pages: &[&str]) -> Extraction {
        Extraction::Pages {
            pages: pages.iter().map(|page| (*page).to_owned()).collect(),
            truncated: false,
        }
    }

    fn opendocument_of(manifest: &str, body: &str) -> Vec<u8> {
        let content = format!(
            "<office:document-content><office:body>{body}</office:body></office:document-content>"
        );
        archive_of(&[
            ("META-INF/manifest.xml", manifest.as_bytes()),
            ("content.xml", content.as_bytes()),
        ])
    }

    fn page_texts(extraction: &Extraction) -> &[String] {
        match extraction {
            Extraction::Pages { pages, .. } => pages,
            Extraction::Skipped { reason } => panic!("the file has text: {reason}"),
        }
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
    fn a_pdf_yields_each_pages_text_and_an_empty_page_where_there_is_none() {
        let scanned = extraction_of("scan.PDF", &pdf_of(&[IMAGE_PAGE, IMAGE_PAGE]), 4096);
        assert_eq!(
            scanned,
            Extraction::Pages {
                pages: vec![String::new(), String::new()],
                truncated: false,
            }
        );

        let pages = [&text_page("Minutes"), IMAGE_PAGE, &text_page("Appendix")];
        let mixed = extraction_of("minutes.pdf", &pdf_of(&pages), 4096);
        let Extraction::Pages { pages, truncated } = &mixed else {
            panic!("a PDF comes back as pages: {mixed:?}");
        };
        assert!(!truncated);
        let [minutes, scan, appendix] = &pages[..] else {
            panic!("a page of the file is a page of its text: {pages:?}");
        };
        assert!(minutes.contains("Minutes"), "{pages:?}");
        assert_eq!(scan, "");
        assert!(appendix.contains("Appendix"), "{pages:?}");
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
    fn an_opendocument_file_yields_markdown_with_a_page_for_each_sheet_or_slide() {
        let cases = [
            (
                "minutes.odt",
                TEXT_BODY,
                pages(&[
                    "## Minutes\n\nPresent:  Ann & Bo, then\nCy.\n\nBy phone.\n\n\
                    | Item | Owner |\n| --- | --- |\n| Budget a\\|b | Ann |\n",
                ]),
            ),
            (
                "costs.ODS",
                SPREADSHEET_BODY,
                pages(&[
                    "## Costs & totals\n\n| Item |  |  | 3 |\n| --- | --- | --- | --- |\n\
                     | x | x |\n| x | x |\n",
                    "",
                ]),
            ),
            (
                "plan.odp",
                PRESENTATION_BODY,
                pages(&["Plan\n\nShip it\n", "", "Questions\n"]),
            ),
        ];
        for (name, body, expected) in cases {
            let file = opendocument_of(OPENDOCUMENT_MANIFEST, body);
            assert_eq!(extraction_of(name, &file, MEBIBYTE), expected, "{name}");
        }
        let locked = opendocument_of(ENCRYPTED_MANIFEST, TEXT_BODY);
        assert_eq!(
            extraction_of("locked.odt", &locked, MEBIBYTE),
            skipped(SkipReason::PasswordProtected)
        );
    }

    #[test]
    fn a_repeat_count_expands_a_spreadsheet_no_further_than_the_text_kept() {
        let file = opendocument_of(OPENDOCUMENT_MANIFEST, REPEATED_PAST_ANY_LIMIT_BODY);
        let Extraction::Pages { pages, truncated } = extraction_of("again.ods", &file, MEBIBYTE)
        else {
            panic!("the rows that fit are kept");
        };
        assert!(truncated);
        assert_eq!(pages.iter().map(String::len).sum::<usize>(), MAX_TEXT_BYTES);
    }

    #[test]
    fn an_epub_yields_its_chapters_in_spine_order_each_as_a_page() {
        let chapter = |text: &str| format!("<html><body><p>{text}</p></body></html>");
        let book = archive_of(&[
            ("mimetype", b"application/epub+zip"),
            ("META-INF/container.xml", EPUB_CONTAINER.as_bytes()),
            ("OEBPS/book.opf", EPUB_PACKAGE.as_bytes()),
            ("OEBPS/art.svg", b"<svg/>"),
            ("OEBPS/text/part 2.xhtml", chapter("Second.").as_bytes()),
            ("OEBPS/text/one.xhtml", EPUB_FIRST_CHAPTER.as_bytes()),
            ("end.xhtml", chapter("Last.").as_bytes()),
        ]);
        assert_eq!(
            extraction_of("book.epub", &book, MEBIBYTE),
            pages(&["# One\n\nFirst\nchapter.\n", "Second.\n", "Last.\n"])
        );
    }

    #[test]
    fn an_office_file_yields_markdown_with_a_page_for_each_slide_or_sheet() {
        let mut letter = Cursor::new(Vec::new());
        DocxWriter::new()
            .add_heading("Lease", 1)
            .add_paragraph("Renewal terms.")
            .write_to(&mut letter)
            .expect("docx");
        let letter = extraction_of("lease.DOCX", letter.get_ref(), MEBIBYTE);
        let [text] = page_texts(&letter) else {
            panic!("a DOCX is one page: {letter:?}");
        };
        assert!(text.contains("# Lease\n\nRenewal terms."), "{text}");

        let mut deck = PptxWriter::new();
        deck.add_slide().set_title("Plan").add_text("Ship it");
        deck.add_slide().add_text("Questions");
        let mut slides = Cursor::new(Vec::new());
        deck.write_to(&mut slides).expect("pptx");
        let slides = extraction_of("plan.pptx", slides.get_ref(), MEBIBYTE);
        let [plan, questions] = page_texts(&slides) else {
            panic!("a slide is a page: {slides:?}");
        };
        assert!(plan.contains("Plan") && plan.contains("Ship it"), "{plan}");
        assert!(questions.contains("Questions"), "{questions}");

        let mut workbook = XlsxWriter::new();
        workbook.add_sheet("Costs").add_row(vec![
            CellData::String("Rent".into()),
            CellData::Number(900.0),
        ]);
        workbook.add_sheet("Unused");
        let mut sheets = Cursor::new(Vec::new());
        workbook.write_to(&mut sheets).expect("xlsx");
        let sheets = extraction_of("costs.xlsx", sheets.get_ref(), MEBIBYTE);
        let [costs, unused] = page_texts(&sheets) else {
            panic!("a sheet is a page: {sheets:?}");
        };
        assert!(
            costs.contains("## Costs") && costs.contains("| Rent | 900 |"),
            "{costs}"
        );
        assert_eq!(unused, "");

        assert_eq!(
            extraction_of(
                "locked.xlsx",
                b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1....",
                MEBIBYTE
            ),
            skipped(SkipReason::PasswordProtected)
        );
    }

    #[test]
    fn an_archive_past_a_decompression_limit_is_skipped_whatever_its_format() {
        let names: Vec<String> = (0..=MAX_ENTRIES).map(|entry| entry.to_string()).collect();
        let entries: Vec<(&str, &[u8])> =
            names.iter().map(|name| (name.as_str(), &b""[..])).collect();
        let crowded = archive_of(&entries);
        for name in [
            "a.docx", "a.pptx", "a.xlsx", "a.odt", "a.ods", "a.odp", "a.epub",
        ] {
            assert_eq!(
                extraction_of(name, &crowded, MEBIBYTE),
                skipped(SkipReason::TooLarge),
                "{name}"
            );
        }

        let filler = " ".repeat(usize::try_from(MAX_ENTRY_BYTES).expect("fits"));
        let mut bomb = Cursor::new(Vec::new());
        DocxWriter::new()
            .add_paragraph(&filler)
            .write_to(&mut bomb)
            .expect("docx");
        assert!(bomb.get_ref().len() < filler.len() / 100);
        assert_eq!(
            extraction_of("bomb.docx", bomb.get_ref(), MEBIBYTE),
            skipped(SkipReason::TooLarge)
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
