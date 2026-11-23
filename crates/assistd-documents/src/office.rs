//! Office Open XML files as Markdown: a DOCX as one page, a PPTX slide by slide
//! and an XLSX sheet by sheet.

use std::io::Cursor;

use office_oxide::core::Error as PackageError;
use office_oxide::docx::DocxError;
use office_oxide::limits::Limits;
use office_oxide::options::ParseOptions;
use office_oxide::pptx::PptxError;
use office_oxide::xlsx::XlsxError;
use office_oxide::{Document, DocumentFormat, DocumentIR, OfficeError};

use crate::archive::{MAX_ARCHIVE_BYTES, MAX_ENTRIES, MAX_ENTRY_BYTES};
use crate::protocol::{PageTexts, SkipReason};

/// What a file starts with when it is an OLE container, which is how Office
/// stores a package it has encrypted.
const OLE_MAGIC: [u8; 8] = [0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OfficeFormat {
    Docx,
    Pptx,
    Xlsx,
}

impl From<OfficeFormat> for DocumentFormat {
    fn from(format: OfficeFormat) -> Self {
        match format {
            OfficeFormat::Docx => Self::Docx,
            OfficeFormat::Pptx => Self::Pptx,
            OfficeFormat::Xlsx => Self::Xlsx,
        }
    }
}

/// The text of the `format` file in `bytes` as Markdown: a DOCX whole, a PPTX
/// or XLSX with a page for each slide or sheet, empty for one that holds nothing.
pub(crate) fn page_texts(bytes: Vec<u8>, format: OfficeFormat) -> Result<PageTexts, SkipReason> {
    if bytes.starts_with(&OLE_MAGIC) {
        return Err(SkipReason::PasswordProtected);
    }
    let document = Document::from_reader_with(Cursor::new(bytes), format.into(), &parse_options())
        .map_err(|error| skip_reason(&error))?;
    if format == OfficeFormat::Docx {
        return Ok(PageTexts::of_one_page(document.to_markdown()));
    }
    let texts = document.to_ir().sections.into_iter().map(|section| {
        if section.elements.is_empty() {
            return String::new();
        }
        let page = DocumentIR {
            sections: vec![section],
            ..DocumentIR::default()
        };
        page.to_markdown()
    });
    Ok(PageTexts::of_pages(texts))
}

/// The library's limits for a file nobody vouches for, with the archive held to
/// the limits every other archive is read under and pictures left compressed.
fn parse_options() -> ParseOptions {
    let mut limits = Limits::strict();
    limits.package_entries = MAX_ENTRIES;
    limits.part_bytes = MAX_ENTRY_BYTES;
    limits.package_bytes = MAX_ARCHIVE_BYTES;
    let mut options = ParseOptions::default();
    options.limits = limits;
    options.lazy_pictures = true;
    options
}

fn skip_reason(error: &OfficeError) -> SkipReason {
    match error {
        OfficeError::Core(limit)
        | OfficeError::Docx(DocxError::Core(limit))
        | OfficeError::Pptx(PptxError::Core(limit))
        | OfficeError::Xlsx(XlsxError::Core(limit))
            if matches!(
                limit,
                PackageError::PackageLimit(_) | PackageError::DecompressionLimit { .. }
            ) =>
        {
            SkipReason::TooLarge
        }
        _ => SkipReason::ExtractorFailed {
            message: error.to_string(),
        },
    }
}
