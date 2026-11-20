//! EPUB books: each chapter the package's spine names, in reading order, as one
//! page of text.

use std::collections::HashMap;
use std::path::Path;

use quick_xml::Reader;
use quick_xml::events::{BytesStart, Event};
use url::Url;

use crate::archive::Archive;
use crate::protocol::{Extraction, SkipReason};
use crate::text::{decode, xhtml_text};
use crate::xml::attribute;

/// The entry that names the book's package document.
const CONTAINER_ENTRY: &str = "META-INF/container.xml";

const CHAPTER_MEDIA_TYPES: [&str; 2] = ["application/xhtml+xml", "text/html"];

/// The text of each chapter of the EPUB in `bytes`, read only as far as the
/// text that is kept.
pub(crate) fn page_texts(bytes: Vec<u8>) -> Result<Extraction, SkipReason> {
    let mut archive = Archive::open(bytes)?;
    let container = decode(&archive.read(CONTAINER_ENTRY)?)?;
    let package_entry = package_entry(&container)?.ok_or_else(|| SkipReason::ExtractorFailed {
        message: "the container names no package document".into(),
    })?;
    let package = decode(&archive.read(&package_entry)?)?;
    let chapters = chapter_entries(&package_entry, &package)?;
    Extraction::try_of_pages(
        chapters
            .iter()
            .map(|entry| xhtml_text(&decode(&archive.read(entry)?)?)),
    )
}

fn package_entry(container: &str) -> Result<Option<String>, quick_xml::Error> {
    let mut reader = Reader::from_str(container);
    loop {
        match reader.read_event()? {
            Event::Eof => return Ok(None),
            Event::Start(element) | Event::Empty(element)
                if element.local_name().as_ref() == "rootfile" =>
            {
                return attribute(&element, "full-path");
            }
            _ => {}
        }
    }
}

/// The entries of the chapters `package`'s spine lists, in its order; an item
/// that is not a chapter, or whose address is not an entry's, is left out.
fn chapter_entries(package_entry: &str, package: &str) -> Result<Vec<String>, quick_xml::Error> {
    let mut reader = Reader::from_str(package);
    let mut chapter_hrefs = HashMap::new();
    let mut spine = Vec::new();
    loop {
        match reader.read_event()? {
            Event::Eof => break,
            Event::Start(element) | Event::Empty(element) => match element.local_name().as_ref() {
                "item" => chapter_hrefs.extend(chapter_item(&element)?),
                "itemref" => spine.extend(attribute(&element, "idref")?),
                _ => {}
            },
            _ => {}
        }
    }
    Ok(spine
        .iter()
        .filter_map(|id| chapter_hrefs.get(id))
        .filter_map(|href| entry_name(package_entry, href))
        .collect())
}

/// The id and address of the manifest item `item`, when it is a chapter.
fn chapter_item(item: &BytesStart<'_>) -> Result<Option<(String, String)>, quick_xml::Error> {
    let is_chapter = attribute(item, "media-type")?
        .is_some_and(|media_type| CHAPTER_MEDIA_TYPES.contains(&media_type.as_str()));
    if !is_chapter {
        return Ok(None);
    }
    Ok(attribute(item, "id")?.zip(attribute(item, "href")?))
}

/// The archive entry the relative URL `href` names from the package document
/// at `package_entry`, resolved as a browser resolves a link.
fn entry_name(package_entry: &str, href: &str) -> Option<String> {
    let package = Url::from_file_path(Path::new("/").join(package_entry)).ok()?;
    let chapter = package.join(href).ok()?.to_file_path().ok()?;
    chapter.strip_prefix("/").ok()?.to_str().map(str::to_owned)
}
