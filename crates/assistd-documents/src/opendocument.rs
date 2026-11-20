//! ODF files: the text of `content.xml` as Markdown, an ODT as one
//! page, an ODS sheet by sheet and an ODP slide by slide.

use std::iter;
use std::mem;

use quick_xml::Reader;
use quick_xml::escape::resolve_predefined_entity;
use quick_xml::events::{BytesRef, BytesStart, Event};

use crate::archive::Archive;
use crate::protocol::{Extraction, MAX_TEXT_BYTES, SkipReason};
use crate::text::decode;
use crate::xml::attribute;

const CONTENT_ENTRY: &str = "content.xml";

const MANIFEST_ENTRY: &str = "META-INF/manifest.xml";

/// The manifest element an encrypted entry is listed with.
const ENCRYPTION_ELEMENT: &str = "manifest:encryption-data";

/// Elements whose text is not the document's own: what a tracked change
/// removed, comments, the messages of a cell's validation, a footnote's mark
/// and a drawing's alternative text.
const SKIPPED_ELEMENTS: [&str; 7] = [
    "text:tracked-changes",
    "table:tracked-changes",
    "table:content-validations",
    "office:annotation",
    "text:note-citation",
    "svg:title",
    "svg:desc",
];

/// Bytes one file's content may be written out to before conversion stops:
/// twice the text kept, since a cell's text is counted again in its row. This
/// is what bounds a row, cell or space repeated by a count.
const MAX_WRITTEN_BYTES: usize = 2 * MAX_TEXT_BYTES;

const DEEPEST_HEADING: usize = 6;

/// A table cell's text and how many columns in a row hold it.
struct Cell {
    text: String,
    repeat: usize,
}

/// An element being read that text or blocks are gathered into.
enum Open {
    Paragraph {
        heading_level: Option<usize>,
        text: String,
    },
    /// A table's rows so far as Markdown, under `heading` when it is a sheet.
    Table {
        heading: Option<String>,
        rows: String,
    },
    Row {
        cells: Vec<Cell>,
        repeat: usize,
    },
    Cell(Cell),
}

/// The pages of one `content.xml`, built as its elements open and close.
struct Content {
    pages: Vec<String>,
    /// Innermost last.
    open: Vec<Open>,
    /// Blocks met inside a paragraph, written after it.
    deferred: Vec<String>,
    in_spreadsheet: bool,
    /// Bytes that may still be written, out of [`MAX_WRITTEN_BYTES`].
    room: usize,
    /// Whether the room ran out before the content did.
    cut: bool,
}

impl Content {
    /// The pages of `xml`, elements matched by the prefixes every writer
    /// gives them.
    fn read(xml: &str) -> Result<Self, quick_xml::Error> {
        let mut reader = Reader::from_str(xml);
        let mut content = Self {
            pages: Vec::new(),
            open: Vec::new(),
            deferred: Vec::new(),
            in_spreadsheet: false,
            room: MAX_WRITTEN_BYTES,
            cut: false,
        };
        while !content.cut {
            match reader.read_event()? {
                Event::Eof => break,
                Event::Start(element) if SKIPPED_ELEMENTS.contains(&element.name().as_ref()) => {
                    reader.read_to_end(element.name())?;
                }
                Event::Start(element) => content.open(&element)?,
                Event::Empty(element) => {
                    content.open(&element)?;
                    content.close(element.name().as_ref());
                }
                Event::End(element) => content.close(element.name().as_ref()),
                Event::Text(text) => content.write_text(&text),
                Event::GeneralRef(reference) => content.write_text(&referenced_text(&reference)?),
                _ => {}
            }
        }
        content.close_all();
        Ok(content)
    }

    fn open(&mut self, element: &BytesStart<'_>) -> Result<(), quick_xml::Error> {
        match element.name().as_ref() {
            "office:spreadsheet" => self.in_spreadsheet = true,
            "draw:page" => self.pages.push(String::new()),
            "text:h" => {
                let level = count(element, "text:outline-level")?.clamp(1, DEEPEST_HEADING);
                self.open_paragraph(Some(level));
            }
            "text:p" => self.open_paragraph(None),
            "text:s" => {
                let spaces = count(element, "text:c")?;
                self.write_spaces(spaces);
            }
            "text:tab" => self.write_spaces(1),
            "text:line-break" => self.write_text_as_is("\n"),
            "table:table" => self.open_table(element)?,
            "table:table-row" => self.open.push(Open::Row {
                cells: Vec::new(),
                repeat: count(element, "table:number-rows-repeated")?,
            }),
            "table:table-cell" | "table:covered-table-cell" => self.open.push(Open::Cell(Cell {
                text: String::new(),
                repeat: count(element, "table:number-columns-repeated")?,
            })),
            _ => {}
        }
        Ok(())
    }

    fn close(&mut self, element: &str) {
        match element {
            "text:p" | "text:h" => self.close_paragraph(),
            "table:table-cell" | "table:covered-table-cell" => self.close_cell(),
            "table:table-row" => self.close_row(),
            "table:table" => self.close_table(),
            _ => {}
        }
    }

    /// Close what is still open, so text read before the room ran out is kept.
    fn close_all(&mut self) {
        while let Some(innermost) = self.open.last() {
            match innermost {
                Open::Paragraph { .. } => self.close_paragraph(),
                Open::Table { .. } => self.close_table(),
                Open::Row { .. } => self.close_row(),
                Open::Cell(_) => self.close_cell(),
            }
        }
    }

    fn open_paragraph(&mut self, heading_level: Option<usize>) {
        self.open.push(Open::Paragraph {
            heading_level,
            text: String::new(),
        });
    }

    fn close_paragraph(&mut self) {
        let Some(Open::Paragraph {
            heading_level,
            text,
        }) = self.open.pop()
        else {
            return;
        };
        let text = text.trim();
        let block = match heading_level {
            _ if text.is_empty() => String::new(),
            Some(level) => format!("{} {text}", "#".repeat(level)),
            None => text.to_owned(),
        };
        self.write_block(block);
        if !matches!(self.block_holder(), Some(Open::Paragraph { .. })) {
            for block in mem::take(&mut self.deferred) {
                self.write_block(block);
            }
        }
    }

    /// Open a table, which in a spreadsheet's body is a sheet: a page of its
    /// own under its name.
    fn open_table(&mut self, table: &BytesStart<'_>) -> Result<(), quick_xml::Error> {
        let in_table = self
            .open
            .iter()
            .any(|open| matches!(open, Open::Table { .. }));
        let heading = if self.in_spreadsheet && !in_table {
            self.pages.push(String::new());
            attribute(table, "table:name")?.filter(|name| !name.is_empty())
        } else {
            None
        };
        self.open.push(Open::Table {
            heading,
            rows: String::new(),
        });
        Ok(())
    }

    fn close_table(&mut self) {
        let Some(Open::Table { heading, rows }) = self.open.pop() else {
            return;
        };
        let block = match heading {
            Some(name) if !rows.is_empty() => format!("## {name}\n\n{rows}"),
            _ => rows,
        };
        self.write_block(block);
    }

    /// Add the row to its table as a Markdown row, as often as it repeats,
    /// without the empty cells that end it; a row with no other cell is left out.
    fn close_row(&mut self) {
        let Some(Open::Row { mut cells, repeat }) = self.open.pop() else {
            return;
        };
        let filled = cells
            .iter()
            .rposition(|cell| !cell.text.is_empty())
            .map_or(0, |last| last + 1);
        cells.truncate(filled);
        if cells.is_empty() {
            return;
        }
        let mut line = String::from("|");
        let mut columns = 0;
        for cell in &cells {
            let copies = self.afford(cell.repeat, cell.text.len() + " |".len() + 1);
            line.extend(iter::repeat_n([" ", &cell.text, " |"], copies).flatten());
            columns += copies;
        }
        let copies = self.afford(repeat, line.len() + 1);
        let Some(Open::Table { rows, .. }) = self.open.last_mut() else {
            return;
        };
        for _ in 0..copies {
            if rows.is_empty() {
                rows.extend([line.as_str(), "\n|", &" --- |".repeat(columns)]);
            } else {
                rows.extend(["\n", line.as_str()]);
            }
        }
    }

    fn close_cell(&mut self) {
        let Some(Open::Cell(cell)) = self.open.pop() else {
            return;
        };
        if let Some(Open::Row { cells, .. }) = self.open.last_mut() {
            cells.push(cell);
        }
    }

    /// Add `text` to the paragraph being read, each run of white space as one
    /// space and none at its start.
    fn write_text(&mut self, text: &str) {
        let Some(Open::Paragraph { text: written, .. }) = self.open.last_mut() else {
            return;
        };
        let fits = text.floor_char_boundary(self.room);
        self.cut |= fits < text.len();
        self.room -= fits;
        for character in text[..fits].chars() {
            if !matches!(character, ' ' | '\t' | '\r' | '\n') {
                written.push(character);
            } else if !written.is_empty() && !written.ends_with([' ', '\n']) {
                written.push(' ');
            }
        }
    }

    fn write_spaces(&mut self, spaces: usize) {
        let spaces = self.afford(spaces, 1);
        self.write_text_as_is(&" ".repeat(spaces));
    }

    fn write_text_as_is(&mut self, text: &str) {
        if let Some(Open::Paragraph { text: written, .. }) = self.open.last_mut() {
            written.push_str(text);
        }
    }

    /// Write `block` where the innermost open element that holds blocks puts
    /// it: on one line of a cell, after a paragraph, or else on the page.
    fn write_block(&mut self, block: String) {
        if block.is_empty() {
            return;
        }
        match self.block_holder() {
            Some(Open::Cell(cell)) => {
                if !cell.text.is_empty() {
                    cell.text.push(' ');
                }
                cell.text
                    .push_str(&block.replace('\n', " ").replace('|', "\\|"));
            }
            Some(_) => self.deferred.push(block),
            None => {
                if self.pages.is_empty() {
                    self.pages.push(String::new());
                }
                let page = self.pages.last_mut().expect("a page was just added");
                if !page.is_empty() {
                    page.push('\n');
                }
                page.extend([block.as_str(), "\n"]);
            }
        }
    }

    fn block_holder(&mut self) -> Option<&mut Open> {
        self.open
            .iter_mut()
            .rev()
            .find(|open| matches!(open, Open::Cell(_) | Open::Paragraph { .. }))
    }

    /// How many of `copies` copies of `bytes` bytes fit the room left, which
    /// they are then taken from.
    fn afford(&mut self, copies: usize, bytes: usize) -> usize {
        let affordable = copies.min(self.room / bytes);
        self.cut |= affordable < copies;
        self.room -= affordable * bytes;
        affordable
    }
}

/// The text of the ODF file in `bytes`: a text document as one page, a
/// spreadsheet's sheets and a presentation's slides each as a page.
pub(crate) fn page_texts(bytes: Vec<u8>) -> Result<Extraction, SkipReason> {
    let mut archive = Archive::open(bytes)?;
    let manifest = decode(&archive.read(MANIFEST_ENTRY)?)?;
    if lists_encryption(&manifest)? {
        return Err(SkipReason::PasswordProtected);
    }
    let content = Content::read(&decode(&archive.read(CONTENT_ENTRY)?)?)?;
    let mut extraction = Extraction::of_pages(content.pages);
    if let Extraction::Pages { truncated, .. } = &mut extraction {
        *truncated |= content.cut;
    }
    Ok(extraction)
}

fn lists_encryption(manifest: &str) -> Result<bool, quick_xml::Error> {
    let mut reader = Reader::from_str(manifest);
    loop {
        match reader.read_event()? {
            Event::Eof => return Ok(false),
            Event::Start(element) | Event::Empty(element)
                if element.name().as_ref() == ENCRYPTION_ELEMENT =>
            {
                return Ok(true);
            }
            _ => {}
        }
    }
}

/// The number in `element`'s attribute `name`, one when it has none.
fn count(element: &BytesStart<'_>, name: &str) -> Result<usize, quick_xml::Error> {
    let number = element.try_get_attribute(name)?;
    Ok(number
        .and_then(|number| number.value.parse().ok())
        .unwrap_or(1))
}

/// The character a character reference names or the text a predefined entity
/// stands for; nothing for an entity the file would have to declare.
fn referenced_text(reference: &BytesRef<'_>) -> Result<String, quick_xml::Error> {
    Ok(match reference.resolve_char_ref()? {
        Some(character) => character.to_string(),
        None => resolve_predefined_entity(reference)
            .unwrap_or_default()
            .to_owned(),
    })
}
