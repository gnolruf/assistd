//! Builds small PDFs for tests, one US Letter page per content stream.

use std::fmt::Write;

/// A page that draws a black 1×1 image stretched over a square, and no text.
pub(crate) const IMAGE_PAGE: &str = "q 300 0 0 300 72 400 cm /Im1 Do Q";

/// A page that draws nothing.
pub(crate) const EMPTY_PAGE: &str = "";

const CATALOG: &str = "<< /Type /Catalog /Pages 2 0 R >>";
const FONT: &str = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>";
const IMAGE: &str = "<< /Type /XObject /Subtype /Image /Width 1 /Height 1 \
    /ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /ASCIIHexDecode /Length 3 >>\n\
    stream\n00>\nendstream";
const FIRST_PAGE_OBJECT: usize = 5;

/// The content stream of a page that shows `line` as one line of Helvetica.
pub(crate) fn text_page(line: &str) -> String {
    format!("BT /F1 24 Tf 72 700 Td ({line}) Tj ET")
}

/// A PDF with one page per entry of `contents`, each a page's content stream,
/// which may show text in `/F1` and draw the image `/Im1`.
pub(crate) fn pdf_of(contents: &[&str]) -> Vec<u8> {
    let page_object = |index: usize| FIRST_PAGE_OBJECT + 2 * index;
    let kids: String = (0..contents.len())
        .map(|index| format!("{} 0 R ", page_object(index)))
        .collect();
    let mut objects = vec![
        CATALOG.to_owned(),
        format!(
            "<< /Type /Pages /Kids [ {kids}] /Count {} >>",
            contents.len()
        ),
        FONT.to_owned(),
        IMAGE.to_owned(),
    ];
    for (index, content) in contents.iter().enumerate() {
        objects.push(format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
             /Resources << /Font << /F1 3 0 R >> /XObject << /Im1 4 0 R >> >> \
             /Contents {} 0 R >>",
            page_object(index) + 1
        ));
        objects.push(format!(
            "<< /Length {} >>\nstream\n{content}\nendstream",
            content.len()
        ));
    }
    file_of(&objects)
}

/// `objects`, numbered from one, as a PDF file with its cross-reference table.
fn file_of(objects: &[String]) -> Vec<u8> {
    let mut file = String::from("%PDF-1.4\n");
    let mut offsets = Vec::with_capacity(objects.len());
    for (index, object) in objects.iter().enumerate() {
        offsets.push(file.len());
        let _ = write!(file, "{} 0 obj\n{object}\nendobj\n", index + 1);
    }
    let table_offset = file.len();
    let _ = write!(file, "xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1);
    for offset in offsets {
        let _ = writeln!(file, "{offset:010} 00000 n ");
    }
    let _ = write!(
        file,
        "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{table_offset}\n%%EOF\n",
        objects.len() + 1
    );
    file.into_bytes()
}
