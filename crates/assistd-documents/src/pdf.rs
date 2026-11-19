//! PDF files: each page's text as Markdown, and one page rendered to an image
//! for a model to transcribe.

use std::io::Cursor;

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use image::{DynamicImage, ImageFormat, RgbaImage, imageops};
use num_traits::ToPrimitive;
use pdf_oxide::PdfDocument;
use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::rendering::{RenderOptions, render_page_fit};

use crate::protocol::{Extraction, MAX_PAGE_IMAGE_BYTES, RenderedPage, SkipReason};

/// Pixels per inch a page is rendered at.
const RENDER_DPI: f32 = 150.0;

/// PDF units per inch.
const POINTS_PER_INCH: f32 = 72.0;

/// Longest edge of a rendered page, in pixels; a larger page is rendered at
/// fewer pixels per inch.
const MAX_RENDER_EDGE_PX: f32 = 2048.0;

/// Side, in pixels, of the squares a rendered page is averaged over before it is
/// checked for marks, which evens out a scanner's noise.
const BLANK_CELL_PX: u32 = 8;

/// The widest gap between a blank page's darkest and lightest square, out of 255.
const BLANK_LUMA_SPREAD: u8 = 32;

/// The text of each page of the PDF in `bytes` as Markdown, empty for a page
/// without a text layer.
pub(crate) fn page_texts(bytes: Vec<u8>) -> Result<Extraction, SkipReason> {
    let document = open(bytes)?;
    let page_count = document.page_count().map_err(failed)?;
    let options = ConversionOptions::default();
    let texts = (0..page_count).map(|page| {
        let text = document.to_markdown(page, &options)?;
        Ok(Some(text)
            .filter(|text| !text.trim().is_empty())
            .unwrap_or_default())
    });
    Extraction::of_pages(texts).map_err(failed)
}

/// Page `page` of the PDF in `bytes`, counted from zero, as a PNG, or blank
/// when it shows only its background.
pub(crate) fn render_page(bytes: Vec<u8>, page: u32) -> RenderedPage {
    let rendered = open(bytes).and_then(|document| {
        let pixels = page_pixels(&document, page)?;
        if is_blank(&pixels) {
            return Ok(RenderedPage::Blank);
        }
        encode_png(&pixels).map(|png| RenderedPage::Image {
            png_base64: STANDARD.encode(png),
        })
    });
    rendered.unwrap_or_else(|reason| RenderedPage::Failed { reason })
}

/// The PDF in `bytes`, opened with the empty password when it is encrypted.
fn open(bytes: Vec<u8>) -> Result<PdfDocument, SkipReason> {
    let document = PdfDocument::from_bytes(bytes).map_err(failed)?;
    if document.is_encrypted() && !document.authenticate(b"").map_err(failed)? {
        return Err(SkipReason::PasswordProtected);
    }
    Ok(document)
}

fn page_pixels(document: &PdfDocument, page: u32) -> Result<DynamicImage, SkipReason> {
    let page = usize::try_from(page).map_err(|error| failure(error.to_string()))?;
    let media_box = document.get_page_info(page).map_err(failed)?.media_box;
    let edge = render_edge_px(media_box.width.max(media_box.height))
        .ok_or_else(|| failure("the page has no size"))?;
    let options = RenderOptions::default().as_raw();
    let rendered = render_page_fit(document, page, edge, edge, &options).map_err(failed)?;
    RgbaImage::from_raw(rendered.width, rendered.height, rendered.data)
        .map(DynamicImage::ImageRgba8)
        .ok_or_else(|| failure("the renderer returned fewer pixels than the page has"))
}

/// The pixels a page's longest edge of `long_edge_points` is rendered to:
/// [`RENDER_DPI`], held to [`MAX_RENDER_EDGE_PX`]. `None` for a page with no length.
fn render_edge_px(long_edge_points: f32) -> Option<u32> {
    (long_edge_points * RENDER_DPI / POINTS_PER_INCH)
        .min(MAX_RENDER_EDGE_PX)
        .round()
        .to_u32()
        .filter(|edge| *edge > 0)
}

/// Whether `page` shows nothing but its background: averaged over squares of
/// [`BLANK_CELL_PX`], no part is more than [`BLANK_LUMA_SPREAD`] from another.
fn is_blank(page: &DynamicImage) -> bool {
    let luma = page.to_luma8();
    let cells = imageops::thumbnail(
        &luma,
        (luma.width() / BLANK_CELL_PX).max(1),
        (luma.height() / BLANK_CELL_PX).max(1),
    );
    let lightest = cells.iter().copied().max().unwrap_or_default();
    let darkest = cells.iter().copied().min().unwrap_or_default();
    lightest - darkest <= BLANK_LUMA_SPREAD
}

fn encode_png(page: &DynamicImage) -> Result<Vec<u8>, SkipReason> {
    let mut png = Cursor::new(Vec::new());
    page.to_rgb8()
        .write_to(&mut png, ImageFormat::Png)
        .map_err(|error| failure(error.to_string()))?;
    let png = png.into_inner();
    if png.len() > MAX_PAGE_IMAGE_BYTES {
        return Err(failure("the page image is larger than the worker may send"));
    }
    Ok(png)
}

fn failed(error: pdf_oxide::Error) -> SkipReason {
    match error {
        pdf_oxide::Error::EncryptedPdf => SkipReason::PasswordProtected,
        error => failure(error.to_string()),
    }
}

fn failure(message: impl Into<String>) -> SkipReason {
    SkipReason::ExtractorFailed {
        message: message.into(),
    }
}
