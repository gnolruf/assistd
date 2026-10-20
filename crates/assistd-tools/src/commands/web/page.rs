//! Renders fetched HTML pages as plain text: title first, links inline
//! with their absolute URLs, page chrome dropped unless the whole page is asked for.

use std::borrow::Cow;

use encoding_rs::{Encoding, UTF_8};
use html2text::config::Config;
use html2text::render::{PlainDecorator, TextDecorator};
use html2text::{Handle, RcDom};
use reqwest::Url;
use reqwest::header::CONTENT_TYPE;

mod readable;

/// Columns HTML pages are wrapped to once rendered as text.
const TEXT_WIDTH: usize = 100;

/// Leading bytes searched for a `<meta>` charset declaration.
const PRESCAN_BYTES: usize = 1024;

const CHARSET: &[u8] = b"charset";

/// How much of an HTML page is rendered as text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Extent {
    /// The main content when the page has one, and never navigation, sidebars or footers.
    Article,
    Whole,
}

/// An HTML response whose body is rendered as text once read.
#[derive(Debug)]
pub(super) struct HtmlPage {
    url: Url,
    declared: Option<&'static Encoding>,
    extent: Extent,
}

impl HtmlPage {
    /// The page `response` carries, or `None` when it is not HTML.
    pub(super) fn of(response: &reqwest::Response, extent: Extent) -> Option<Self> {
        let content_type = response.headers().get(CONTENT_TYPE)?.to_str().ok()?;
        let mut parts = content_type.split(';');
        let mime = parts.next()?.trim();
        let is_html = mime.eq_ignore_ascii_case("text/html")
            || mime.eq_ignore_ascii_case("application/xhtml+xml");
        is_html.then(|| Self {
            url: response.url().clone(),
            declared: parts.find_map(|part| {
                let (name, rest) = part
                    .trim_start()
                    .as_bytes()
                    .split_at_checked(CHARSET.len())?;
                name.eq_ignore_ascii_case(CHARSET)
                    .then(|| charset_label(rest))
                    .flatten()
            }),
            extent,
        })
    }

    /// `body` decoded and rendered as text; returned unchanged if it cannot be parsed.
    pub(super) fn render(&self, body: Vec<u8>) -> Vec<u8> {
        let html = decode(&body, self.declared);
        page_text(&html, &self.url, self.extent).map_or(body, String::into_bytes)
    }
}

/// Plain-text rendering that prints each followable link as `text (url)`.
#[derive(Clone, Debug)]
struct LinkDecorator {
    page: Url,
    plain: PlainDecorator,
    open_links: Vec<Option<Url>>,
}

impl LinkDecorator {
    fn new(page: &Url) -> Self {
        let mut page = page.clone();
        page.set_fragment(None);
        Self {
            page,
            plain: PlainDecorator::new(),
            open_links: Vec::new(),
        }
    }

    /// The absolute http(s) URL `href` points at, unless it only names a
    /// fragment of this page.
    fn followable(&self, href: &str) -> Option<Url> {
        let target = self
            .page
            .join(href)
            .ok()
            .filter(|url| matches!(url.scheme(), "http" | "https"))?;
        let mut without_fragment = target.clone();
        without_fragment.set_fragment(None);
        (without_fragment != self.page).then_some(target)
    }
}

impl TextDecorator for LinkDecorator {
    type Annotation = ();

    fn decorate_link_start(&mut self, url: &str) -> (String, ()) {
        let target = self.followable(url);
        self.open_links.push(target);
        (String::new(), ())
    }

    fn decorate_link_end(&mut self) -> String {
        self.open_links
            .pop()
            .flatten()
            .map_or_else(String::new, |url| format!(" ({url})"))
    }

    fn decorate_em_start(&self) -> (String, ()) {
        self.plain.decorate_em_start()
    }

    fn decorate_em_end(&self) -> String {
        self.plain.decorate_em_end()
    }

    fn decorate_strong_start(&self) -> (String, ()) {
        self.plain.decorate_strong_start()
    }

    fn decorate_strong_end(&self) -> String {
        self.plain.decorate_strong_end()
    }

    fn decorate_strikeout_start(&self) -> (String, ()) {
        self.plain.decorate_strikeout_start()
    }

    fn decorate_strikeout_end(&self) -> String {
        self.plain.decorate_strikeout_end()
    }

    fn decorate_code_start(&self) -> (String, ()) {
        self.plain.decorate_code_start()
    }

    fn decorate_code_end(&self) -> String {
        self.plain.decorate_code_end()
    }

    fn decorate_preformat_first(&self) {}

    fn decorate_preformat_cont(&self) {}

    fn decorate_image(&mut self, src: &str, title: &str) -> (String, ()) {
        self.plain.decorate_image(src, title)
    }

    fn header_prefix(&self, level: usize) -> String {
        self.plain.header_prefix(level)
    }

    fn quote_prefix(&self) -> String {
        self.plain.quote_prefix()
    }

    fn unordered_item_prefix(&self) -> String {
        "- ".to_string()
    }

    fn ordered_item_prefix(&self, index: i64) -> String {
        self.plain.ordered_item_prefix(index)
    }

    fn make_subblock_decorator(&self) -> Self {
        self.clone()
    }
}

/// `html` rendered as text, `<title>` first, with links resolved against `page`.
fn page_text(html: &str, page: &Url, extent: Extent) -> Result<String, html2text::Error> {
    let config = html2text::config::with_decorator(LinkDecorator::new(page))
        .allow_width_overflow()
        .no_link_wrapping()
        .no_table_borders();
    let dom = config.parse_html(html.as_bytes())?;
    let title = title_element(&dom);
    let root = match extent {
        Extent::Article => readable::main_content(&dom),
        Extent::Whole => dom.document.clone(),
    };
    let body = node_text(&config, root)?;
    let Some(title) = title else {
        return Ok(body);
    };
    Ok(match node_text(&config, title)?.trim() {
        "" => body,
        title => format!("{title}\n\n{body}"),
    })
}

fn node_text(config: &Config<LinkDecorator>, root: Handle) -> Result<String, html2text::Error> {
    let dom = RcDom {
        document: root,
        ..RcDom::default()
    };
    config.render_to_string(config.dom_to_render_tree(&dom)?, TEXT_WIDTH)
}

fn decode<'a>(body: &'a [u8], declared: Option<&'static Encoding>) -> Cow<'a, str> {
    let encoding = declared.or_else(|| meta_charset(body)).unwrap_or(UTF_8);
    encoding.decode(body).0
}

/// The encoding a `charset=` label in the first bytes of the page names,
/// as an HTML parser's prescan would find it.
fn meta_charset(body: &[u8]) -> Option<&'static Encoding> {
    let head = &body[..body.len().min(PRESCAN_BYTES)];
    head.windows(CHARSET.len())
        .enumerate()
        .filter(|(_, window)| window.eq_ignore_ascii_case(CHARSET))
        .find_map(|(at, _)| charset_label(&head[at + CHARSET.len()..]))
        .map(Encoding::output_encoding)
}

/// The encoding named by `rest`, the bytes after a `charset` keyword.
fn charset_label(rest: &[u8]) -> Option<&'static Encoding> {
    let value = rest
        .trim_ascii_start()
        .strip_prefix(b"=")?
        .trim_ascii_start();
    let value = value
        .strip_prefix(b"\"")
        .or_else(|| value.strip_prefix(b"'"))
        .unwrap_or(value);
    let end = value
        .iter()
        .position(|byte| !(byte.is_ascii_alphanumeric() || b"-_:.".contains(byte)))
        .unwrap_or(value.len());
    Encoding::for_label(&value[..end])
}

fn title_element(dom: &RcDom) -> Option<Handle> {
    let html = child_element(&dom.document, "html")?;
    let head = child_element(&html, "head")?;
    child_element(&head, "title")
}

fn child_element(parent: &Handle, name: &str) -> Option<Handle> {
    parent
        .children
        .borrow()
        .iter()
        .find(|child| child.element_name().as_deref() == Some(name))
        .cloned()
}
