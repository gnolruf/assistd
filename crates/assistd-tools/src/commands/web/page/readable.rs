//! Readability-style extraction: prunes page chrome from a parsed document
//! and picks the element holding its main content.

use std::rc::Rc;

use html2text::{Comment, Document, Element, Handle, RcDom};

/// Elements that only ever hold page chrome.
const CHROME_TAGS: [&str; 5] = ["nav", "aside", "footer", "header", "form"];

/// ARIA landmark roles that mark page chrome.
const CHROME_ROLES: [&str; 4] = ["navigation", "banner", "contentinfo", "complementary"];

/// Class or id fragments that mark boilerplate.
const UNLIKELY_NAMES: [&str; 16] = [
    "banner",
    "breadcrumb",
    "comment",
    "cookie",
    "footer",
    "header",
    "menu",
    "newsletter",
    "pagination",
    "popup",
    "promo",
    "related",
    "share",
    "sidebar",
    "social",
    "sponsor",
];

/// Class or id fragments that keep an element an unlikely fragment would prune.
const LIKELY_NAMES: [&str; 4] = ["article", "body", "content", "main"];

/// Elements never pruned for their class or id.
const NAME_EXEMPT_TAGS: [&str; 4] = ["html", "body", "article", "main"];

/// Elements whose text never reaches the rendered page.
const NON_TEXT_TAGS: [&str; 3] = ["script", "style", "template"];

/// Blocks whose text is scored.
const SCORED_TAGS: [&str; 3] = ["p", "pre", "td"];

/// Blocks with fewer characters than this are not scored.
const MIN_BLOCK_CHARS: usize = 25;

const CHARS_PER_LENGTH_POINT: usize = 100;

const MAX_LENGTH_POINTS: usize = 3;

/// A best container scoring below this means the page is not an article.
const MIN_CONTENT_SCORE: f64 = 10.0;

/// A best container with fewer characters than this means the page is not an article.
const MIN_CONTENT_CHARS: usize = 500;

/// Visible text measured over a subtree, in whitespace-collapsed characters.
#[derive(Clone, Copy, Debug, Default)]
struct TextStats {
    chars: usize,
    link_chars: usize,
    commas: usize,
}

impl TextStats {
    fn of(root: &Handle) -> Self {
        let mut stats = Self::default();
        let mut pending = vec![(root.clone(), false)];
        while let Some((node, in_link)) = pending.pop() {
            if let Some(text) = text_of(&node) {
                stats.add(&text, in_link);
                continue;
            }
            let tag = tag(&node);
            if tag.is_some_and(|tag| NON_TEXT_TAGS.contains(&tag)) {
                continue;
            }
            let in_link = in_link || tag == Some("a");
            pending.extend(
                node.children
                    .borrow()
                    .iter()
                    .map(|child| (child.clone(), in_link)),
            );
        }
        stats
    }

    fn add(&mut self, text: &str, in_link: bool) {
        let chars: usize = text
            .split_whitespace()
            .map(|word| word.chars().count() + 1)
            .sum();
        self.chars += chars;
        if in_link {
            self.link_chars += chars;
        }
        self.commas += text.matches(',').count();
    }

    fn link_density(&self) -> f64 {
        match self.chars {
            0 => 0.0,
            chars => self.link_chars as f64 / chars as f64,
        }
    }

    /// One point per block, one per comma, one per 100 characters up to three.
    fn block_score(&self) -> f64 {
        let length_points = (self.chars / CHARS_PER_LENGTH_POINT).min(MAX_LENGTH_POINTS);
        (1 + self.commas + length_points) as f64
    }
}

/// An element accumulating the scores of the blocks below it.
#[derive(Debug)]
struct Candidate {
    node: Handle,
    score: f64,
}

/// The node to render for `dom`: its main content when one stands out,
/// otherwise the whole document. Page chrome is pruned from `dom` either way.
pub(super) fn main_content(dom: &RcDom) -> Handle {
    prune(&dom.document);
    sole_landmark(&dom.document)
        .or_else(|| best_container(&dom.document))
        .unwrap_or_else(|| dom.document.clone())
}

fn prune(root: &Handle) {
    let mut pending = vec![root.clone()];
    while let Some(node) = pending.pop() {
        node.children
            .borrow_mut()
            .retain(|child| !is_boilerplate(child));
        pending.extend(node.children.borrow().iter().cloned());
    }
}

fn is_boilerplate(node: &Handle) -> bool {
    let Some(tag) = tag(node) else {
        return false;
    };
    CHROME_TAGS.contains(&tag)
        || attribute(node, "role").is_some_and(|role| {
            CHROME_ROLES
                .iter()
                .any(|chrome| role.eq_ignore_ascii_case(chrome))
        })
        || attribute(node, "hidden").is_some_and(|hidden| hidden != "until-found")
        || attribute(node, "aria-hidden").is_some_and(|hidden| hidden.eq_ignore_ascii_case("true"))
        || (!NAME_EXEMPT_TAGS.contains(&tag) && has_unlikely_name(node))
}

fn has_unlikely_name(node: &Handle) -> bool {
    let class = attribute(node, "class").unwrap_or_default();
    let id = attribute(node, "id").unwrap_or_default();
    let names = format!("{class} {id}").to_ascii_lowercase();
    UNLIKELY_NAMES.iter().any(|name| names.contains(name))
        && !LIKELY_NAMES.iter().any(|name| names.contains(name))
}

/// The page's one `<article>`, `<main>` or `role=main` element, if it has exactly one.
fn sole_landmark(root: &Handle) -> Option<Handle> {
    let mut landmarks = descendants(root).into_iter().filter(|node| {
        matches!(tag(node), Some("article" | "main"))
            || attribute(node, "role").is_some_and(|role| role.eq_ignore_ascii_case("main"))
    });
    let landmark = landmarks.next()?;
    landmarks.next().is_none().then_some(landmark)
}

/// The top-scoring container, unless its score or text is too slight to be an article.
fn best_container(root: &Handle) -> Option<Handle> {
    scored_candidates(root)
        .into_iter()
        .map(|candidate| {
            let stats = TextStats::of(&candidate.node);
            let score = candidate.score * (1.0 - stats.link_density());
            (candidate.node, score, stats.chars)
        })
        .max_by(|(_, left, _), (_, right, _)| left.total_cmp(right))
        .filter(|&(_, score, chars)| score >= MIN_CONTENT_SCORE && chars >= MIN_CONTENT_CHARS)
        .map(|(node, ..)| node)
}

/// Every parent and grandparent of a scored block, with the block scores added
/// to the parent in full and to the grandparent at half.
fn scored_candidates(root: &Handle) -> Vec<Candidate> {
    let mut candidates = Vec::new();
    let blocks = descendants(root)
        .into_iter()
        .filter(|node| tag(node).is_some_and(|tag| SCORED_TAGS.contains(&tag)));
    for block in blocks {
        let stats = TextStats::of(&block);
        if stats.chars < MIN_BLOCK_CHARS {
            continue;
        }
        let Some(parent) = element_parent(&block) else {
            continue;
        };
        if let Some(grandparent) = element_parent(&parent) {
            add_score(&mut candidates, grandparent, stats.block_score() / 2.0);
        }
        add_score(&mut candidates, parent, stats.block_score());
    }
    candidates
}

fn add_score(candidates: &mut Vec<Candidate>, node: Handle, score: f64) {
    match candidates
        .iter_mut()
        .find(|candidate| Rc::ptr_eq(&candidate.node, &node))
    {
        Some(candidate) => candidate.score += score,
        None => candidates.push(Candidate { node, score }),
    }
}

/// `root` and every node below it, in document order.
fn descendants(root: &Handle) -> Vec<Handle> {
    let mut found = Vec::new();
    let mut pending = vec![root.clone()];
    while let Some(node) = pending.pop() {
        pending.extend(node.children.borrow().iter().rev().cloned());
        found.push(node);
    }
    found
}

fn element_parent(node: &Handle) -> Option<Handle> {
    node.get_parent().filter(|parent| tag(parent).is_some())
}

fn tag(node: &Handle) -> Option<&str> {
    let Element { name, .. } = &node.data else {
        return None;
    };
    Some(&name.local)
}

fn attribute(node: &Handle, name: &str) -> Option<String> {
    let Element { attrs, .. } = &node.data else {
        return None;
    };
    attrs
        .borrow()
        .iter()
        .find(|attr| &*attr.name.local == name)
        .map(|attr| attr.value.to_string())
}

/// The text `node` holds when it is a text node, HTML-escaped. html2text does
/// not export its `Text` variant, so below `<html>` any node that is not an
/// element or comment is text, read back through the serializer.
fn text_of(node: &Handle) -> Option<String> {
    if matches!(node.data, Element { .. } | Comment { .. } | Document) {
        return None;
    }
    let mut serialized = Vec::new();
    node.serialize(&mut serialized).ok()?;
    String::from_utf8(serialized).ok()
}
