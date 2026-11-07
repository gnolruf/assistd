//! Readability-style extraction: prunes page chrome from a parsed document
//! and picks the element holding its main content.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
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

/// Table parts html2text cannot render outside their `<table>`.
const TABLE_PARTS: [&str; 6] = ["tbody", "thead", "tfoot", "tr", "td", "th"];

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
    fn of_text(text: &str) -> Self {
        Self {
            chars: text
                .split_whitespace()
                .map(|word| word.chars().count() + 1)
                .sum(),
            link_chars: 0,
            commas: text.matches(',').count(),
        }
    }

    fn add(&mut self, other: Self) {
        self.chars += other.chars;
        self.link_chars += other.link_chars;
        self.commas += other.commas;
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

/// The [`TextStats`] of every element in a tree, measured in one pass from the leaves up.
struct TextIndex(HashMap<*const (), TextStats>);

impl TextIndex {
    fn of(root: &Handle) -> Self {
        let mut index = Self(HashMap::new());
        for node in descendants(root).iter().rev() {
            if tag(node).is_some() {
                let stats = index.measure(node);
                index.0.insert(node_key(node), stats);
            }
        }
        index
    }

    /// `node`'s stats from those of its children, which are already indexed.
    fn measure(&self, node: &Handle) -> TextStats {
        let tag = tag(node);
        let mut stats = TextStats::default();
        if tag.is_some_and(|tag| NON_TEXT_TAGS.contains(&tag)) {
            return stats;
        }
        for child in node.children.borrow().iter() {
            stats.add(self.get(child));
        }
        if tag == Some("a") {
            stats.link_chars = stats.chars;
        }
        stats
    }

    fn get(&self, node: &Handle) -> TextStats {
        match text_of(node) {
            Some(text) => TextStats::of_text(&text),
            None => self.0.get(&node_key(node)).copied().unwrap_or_default(),
        }
    }
}

/// An element accumulating the scores of the blocks below it.
#[derive(Debug)]
struct Candidate {
    node: Handle,
    score: f64,
}

/// The candidates found so far, each indexed by its node.
#[derive(Default)]
struct Candidates {
    found: Vec<Candidate>,
    by_node: HashMap<*const (), usize>,
}

impl Candidates {
    fn add_score(&mut self, node: Handle, score: f64) {
        let node = enclosing_table(node);
        match self.by_node.entry(node_key(&node)) {
            Entry::Occupied(at) => self.found[*at.get()].score += score,
            Entry::Vacant(slot) => {
                slot.insert(self.found.len());
                self.found.push(Candidate { node, score });
            }
        }
    }
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
    let text = TextIndex::of(root);
    scored_candidates(root, &text)
        .into_iter()
        .map(|candidate| {
            let stats = text.get(&candidate.node);
            let score = candidate.score * (1.0 - stats.link_density());
            (candidate.node, score, stats.chars)
        })
        .max_by(|(_, left, _), (_, right, _)| left.total_cmp(right))
        .filter(|&(_, score, chars)| score >= MIN_CONTENT_SCORE && chars >= MIN_CONTENT_CHARS)
        .map(|(node, ..)| node)
}

/// Every parent and grandparent of a scored block, with the block scores added
/// to the parent in full and to the grandparent at half.
fn scored_candidates(root: &Handle, text: &TextIndex) -> Vec<Candidate> {
    let mut candidates = Candidates::default();
    let blocks = descendants(root)
        .into_iter()
        .filter(|node| tag(node).is_some_and(|tag| SCORED_TAGS.contains(&tag)));
    for block in blocks {
        let stats = text.get(&block);
        if stats.chars < MIN_BLOCK_CHARS {
            continue;
        }
        let Some(parent) = element_parent(&block) else {
            continue;
        };
        if let Some(grandparent) = element_parent(&parent) {
            candidates.add_score(grandparent, stats.block_score() / 2.0);
        }
        candidates.add_score(parent, stats.block_score());
    }
    candidates.found
}

/// The identity of `node`, for indexing by node.
fn node_key(node: &Handle) -> *const () {
    Rc::as_ptr(node).cast()
}

/// `node`, or the `<table>` around it when it is a table part.
fn enclosing_table(mut node: Handle) -> Handle {
    while tag(&node).is_some_and(|tag| TABLE_PARTS.contains(&tag)) {
        let Some(parent) = element_parent(&node) else {
            break;
        };
        node = parent;
    }
    node
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
