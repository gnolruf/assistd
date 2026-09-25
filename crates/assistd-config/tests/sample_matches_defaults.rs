//! Checks `config/config.sample.toml` against the schema in both directions:
//! every defaulted key is documented, and no documented key is stale.

use std::collections::BTreeSet;

use assistd_config::Config;

const SAMPLE: &str = include_str!("../../../config/config.sample.toml");

/// `[[mcp.servers]]` entries are example user data, not schema.
const USER_DATA_SECTION: &str = "mcp.servers";

/// One `key = value` the sample documents, live or commented out, with
/// the section it sits under.
struct DocumentedKey {
    section: String,
    key: String,
    value: String,
}

impl DocumentedKey {
    fn path(&self) -> String {
        if self.section.is_empty() {
            self.key.clone()
        } else {
            format!("{}.{}", self.section, self.key)
        }
    }
}

fn is_bare_key(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn is_section_path(s: &str) -> bool {
    !s.is_empty() && s.split('.').all(is_bare_key)
}

/// Every assignment the sample documents, tracking sections across live and
/// commented-out headers. Textual, because most of the sample is commented out.
fn documented_keys() -> Vec<DocumentedKey> {
    let mut out = Vec::new();
    let mut section = String::new();
    for line in SAMPLE.lines() {
        let content = line.trim_start().trim_start_matches('#').trim();
        if let Some(header) = section_header(content) {
            section = header.to_string();
            continue;
        }
        if section.starts_with(USER_DATA_SECTION) {
            continue;
        }
        let Some((key, value)) = bare_assignment(content) else {
            continue;
        };
        out.push(DocumentedKey {
            section: section.clone(),
            key: key.to_string(),
            value: value.to_string(),
        });
    }
    out
}

/// The path in a `[a.b]` or `[[a.b]]` header; `None` for bracketed prose.
fn section_header(content: &str) -> Option<&str> {
    let inner = content.strip_prefix('[')?.strip_suffix(']')?;
    let header = inner.trim_start_matches('[').trim_end_matches(']');
    is_section_path(header).then_some(header)
}

/// A `bare_key = value` line; `None` for prose containing `=` and for
/// continuation lines of a multi-line array.
fn bare_assignment(content: &str) -> Option<(&str, &str)> {
    let (key, value) = content.split_once('=')?;
    let key = key.trim();
    is_bare_key(key).then_some((key, value.trim()))
}

/// Key paths reachable from a serialized `Config::default()`: the keys the
/// sample must document. `None` fields are omitted by `toml`, and
/// [`USER_DATA_SECTION`] is documented as example tables instead.
fn keys_with_a_default() -> BTreeSet<String> {
    let serialized =
        toml::to_string_pretty(&Config::default()).expect("Config::default must serialize");
    let value: toml::Value = toml::from_str(&serialized).expect("re-parse of serialized default");
    let mut out = BTreeSet::new();
    collect_key_paths(&value, "", &mut out);
    out.remove(USER_DATA_SECTION);
    out
}

fn collect_key_paths(value: &toml::Value, prefix: &str, out: &mut BTreeSet<String>) {
    let toml::Value::Table(table) = value else {
        out.insert(prefix.to_string());
        return;
    };
    for (key, child) in table {
        let path = if prefix.is_empty() {
            key.clone()
        } else {
            format!("{prefix}.{key}")
        };
        collect_key_paths(child, &path, out);
    }
}

#[test]
fn sample_as_shipped_is_a_valid_config() {
    let cfg: Config = toml::from_str(SAMPLE).expect("config.sample.toml must parse as a Config");
    cfg.validate()
        .expect("config.sample.toml must be a valid configuration");
}

#[test]
fn every_key_with_a_default_is_documented() {
    let documented: BTreeSet<String> = documented_keys().iter().map(DocumentedKey::path).collect();
    let missing: Vec<_> = keys_with_a_default()
        .difference(&documented)
        .cloned()
        .collect();
    assert!(
        missing.is_empty(),
        "these keys exist in Config but config.sample.toml never mentions them: {missing:#?}"
    );
}

/// Replays each documented assignment alone; only an "unknown field" error
/// marks it stale, since a mismatched example value is a type error.
#[test]
fn sample_documents_no_key_the_schema_has_dropped() {
    let mut stale = Vec::new();
    for doc in documented_keys() {
        let snippet = if doc.section.is_empty() {
            format!("{} = {}\n", doc.key, doc.value)
        } else {
            format!("[{}]\n{} = {}\n", doc.section, doc.key, doc.value)
        };
        let Err(error) = toml::from_str::<Config>(&snippet) else {
            continue;
        };
        if error.to_string().contains("unknown field") {
            stale.push(doc.path());
        }
    }
    assert!(
        stale.is_empty(),
        "config.sample.toml documents keys the schema no longer has: {stale:#?}"
    );
}
