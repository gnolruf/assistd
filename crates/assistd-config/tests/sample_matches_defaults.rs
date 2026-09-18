//! `config/config.sample.toml` is the documentation for the schema, and
//! documentation drifts. These check it against the code in both
//! directions, so a key added without the sample — or left in the sample
//! after being deleted from the code — fails here rather than misleading
//! someone later.

use std::collections::BTreeSet;

use assistd_config::Config;

const SAMPLE: &str = include_str!("../../../config/config.sample.toml");

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

/// Walk the sample, tracking the current section across both live and
/// commented-out headers, and collect every assignment it documents.
/// Extracted textually rather than by parsing, because most of the
/// sample is deliberately commented out.
fn documented_keys() -> Vec<DocumentedKey> {
    let mut out = Vec::new();
    let mut section = String::new();
    for line in SAMPLE.lines() {
        let content = line.trim_start().trim_start_matches('#').trim();

        if let Some(inner) = content.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
            let header = inner.trim_start_matches('[').trim_end_matches(']');
            // Guard against prose that happens to be bracketed.
            if is_section_path(header) {
                section = header.to_string();
                continue;
            }
        }
        // `[[mcp.servers]]` entries are user data, not schema.
        if section.starts_with("mcp.servers") {
            continue;
        }
        let Some((key, value)) = content.split_once('=') else {
            continue;
        };
        let (key, value) = (key.trim(), value.trim());
        // A bare key only; anything else is prose containing an `=`, or a
        // continuation line of a multi-line array.
        if !is_bare_key(key) {
            continue;
        }
        out.push(DocumentedKey {
            section: section.clone(),
            key: key.to_string(),
            value: value.to_string(),
        });
    }
    out
}

/// Key paths reachable from a serialized `Config::default()`. `Option`
/// fields default to `None` and `toml` omits them, so this is the set of
/// keys that must appear in the sample, not the full schema.
fn keys_with_a_default() -> BTreeSet<String> {
    fn flatten(value: &toml::Value, prefix: &str, out: &mut BTreeSet<String>) {
        let toml::Value::Table(t) = value else {
            out.insert(prefix.to_string());
            return;
        };
        for (k, v) in t {
            let path = if prefix.is_empty() {
                k.clone()
            } else {
                format!("{prefix}.{k}")
            };
            match v {
                toml::Value::Table(_) => flatten(v, &path, out),
                _ => {
                    out.insert(path);
                }
            }
        }
    }

    let serialized =
        toml::to_string_pretty(&Config::default()).expect("Config::default must serialize");
    let value: toml::Value = toml::from_str(&serialized).expect("re-parse of serialized default");
    let mut out = BTreeSet::new();
    flatten(&value, "", &mut out);
    // `servers` is an empty array in the default; the sample documents it
    // as `[[mcp.servers]]` tables instead.
    out.remove("mcp.servers");
    out
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

/// The reverse direction, using `deny_unknown_fields` as the oracle:
/// replay each documented assignment on its own and see whether the
/// schema still knows the key. A wrong *value* yields a type error,
/// which is fine here — only "unknown field" means the key is stale.
#[test]
fn sample_documents_no_key_the_schema_has_dropped() {
    let mut stale = Vec::new();
    for doc in documented_keys() {
        let snippet = if doc.section.is_empty() {
            format!("{} = {}\n", doc.key, doc.value)
        } else {
            format!("[{}]\n{} = {}\n", doc.section, doc.key, doc.value)
        };
        let Err(e) = toml::from_str::<Config>(&snippet) else {
            continue;
        };
        if e.to_string().contains("unknown field") {
            stale.push(doc.path());
        }
    }
    assert!(
        stale.is_empty(),
        "config.sample.toml documents keys the schema no longer has: {stale:#?}"
    );
}
