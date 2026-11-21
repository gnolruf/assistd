//! A config file opened for editing: changes keep the file's comments and
//! layout, and each one is checked against the schema as it is made.

use std::collections::BTreeSet;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use toml_edit::{ArrayOfTables, DocumentMut, Item, Table};

use crate::errors::{ConfigError, ConfigIssue};
use crate::settings::{Consumer, consumers_of_changes, value_at};
use crate::top::{Config, create_private_dir_all};

/// An editable `config.toml`. Every accepted change leaves a document that
/// parses into a [`Config`]; [`save`](Self::save) also requires it to have no issues.
#[derive(Debug, Clone)]
pub struct ConfigDocument {
    path: PathBuf,
    document: DocumentMut,
    /// The file's text as last read or saved.
    saved_text: String,
    /// The config that text describes, serialized with its defaults filled in.
    saved_values: toml::Table,
    /// Whether the document's text differs from `saved_text`.
    modified: bool,
    /// The document's keys as parsed, without defaults.
    raw: toml::Table,
    config: Config,
    /// Dotted paths of the document's keys that the schema does not know.
    unknown_keys: Vec<String>,
    defaults: toml::Table,
}

impl ConfigDocument {
    /// Opens `path`; a missing file opens as an empty document that saving
    /// creates.
    pub fn open(path: &Path) -> Result<Self, ConfigError> {
        let text = read_text(path)?;
        let parse_error = |source| ConfigError::parse(path, &text, &source);
        let config: Config = toml::from_str(&text).map_err(parse_error)?;
        let raw = toml::from_str(&text).map_err(parse_error)?;
        let unknown_keys = config.unknown_keys(&raw)?;
        let document = text.parse().map_err(|source| ConfigError::Document {
            path: path.to_path_buf(),
            source,
        })?;
        Ok(Self {
            path: path.to_path_buf(),
            document,
            saved_text: text,
            saved_values: toml::Table::try_from(&config)?,
            modified: false,
            raw,
            config,
            unknown_keys,
            defaults: toml::Table::try_from(Config::default())?,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The config the document describes, with defaults filled in.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Dotted paths of the document's keys that the schema does not know.
    pub fn unknown_keys(&self) -> &[String] {
        &self.unknown_keys
    }

    /// Why the document would not load: an issue for each of
    /// [`Self::unknown_keys`], then its config's [`Config::issues`].
    pub fn issues(&self) -> Vec<ConfigIssue> {
        self.config.issues_after_unknown_keys(&self.unknown_keys)
    }

    /// The value the document sets at dotted `key`, if it sets one.
    pub fn file_value(&self, key: &str) -> Option<&toml::Value> {
        value_at(&self.raw, key)
    }

    /// The default at dotted `key`; `None` for keys that are unset by default.
    pub fn default_value(&self, key: &str) -> Option<&toml::Value> {
        value_at(&self.defaults, key)
    }

    /// Whether the document differs from the file as last read or saved.
    pub fn is_modified(&self) -> bool {
        self.modified
    }

    /// Sets dotted `key` to `value`, keeping any comment beside the old value.
    /// Refused, leaving the document unchanged, when `key` is not in the
    /// schema or `value` does not fit it.
    pub fn set(&mut self, key: &str, value: &toml::Value) -> Result<(), ConfigError> {
        let mut literal = literal(key, value)?;
        let mut document = self.document.clone();
        let slot = slot_mut(document.as_item_mut(), key).ok_or_else(|| not_in_a_table(key))?;
        if let Some(existing) = slot.as_value() {
            *literal.decor_mut() = existing.decor().clone();
        }
        *slot = Item::Value(literal);
        self.replace(document, key)
    }

    /// Removes dotted `key` from the document, so its default applies.
    pub fn reset(&mut self, key: &str) -> Result<(), ConfigError> {
        let mut document = self.document.clone();
        let (parents, leaf) = split_leaf(key);
        let parent = parents
            .into_iter()
            .flat_map(|parents| parents.split('.'))
            .try_fold(document.as_item_mut(), |item, segment| {
                item.get_mut(segment)
            });
        if let Some(table) = parent.and_then(Item::as_table_like_mut) {
            table.remove(leaf);
        }
        self.replace(document, key)
    }

    /// Puts `entry`, its keys in the order given, at `index` of the `[[key]]` tables at
    /// dotted `key`, appending when `index` is their count; the other tables keep their
    /// comments. Refused like [`Self::set`].
    pub fn put_table(
        &mut self,
        key: &str,
        index: usize,
        entry: &[(String, toml::Value)],
    ) -> Result<(), ConfigError> {
        let mut table = Table::new();
        for (name, value) in entry {
            table.insert(name, Item::Value(literal(key, value)?));
        }
        let mut document = self.document.clone();
        let tables = tables_mut(document.as_item_mut(), key).ok_or_else(|| not_in_a_table(key))?;
        match tables.get_mut(index) {
            Some(held) => {
                *table.decor_mut() = held.decor().clone();
                table.set_position(held.position());
                *held = table;
            }
            None => tables.push(table),
        }
        self.replace(document, key)
    }

    /// Removes the table at `index` of the `[[key]]` tables at dotted `key`, if there is one.
    pub fn remove_table(&mut self, key: &str, index: usize) -> Result<(), ConfigError> {
        let mut document = self.document.clone();
        let tables = tables_mut(document.as_item_mut(), key).ok_or_else(|| not_in_a_table(key))?;
        if index < tables.len() {
            tables.remove(index);
        }
        self.replace(document, key)
    }

    /// [`Self::overwrite`], refused with [`ConfigError::ChangedOnDisk`] when the
    /// file no longer holds the text last read or saved.
    pub fn save(&mut self) -> Result<BTreeSet<Consumer>, ConfigError> {
        ConfigError::reject(self.issues())?;
        if read_text(&self.path)? != self.saved_text {
            return Err(ConfigError::ChangedOnDisk {
                path: self.path.clone(),
            });
        }
        self.write()
    }

    /// Writes the document over the file, owner-only and atomically (a symlink's
    /// target, not the link), returning the consumers of the keys whose values it
    /// changed. Refused with [`ConfigError::Validation`] on any [`Self::issues`].
    pub fn overwrite(&mut self) -> Result<BTreeSet<Consumer>, ConfigError> {
        ConfigError::reject(self.issues())?;
        self.write()
    }

    fn write(&mut self) -> Result<BTreeSet<Consumer>, ConfigError> {
        let values = toml::Table::try_from(&self.config)?;
        let changed = consumers_of_changes(&self.saved_values, &values);
        let text = self.document.to_string();
        replace_private_file(&self.path, &text)?;
        self.saved_text = text;
        self.saved_values = values;
        self.modified = false;
        Ok(changed)
    }

    fn replace(&mut self, document: DocumentMut, key: &str) -> Result<(), ConfigError> {
        let text = document.to_string();
        let schema_error = |source: toml::de::Error| edit_error(key, source.message());
        let raw: toml::Table = toml::from_str(&text).map_err(schema_error)?;
        let config: Config = raw.clone().try_into().map_err(schema_error)?;
        let names_key = |unknown: &String| {
            key.strip_prefix(unknown.as_str())
                .is_some_and(|rest| rest.is_empty() || rest.starts_with('.'))
        };
        let unknown_keys = config.unknown_keys(&raw)?;
        if unknown_keys.iter().any(names_key) {
            return Err(edit_error(key, "not a config key"));
        }
        self.modified = text != self.saved_text;
        self.document = document;
        self.raw = raw;
        self.config = config;
        self.unknown_keys = unknown_keys;
        Ok(())
    }
}

fn edit_error(key: &str, reason: &str) -> ConfigError {
    ConfigError::Edit {
        key: key.to_string(),
        reason: reason.to_string(),
    }
}

fn not_in_a_table(key: &str) -> ConfigError {
    edit_error(key, "a parent key is not a table")
}

/// `value` as the document writes it.
fn literal(key: &str, value: &toml::Value) -> Result<toml_edit::Value, ConfigError> {
    value
        .to_string()
        .parse()
        .map_err(|source: toml_edit::TomlError| edit_error(key, source.message()))
}

/// The text of the file at `path`; empty when there is no file.
fn read_text(path: &Path) -> Result<String, ConfigError> {
    match fs::read_to_string(path) {
        Ok(text) => Ok(text),
        Err(source) if source.kind() == io::ErrorKind::NotFound => Ok(String::new()),
        Err(source) => Err(ConfigError::Read {
            path: path.to_path_buf(),
            source,
        }),
    }
}

/// The item at dotted `key`, creating missing parent tables (inline inside
/// inline tables, headerless elsewhere); `None` when a parent is a value.
fn slot_mut<'a>(root: &'a mut Item, key: &str) -> Option<&'a mut Item> {
    key.split('.').try_fold(root, |item, segment| {
        let vacant = if item.is_inline_table() {
            Item::Value(toml_edit::InlineTable::new().into())
        } else {
            Item::Table(implicit_table())
        };
        Some(item.as_table_like_mut()?.entry(segment).or_insert(vacant))
    })
}

/// The `[[key]]` tables at dotted `key`, made from an inline array there and from inline
/// tables above it, and created when missing; `None` when a parent is a value.
fn tables_mut<'a>(root: &'a mut Item, key: &str) -> Option<&'a mut ArrayOfTables> {
    let (parents, leaf) = split_leaf(key);
    let parent = parents
        .into_iter()
        .flat_map(|parents| parents.split('.'))
        .try_fold(root, |item, segment| {
            let table = item.as_table_like_mut()?;
            if !table.get(segment).is_some_and(Item::is_table) {
                let held = match table.remove(segment) {
                    Some(held) => held.into_table().ok()?,
                    None => implicit_table(),
                };
                table.insert(segment, Item::Table(held));
            }
            table.get_mut(segment)
        })?;
    let table = parent.as_table_like_mut()?;
    if !table.get(leaf).is_some_and(Item::is_array_of_tables) {
        let held = table
            .remove(leaf)
            .and_then(|held| held.into_array_of_tables().ok());
        table.insert(leaf, Item::ArrayOfTables(held.unwrap_or_default()));
    }
    table.get_mut(leaf)?.as_array_of_tables_mut()
}

/// Dotted `key` as its parents' path, if it has parents, and its last segment.
fn split_leaf(key: &str) -> (Option<&str>, &str) {
    key.rsplit_once('.')
        .map_or((None, key), |(parents, leaf)| (Some(parents), leaf))
}

/// A table that is written only once it holds a key.
fn implicit_table() -> Table {
    let mut table = Table::new();
    table.set_implicit(true);
    table
}

/// Writes `contents` beside the file `path` resolves to, then renames it into
/// place, so a reader never sees a partial file.
fn replace_private_file(path: &Path, contents: &str) -> Result<(), ConfigError> {
    let target = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    if let Some(parent) = target.parent() {
        create_private_dir_all(parent)?;
    }
    assistd_utils::fs::replace_private_file(&target, contents.as_bytes()).map_err(|source| {
        ConfigError::Write {
            path: target.clone(),
            source,
        }
    })
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::PermissionsExt;

    use super::*;

    const COMMENTED: &str = "# assistd configuration\n\n\
        [model]\n\
        name = \"owner/repo:Q4_K_M\"  # owner/repo:quant\n\
        port = 8385\n";

    const SERVERS: &str = "[mcp]\nenabled = true\n\n\
        # local files\n[[mcp.servers]]\nname = \"files\"  # the label\ncommand = \"npx\"\n\n\
        # hosted\n[[mcp.servers]]\nname = \"remote\"\nurl = \"https://mcp.example.com/mcp\"\n\n\
        [memory]\nenabled = false\n";

    #[test]
    fn edits_keep_comments_and_refuse_values_the_schema_rejects() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("config.toml");
        fs::write(&path, COMMENTED).unwrap();
        let mut document = ConfigDocument::open(&path).unwrap();

        document
            .set("model.name", &toml::Value::from("other/repo:Q8_0"))
            .unwrap();
        document
            .set("voice.wakeword.silence_ms", &toml::Value::from(600))
            .unwrap();
        document.reset("model.port").unwrap();
        assert!(
            document
                .set("model.port", &toml::Value::from("not a port"))
                .is_err()
        );
        assert!(
            document
                .set("model.no_such_key", &toml::Value::from(1))
                .is_err()
        );
        assert!(document.is_modified());
        document.save().unwrap();

        let saved = fs::read_to_string(&path).unwrap();
        assert!(saved.starts_with("# assistd configuration\n"));
        assert!(saved.contains("name = \"other/repo:Q8_0\"  # owner/repo:quant"));
        assert!(!saved.contains("port"));
        assert!(!saved.contains("no_such_key"));
        let reloaded = Config::load(&path).unwrap();
        assert_eq!(reloaded.voice.wakeword.silence_ms.get(), 600);
        assert_eq!(reloaded.model.port, Config::default().model.port);
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert!(!document.is_modified());
    }

    #[test]
    fn save_refuses_a_config_that_does_not_validate() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("config.toml");
        let mut document = ConfigDocument::open(&path).unwrap();

        document
            .set("chat.max_reasoning_tokens", &toml::Value::from(1_000_000))
            .unwrap();

        assert!(matches!(document.save(), Err(ConfigError::Validation(_))));
        assert!(!path.exists());
    }

    #[test]
    fn save_names_who_reads_the_changed_keys_and_refuses_a_file_changed_on_disk() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("config.toml");
        fs::write(&path, COMMENTED).unwrap();
        let mut document = ConfigDocument::open(&path).unwrap();

        document
            .set("notifications.auto_hide_ms", &toml::Value::from(900))
            .unwrap();
        document
            .set("chat.theme.accent", &toml::Value::from("red"))
            .unwrap();
        document
            .set("model.port", &toml::Value::from(8385))
            .unwrap();
        assert_eq!(
            document.save().unwrap(),
            BTreeSet::from([Consumer::Tray, Consumer::Chat])
        );

        let hand_edited = format!("{}\n[memory]\nenabled = false\n", document.document);
        fs::write(&path, &hand_edited).unwrap();
        document
            .set("voice.synthesis.conversational", &toml::Value::from(false))
            .unwrap();
        assert!(matches!(
            document.save(),
            Err(ConfigError::ChangedOnDisk { .. })
        ));
        assert_eq!(fs::read_to_string(&path).unwrap(), hand_edited);
        assert!(document.is_modified());

        assert_eq!(
            document.overwrite().unwrap(),
            BTreeSet::from([Consumer::Daemon])
        );
        assert!(!fs::read_to_string(&path).unwrap().contains("[memory]"));
    }

    #[test]
    fn tables_are_written_as_blocks_and_the_ones_left_alone_keep_their_comments() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("config.toml");
        fs::write(&path, SERVERS).unwrap();
        let mut document = ConfigDocument::open(&path).unwrap();
        let entry = |pairs: &[(&str, toml::Value)]| -> Vec<(String, toml::Value)> {
            pairs
                .iter()
                .map(|(name, value)| (name.to_string(), value.clone()))
                .collect()
        };
        let headers = toml::Table::from_iter([("X-Key".to_string(), toml::Value::from("k"))]);

        let remote = entry(&[
            ("name", "remote".into()),
            ("url", "https://mcp.example.com/v2".into()),
            ("headers", headers.into()),
        ]);
        document.put_table("mcp.servers", 1, &remote).unwrap();
        let local = entry(&[
            ("name", "local".into()),
            ("command", "server".into()),
            ("args", toml::Value::Array(vec!["--stdio".into()])),
        ]);
        document.put_table("mcp.servers", 2, &local).unwrap();
        let both = entry(&[
            ("name", "both".into()),
            ("command", "server".into()),
            ("url", "https://mcp.example.com/mcp".into()),
        ]);
        assert!(document.put_table("mcp.servers", 3, &both).is_err());

        assert_eq!(
            document.document.to_string(),
            "[mcp]\nenabled = true\n\n\
             # local files\n[[mcp.servers]]\nname = \"files\"  # the label\ncommand = \"npx\"\n\n\
             # hosted\n[[mcp.servers]]\nname = \"remote\"\nurl = \"https://mcp.example.com/v2\"\n\
             headers = { X-Key = \"k\" }\n\n\
             [[mcp.servers]]\nname = \"local\"\ncommand = \"server\"\nargs = [\"--stdio\"]\n\n\
             [memory]\nenabled = false\n"
        );

        document.remove_table("mcp.servers", 0).unwrap();
        let names: Vec<&str> = document
            .config()
            .mcp
            .servers
            .iter()
            .map(|server| server.name.as_str())
            .collect();
        assert_eq!(names, ["remote", "local"]);
    }
}
