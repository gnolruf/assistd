//! A config file opened for editing: changes keep the file's comments and
//! layout, and each one is checked against the schema as it is made.

use std::collections::BTreeSet;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use toml_edit::{DocumentMut, Item, Table};

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
        let mut literal: toml_edit::Value = value
            .to_string()
            .parse()
            .map_err(|source: toml_edit::TomlError| edit_error(key, source.message()))?;
        let mut document = self.document.clone();
        let slot = slot_mut(document.as_item_mut(), key)
            .ok_or_else(|| edit_error(key, "a parent key is not a table"))?;
        if let Some(existing) = slot.as_value() {
            *literal.decor_mut() = existing.decor().clone();
        }
        *slot = Item::Value(literal);
        self.replace(document, key)
    }

    /// Removes dotted `key` from the document, so its default applies.
    pub fn reset(&mut self, key: &str) -> Result<(), ConfigError> {
        let mut document = self.document.clone();
        let (parents, leaf) = key
            .rsplit_once('.')
            .map_or((None, key), |(parents, leaf)| (Some(parents), leaf));
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
            let mut table = Table::new();
            table.set_implicit(true);
            Item::Table(table)
        };
        Some(item.as_table_like_mut()?.entry(segment).or_insert(vacant))
    })
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
mod tests;
