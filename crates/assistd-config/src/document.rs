//! A config file opened for editing: changes keep the file's comments and
//! layout, and each one is checked against the schema as it is made.

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use toml_edit::{DocumentMut, Item, Table};

use crate::errors::ConfigError;
use crate::top::{Config, create_private_dir_all, create_private_file};

/// An editable `config.toml`. Every accepted change leaves a document that
/// parses into a [`Config`]; [`save`](Self::save) also requires it to validate.
#[derive(Debug, Clone)]
pub struct ConfigDocument {
    path: PathBuf,
    document: DocumentMut,
    /// The file's text as last read or saved.
    saved_text: String,
    /// The document's keys as parsed, without defaults.
    raw: toml::Table,
    config: Config,
    defaults: toml::Table,
}

impl ConfigDocument {
    /// Opens `path`; a missing file opens as an empty document that saving
    /// creates.
    pub fn open(path: &Path) -> Result<Self, ConfigError> {
        let text = match fs::read_to_string(path) {
            Ok(text) => text,
            Err(source) if source.kind() == io::ErrorKind::NotFound => String::new(),
            Err(source) => {
                return Err(ConfigError::Read {
                    path: path.to_path_buf(),
                    source,
                });
            }
        };
        let parse_error = |source| ConfigError::Parse {
            path: path.to_path_buf(),
            source,
        };
        let config = toml::from_str(&text).map_err(parse_error)?;
        let raw = toml::from_str(&text).map_err(parse_error)?;
        let document = text.parse().map_err(|source| ConfigError::Document {
            path: path.to_path_buf(),
            source,
        })?;
        Ok(Self {
            path: path.to_path_buf(),
            document,
            saved_text: text,
            raw,
            config,
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

    /// The value the document sets at dotted `key`, if it sets one.
    pub fn file_value(&self, key: &str) -> Option<&toml::Value> {
        lookup(&self.raw, key)
    }

    /// The default at dotted `key`; `None` for keys that are unset by default.
    pub fn default_value(&self, key: &str) -> Option<&toml::Value> {
        lookup(&self.defaults, key)
    }

    /// Whether the document differs from the file as last read or saved.
    pub fn is_modified(&self) -> bool {
        self.document.to_string() != self.saved_text
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

    /// Writes the document over the file, owner-only, replacing it atomically
    /// (a symlink's target, not the link). Refused when the config does not
    /// validate.
    pub fn save(&mut self) -> Result<(), ConfigError> {
        self.config.validate()?;
        let text = self.document.to_string();
        replace_private_file(&self.path, &text)?;
        self.saved_text = text;
        Ok(())
    }

    fn replace(&mut self, document: DocumentMut, key: &str) -> Result<(), ConfigError> {
        let text = document.to_string();
        let schema_error = |source: toml::de::Error| edit_error(key, source.message());
        let config: Config = toml::from_str(&text).map_err(schema_error)?;
        let raw: toml::Table = toml::from_str(&text).map_err(schema_error)?;
        let names_key = |unknown: &String| {
            key.strip_prefix(unknown.as_str())
                .is_some_and(|rest| rest.is_empty() || rest.starts_with('.'))
        };
        if config.unknown_keys(&raw)?.iter().any(names_key) {
            return Err(edit_error(key, "not a config key"));
        }
        self.document = document;
        self.raw = raw;
        self.config = config;
        Ok(())
    }
}

fn edit_error(key: &str, reason: &str) -> ConfigError {
    ConfigError::Edit {
        key: key.to_string(),
        reason: reason.to_string(),
    }
}

fn lookup<'a>(table: &'a toml::Table, key: &str) -> Option<&'a toml::Value> {
    let mut segments = key.split('.');
    let first = table.get(segments.next()?)?;
    segments.try_fold(first, |value, segment| value.get(segment))
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
    let mut staging_name = target.file_name().unwrap_or_default().to_os_string();
    staging_name.push(".tmp");
    let staging = target.with_file_name(staging_name);
    let write_error = |source| ConfigError::Write {
        path: target.clone(),
        source,
    };
    match fs::remove_file(&staging) {
        Err(source) if source.kind() != io::ErrorKind::NotFound => return Err(write_error(source)),
        _ => {}
    }
    let mut file = create_private_file(&staging)?;
    file.write_all(contents.as_bytes())
        .and_then(|()| file.sync_all())
        .map_err(write_error)?;
    fs::rename(&staging, &target).map_err(write_error)
}

#[cfg(test)]
mod tests;
