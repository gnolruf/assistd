//! The settings file the daemon's SearXNG instance loads, holding only what
//! has no `SEARXNG_*` environment override.

use std::fs::{DirBuilder, Permissions};
use std::io::{self, Write};
use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
use std::path::{Path, PathBuf};

use assistd_config::SearchEngine;

use crate::error::SearchServerError;

const SETTINGS_DIR_MODE: u32 = 0o700;
const SETTINGS_FILE_MODE: u32 = 0o600;
const SETTINGS_FILE_NAME: &str = "settings.yml";

/// Overlays SearXNG's defaults, after the engine selection. The limiter needs
/// valkey and guards a loopback-only port; the JSON format is what the
/// daemon's client reads.
const SETTINGS: &str = "\
general:
  debug: false
  instance_name: assistd
server:
  limiter: false
  public_instance: false
  image_proxy: false
search:
  formats: [json]
";

/// One engine as the settings file names it, with its API key if it takes one.
struct EngineEntry<'a> {
    name: &'a str,
    api_key: Option<String>,
}

/// Write the settings file loading only `engines`, with their API keys read
/// from their key files, replacing any left by an earlier launch, and return
/// its path.
pub(crate) fn write_settings(engines: &[SearchEngine]) -> Result<PathBuf, SearchServerError> {
    let entries = engines
        .iter()
        .map(engine_entry)
        .collect::<Result<Vec<_>, _>>()?;
    let dir = settings_dir();
    let path = dir.join(SETTINGS_FILE_NAME);
    let settings = settings_yaml(&entries);
    replace_file(&dir, &path, &settings).map_err(|source| SearchServerError::Settings {
        path: path.clone(),
        source,
    })?;
    Ok(path)
}

fn engine_entry(engine: &SearchEngine) -> Result<EngineEntry<'_>, SearchServerError> {
    let name = engine.name();
    let api_key = engine
        .api_key_file()
        .map(|path| read_api_key(name, path))
        .transpose()?;
    Ok(EngineEntry { name, api_key })
}

fn read_api_key(engine: &str, path: &Path) -> Result<String, SearchServerError> {
    let contents = std::fs::read_to_string(path).map_err(|source| SearchServerError::ApiKey {
        engine: engine.to_owned(),
        path: path.to_path_buf(),
        source,
    })?;
    let key = contents.trim();
    if key.is_empty() {
        return Err(SearchServerError::EmptyApiKey {
            engine: engine.to_owned(),
            path: path.to_path_buf(),
        });
    }
    Ok(key.to_owned())
}

/// `keep_only` drops every other engine, but keeps each kept one's default
/// `disabled` and `inactive` flags, so each also gets an entry clearing them.
/// Strings are written as JSON, which YAML reads as double-quoted scalars.
fn settings_yaml(entries: &[EngineEntry<'_>]) -> String {
    let names: Vec<String> = entries
        .iter()
        .map(|entry| json_string(entry.name))
        .collect();
    let mut yaml = format!(
        "use_default_settings:\n  engines:\n    keep_only: [{}]\n{SETTINGS}engines:\n",
        names.join(", ")
    );
    for (name, entry) in names.iter().zip(entries) {
        yaml.push_str(&format!(
            "  - name: {name}\n    disabled: false\n    inactive: false\n"
        ));
        if let Some(key) = &entry.api_key {
            yaml.push_str(&format!("    api_key: {}\n", json_string(key)));
        }
    }
    yaml
}

fn json_string(text: &str) -> String {
    serde_json::to_string(text).expect("a string always serializes to JSON")
}

fn settings_dir() -> PathBuf {
    dirs::runtime_dir()
        .or_else(dirs::cache_dir)
        .unwrap_or_else(|| PathBuf::from(".cache"))
        .join("assistd")
        .join("searxng")
}

/// Renamed into place, so a symlink planted at `path` is replaced, not followed.
fn replace_file(dir: &Path, path: &Path, contents: &str) -> io::Result<()> {
    DirBuilder::new()
        .recursive(true)
        .mode(SETTINGS_DIR_MODE)
        .create(dir)?;
    let mut file = tempfile::Builder::new()
        .prefix(SETTINGS_FILE_NAME)
        .permissions(Permissions::from_mode(SETTINGS_FILE_MODE))
        .tempfile_in(dir)?;
    file.write_all(contents.as_bytes())?;
    file.persist(path).map_err(|e| e.error)?;
    Ok(())
}
