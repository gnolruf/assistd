//! The settings file the daemon's SearXNG instance loads, holding only what
//! has no `SEARXNG_*` environment override.

use std::fs::{DirBuilder, Permissions};
use std::io::{self, Write};
use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
use std::path::{Path, PathBuf};

use crate::error::SearchServerError;

const SETTINGS_DIR_MODE: u32 = 0o700;
const SETTINGS_FILE_MODE: u32 = 0o600;
const SETTINGS_FILE_NAME: &str = "settings.yml";

/// Overlays SearXNG's defaults. The limiter needs valkey and guards a
/// loopback-only port; the JSON format is what the daemon's client reads.
const SETTINGS: &str = "\
use_default_settings: true
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

/// Write the settings file, replacing any left by an earlier launch, and
/// return its path.
pub(crate) fn write_settings() -> Result<PathBuf, SearchServerError> {
    let dir = settings_dir();
    let path = dir.join(SETTINGS_FILE_NAME);
    replace_file(&dir, &path).map_err(|source| SearchServerError::Settings {
        path: path.clone(),
        source,
    })?;
    Ok(path)
}

fn settings_dir() -> PathBuf {
    dirs::runtime_dir()
        .or_else(dirs::cache_dir)
        .unwrap_or_else(|| PathBuf::from(".cache"))
        .join("assistd")
        .join("searxng")
}

/// Renamed into place, so a symlink planted at `path` is replaced, not followed.
fn replace_file(dir: &Path, path: &Path) -> io::Result<()> {
    DirBuilder::new()
        .recursive(true)
        .mode(SETTINGS_DIR_MODE)
        .create(dir)?;
    let mut file = tempfile::Builder::new()
        .prefix(SETTINGS_FILE_NAME)
        .permissions(Permissions::from_mode(SETTINGS_FILE_MODE))
        .tempfile_in(dir)?;
    file.write_all(SETTINGS.as_bytes())?;
    file.persist(path).map_err(|e| e.error)?;
    Ok(())
}
