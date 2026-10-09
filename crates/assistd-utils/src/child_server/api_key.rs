//! A random API key for a spawned llama-server, handed over in a file only
//! its owner can read so it never shows up in the process's command line.

use std::fmt;
use std::fs::{File, Permissions};
use std::io::{self, Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};
use tempfile::TempPath;

const KEY_BYTES: usize = 32;
const KEY_FILE_PREFIX: &str = "assistd-api-key-";
const OWNER_ONLY: u32 = 0o600;

/// A secret generated per daemon launch, plus the file the server reads it
/// from. Clones share the file, which is removed when the last clone drops.
#[derive(Clone)]
pub struct ApiKey {
    secret: Arc<str>,
    file: Arc<TempPath>,
}

impl ApiKey {
    /// Draw a fresh key from the kernel's CSPRNG and write it to a mode-0600
    /// file under `$XDG_RUNTIME_DIR`, or the temp dir when that is unset.
    pub fn generate() -> io::Result<Self> {
        let secret: Arc<str> = random_hex()?.into();
        let file = write_key_file(&secret)?;
        Ok(Self {
            secret,
            file: Arc::new(file),
        })
    }

    /// Path to pass as the server's `--api-key-file`.
    pub fn file_path(&self) -> &Path {
        &self.file
    }

    /// An `Authorization: Bearer` header carrying the key, marked sensitive
    /// so it is redacted from debug output.
    pub fn authorization_headers(&self) -> HeaderMap {
        let mut value = HeaderValue::try_from(format!("Bearer {}", self.secret))
            .expect("a hex key is a valid header value");
        value.set_sensitive(true);
        HeaderMap::from_iter([(AUTHORIZATION, value)])
    }
}

impl fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ApiKey")
            .field("file", &self.file_path())
            .finish_non_exhaustive()
    }
}

fn random_hex() -> io::Result<String> {
    let mut bytes = [0u8; KEY_BYTES];
    File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    Ok(bytes.iter().map(|byte| format!("{byte:02x}")).collect())
}

fn write_key_file(secret: &str) -> io::Result<TempPath> {
    let mut file = tempfile::Builder::new()
        .prefix(KEY_FILE_PREFIX)
        .permissions(Permissions::from_mode(OWNER_ONLY))
        .tempfile_in(key_dir())?;
    writeln!(file, "{secret}")?;
    file.as_file().sync_all()?;
    Ok(file.into_temp_path())
}

fn key_dir() -> PathBuf {
    std::env::var_os("XDG_RUNTIME_DIR").map_or_else(std::env::temp_dir, PathBuf::from)
}
