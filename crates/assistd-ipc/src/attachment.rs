//! Reading and validating image files before they are attached to a
//! [`Request::Query`](crate::Request::Query).

use std::fmt;
use std::io;
use std::path::Path;

/// MIME types llama.cpp's vision adapters accept; `infer::is_image` alone also passes GIF, BMP,
/// TIFF and HEIC.
const SUPPORTED_MIMES: &[&str] = &["image/png", "image/jpeg", "image/webp"];

/// Largest accepted image: room for a 4K PNG screenshot, small enough to reject video or RAW.
pub const MAX_IMAGE_BYTES: u64 = 32 * 1024 * 1024;

/// A validated image read by [`load_image`].
#[derive(Debug)]
pub struct LoadedImage {
    /// One of `image/png`, `image/jpeg` or `image/webp`.
    pub mime: String,
    pub bytes: Vec<u8>,
}

/// Why [`load_image`] refused a file.
#[derive(Debug)]
pub enum LoadImageError {
    Io {
        path: String,
        source: io::Error,
    },
    /// File exceeds [`MAX_IMAGE_BYTES`].
    TooLarge {
        path: String,
        size: u64,
        max: u64,
    },
    /// No magic number matched.
    Unrecognized {
        path: String,
    },
    NotAnImage {
        path: String,
        detected: String,
    },
    /// An image, but not PNG, JPEG or WebP.
    UnsupportedFormat {
        path: String,
        mime: String,
    },
}

impl LoadImageError {
    /// One-line, unprefixed, human-readable description of the failure.
    pub fn user_message(&self) -> String {
        match self {
            LoadImageError::Io { path, source } => match source.kind() {
                io::ErrorKind::NotFound => format!("file not found: {path}"),
                io::ErrorKind::PermissionDenied => format!("permission denied: {path}"),
                _ => format!("{path}: {source}"),
            },
            LoadImageError::TooLarge { path, size, max } => format!(
                "image too large: {path} ({} > {} max)",
                human_size(*size),
                human_size(*max),
            ),
            LoadImageError::Unrecognized { path } => {
                format!("not a recognized image file: {path}")
            }
            LoadImageError::NotAnImage { path, detected } => {
                format!("not an image file: {path} (detected {detected})")
            }
            LoadImageError::UnsupportedFormat { path, mime } => format!(
                "unsupported image format: {path} ({mime}). Supported: {}",
                SUPPORTED_MIMES.join(", ")
            ),
        }
    }
}

impl fmt::Display for LoadImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.user_message())
    }
}

impl std::error::Error for LoadImageError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LoadImageError::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Read `path` and validate it is a supported image no larger than [`MAX_IMAGE_BYTES`]. The size
/// is checked from metadata, before any read.
pub async fn load_image(path: &Path) -> Result<LoadedImage, LoadImageError> {
    let display_path = || path.display().to_string();
    let meta = tokio::fs::metadata(path)
        .await
        .map_err(|source| LoadImageError::Io {
            path: display_path(),
            source,
        })?;
    if !meta.is_file() {
        return Err(LoadImageError::Io {
            path: display_path(),
            source: io::Error::new(
                io::ErrorKind::InvalidInput,
                "not a regular file (device, pipe, or socket)",
            ),
        });
    }
    if meta.len() > MAX_IMAGE_BYTES {
        return Err(LoadImageError::TooLarge {
            path: display_path(),
            size: meta.len(),
            max: MAX_IMAGE_BYTES,
        });
    }
    let bytes = tokio::fs::read(path)
        .await
        .map_err(|source| LoadImageError::Io {
            path: display_path(),
            source,
        })?;
    let Some(detected) = infer::get(&bytes) else {
        return Err(LoadImageError::Unrecognized {
            path: display_path(),
        });
    };
    let mime = detected.mime_type();
    if !infer::is_image(&bytes) {
        return Err(LoadImageError::NotAnImage {
            path: display_path(),
            detected: mime.to_string(),
        });
    }
    if !SUPPORTED_MIMES.contains(&mime) {
        return Err(LoadImageError::UnsupportedFormat {
            path: display_path(),
            mime: mime.to_string(),
        });
    }
    Ok(LoadedImage {
        mime: mime.to_string(),
        bytes,
    })
}

fn human_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    if bytes >= GB {
        format!("{:.1}GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1}MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{}KB", bytes / KB)
    } else {
        format!("{bytes}B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    /// Minimal GIF89a: an image to `infer`, but outside [`SUPPORTED_MIMES`].
    const GIF_BYTES: &[u8] = &[
        0x47, 0x49, 0x46, 0x38, 0x39, 0x61, 0x01, 0x00, 0x01, 0x00, 0x80, 0x00, 0x00, 0xFF, 0xFF,
        0xFF, 0x00, 0x00, 0x00, 0x21, 0xF9, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2C, 0x00, 0x00,
        0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x02, 0x02, 0x44, 0x01, 0x00, 0x3B,
    ];

    #[tokio::test]
    async fn missing_file_is_io_error() {
        let err = load_image(Path::new("/nonexistent/x.png"))
            .await
            .unwrap_err();
        assert!(matches!(err, LoadImageError::Io { .. }), "{err:?}");
        assert_eq!(err.user_message(), "file not found: /nonexistent/x.png");
    }

    #[tokio::test]
    async fn text_file_is_unrecognized() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("notes.txt");
        tokio::fs::write(&path, b"just some text").await.unwrap();
        let err = load_image(&path).await.unwrap_err();
        assert!(
            matches!(err, LoadImageError::Unrecognized { .. }),
            "{err:?}"
        );
        assert_eq!(
            err.user_message(),
            format!("not a recognized image file: {}", path.display())
        );
    }

    #[tokio::test]
    async fn gif_is_rejected_by_allowlist() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("anim.gif");
        tokio::fs::write(&path, GIF_BYTES).await.unwrap();
        let err = load_image(&path).await.unwrap_err();
        match err {
            LoadImageError::UnsupportedFormat { mime, .. } => assert_eq!(mime, "image/gif"),
            other => panic!("expected UnsupportedFormat, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn oversize_file_is_rejected_before_read() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("huge.png");
        let sparse = tokio::fs::File::create(&path).await.unwrap();
        sparse.set_len(MAX_IMAGE_BYTES + 1).await.unwrap();
        drop(sparse);
        let err = load_image(&path).await.unwrap_err();
        match err {
            LoadImageError::TooLarge { size, max, .. } => {
                assert_eq!(size, MAX_IMAGE_BYTES + 1);
                assert_eq!(max, MAX_IMAGE_BYTES);
            }
            other => panic!("expected TooLarge, got {other:?}"),
        }
        assert!(err.user_message().starts_with("image too large:"));
    }
}
