//! Reading and validating image files before they are attached to a
//! [`Request::Query`](crate::Request::Query).

use std::path::Path;

/// Image MIME types llama.cpp's vision adapters accept. `infer::is_image`
/// also passes GIF, BMP, TIFF and HEIC, so the list is explicit.
const SUPPORTED_MIMES: &[&str] = &["image/png", "image/jpeg", "image/webp"];

/// Upper bound on an attached image. Generous for a 4K PNG screenshot,
/// but small enough to catch a video or RAW file before it is read.
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
        source: std::io::Error,
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
    /// One-line, human-readable description of the failure, with no
    /// prefix.
    pub fn user_message(&self) -> String {
        match self {
            LoadImageError::Io { path, source } => match source.kind() {
                std::io::ErrorKind::NotFound => format!("file not found: {path}"),
                std::io::ErrorKind::PermissionDenied => format!("permission denied: {path}"),
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

impl std::fmt::Display for LoadImageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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

/// Read `path` and validate it is a supported image no larger than
/// [`MAX_IMAGE_BYTES`].
pub async fn load_image(path: &Path) -> Result<LoadedImage, LoadImageError> {
    // Stat first so a huge file is rejected before a buffer is allocated.
    let meta = tokio::fs::metadata(path)
        .await
        .map_err(|e| LoadImageError::Io {
            path: path.display().to_string(),
            source: e,
        })?;
    if !meta.is_file() {
        return Err(LoadImageError::Io {
            path: path.display().to_string(),
            source: std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "not a regular file (device, pipe, or socket)",
            ),
        });
    }
    if meta.len() > MAX_IMAGE_BYTES {
        return Err(LoadImageError::TooLarge {
            path: path.display().to_string(),
            size: meta.len(),
            max: MAX_IMAGE_BYTES,
        });
    }
    let bytes = tokio::fs::read(path)
        .await
        .map_err(|e| LoadImageError::Io {
            path: path.display().to_string(),
            source: e,
        })?;
    let Some(t) = infer::get(&bytes) else {
        return Err(LoadImageError::Unrecognized {
            path: path.display().to_string(),
        });
    };
    if !infer::is_image(&bytes) {
        return Err(LoadImageError::NotAnImage {
            path: path.display().to_string(),
            detected: t.mime_type().to_string(),
        });
    }
    let mime = t.mime_type();
    if !SUPPORTED_MIMES.contains(&mime) {
        return Err(LoadImageError::UnsupportedFormat {
            path: path.display().to_string(),
            mime: mime.to_string(),
        });
    }
    Ok(LoadedImage {
        mime: mime.to_string(),
        bytes,
    })
}

fn human_size(n: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    if n >= GB {
        format!("{:.1}GB", n as f64 / GB as f64)
    } else if n >= MB {
        format!("{:.1}MB", n as f64 / MB as f64)
    } else if n >= KB {
        format!("{}KB", n / KB)
    } else {
        format!("{n}B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    // A minimal GIF89a image: `infer::is_image` accepts it, but it is
    // outside the supported-format allowlist.
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
        // A sparse file reports the oversize logical length without
        // allocating disk, so rejection must come from metadata alone.
        let dir = tempdir().unwrap();
        let path = dir.path().join("huge.png");
        let f = tokio::fs::File::create(&path).await.unwrap();
        f.set_len(MAX_IMAGE_BYTES + 1).await.unwrap();
        drop(f);
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
