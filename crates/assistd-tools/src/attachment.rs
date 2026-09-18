use std::path::Path;

use crate::command::Attachment;
use crate::commands::cat::human_size;

/// Image MIME types llama.cpp's vision adapters accept. `infer::is_image`
/// also passes GIF, BMP, TIFF and HEIC, so the list is explicit.
const SUPPORTED_MIMES: &[&str] = &["image/png", "image/jpeg", "image/webp"];

/// Upper bound on an attached image. Generous for a 4K PNG screenshot,
/// but small enough to catch a video or RAW file before it is read.
pub const MAX_IMAGE_BYTES: u64 = 32 * 1024 * 1024;

/// Why [`load_image_attachment`] refused a file.
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
    /// An image, but not one of [`SUPPORTED_MIMES`].
    UnsupportedFormat {
        path: String,
        mime: String,
    },
}

impl LoadImageError {
    /// One-line message without any `[error] <cmd>:` prefix.
    pub fn user_message(&self) -> String {
        match self {
            LoadImageError::Io { path, source } => match source.kind() {
                std::io::ErrorKind::NotFound => format!("file not found: {path}"),
                std::io::ErrorKind::PermissionDenied => format!("permission denied: {path}"),
                _ => format!("{path}: {source}"),
            },
            LoadImageError::TooLarge { path, size, max } => format!(
                "image too large: {path} ({} > {} max)",
                human_size(*size as usize),
                human_size(*max as usize),
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

/// Read `path` and validate it is a supported image. Returns the
/// attachment and its size in bytes.
pub async fn load_image_attachment(path: &Path) -> Result<(Attachment, usize), LoadImageError> {
    // Stat first so a huge file is rejected before a buffer is allocated.
    let meta = tokio::fs::metadata(path)
        .await
        .map_err(|e| LoadImageError::Io {
            path: path.display().to_string(),
            source: e,
        })?;
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
    let size = bytes.len();
    Ok((
        Attachment::Image {
            mime: mime.to_string(),
            bytes,
        },
        size,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::PNG_BYTES;
    use tempfile::tempdir;

    // Minimal GIF89a header → infer reports image/gif → must be rejected
    // by the supported-format allowlist.
    const GIF_BYTES: &[u8] = &[
        0x47, 0x49, 0x46, 0x38, 0x39, 0x61, 0x01, 0x00, 0x01, 0x00, 0x80, 0x00, 0x00, 0xFF, 0xFF,
        0xFF, 0x00, 0x00, 0x00, 0x21, 0xF9, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2C, 0x00, 0x00,
        0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x02, 0x02, 0x44, 0x01, 0x00, 0x3B,
    ];

    #[tokio::test]
    async fn loads_png() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("shot.png");
        tokio::fs::write(&path, PNG_BYTES).await.unwrap();
        let (att, size) = load_image_attachment(&path).await.unwrap();
        assert_eq!(size, PNG_BYTES.len());
        match att {
            Attachment::Image { mime, bytes } => {
                assert_eq!(mime, "image/png");
                assert_eq!(bytes, PNG_BYTES);
            }
        }
    }

    #[tokio::test]
    async fn missing_file_is_io_error() {
        let err = load_image_attachment(Path::new("/nonexistent/x.png"))
            .await
            .unwrap_err();
        assert!(matches!(err, LoadImageError::Io { .. }));
        assert!(err.user_message().starts_with("file not found:"));
    }

    #[tokio::test]
    async fn text_file_is_unrecognized() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("notes.txt");
        tokio::fs::write(&path, b"just some text").await.unwrap();
        let err = load_image_attachment(&path).await.unwrap_err();
        assert!(matches!(err, LoadImageError::Unrecognized { .. }));
        assert!(err.user_message().contains("not a recognized image"));
    }

    #[tokio::test]
    async fn gif_is_rejected_by_allowlist() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("anim.gif");
        tokio::fs::write(&path, GIF_BYTES).await.unwrap();
        let err = load_image_attachment(&path).await.unwrap_err();
        match err {
            LoadImageError::UnsupportedFormat { mime, .. } => assert_eq!(mime, "image/gif"),
            other => panic!("expected UnsupportedFormat, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn oversize_file_is_rejected_before_read() {
        // Use a sparse file so the test doesn't actually allocate
        // MAX_IMAGE_BYTES + 1 of disk. set_len + drop is enough; metadata()
        // reports the logical size and load_image_attachment rejects on
        // that, never reaching the read path.
        let dir = tempdir().unwrap();
        let path = dir.path().join("huge.png");
        let f = tokio::fs::File::create(&path).await.unwrap();
        f.set_len(MAX_IMAGE_BYTES + 1).await.unwrap();
        drop(f);
        let err = load_image_attachment(&path).await.unwrap_err();
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
