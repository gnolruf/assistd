//! Shared image-attachment loading. Used by:
//! - `SeeCommand` (LLM-invoked `see PATH` tool call)
//! - The TUI's `/attach` slash command
//!
//! Both call paths need identical validation so error messages match
//! and the allowlisted MIME types stay in sync.

use std::path::Path;

use crate::command::Attachment;

/// Image MIME types accepted by llama.cpp's vision adapters today.
/// `infer::is_image` is too permissive (accepts GIF, BMP, TIFF, HEIC) so
/// we filter explicitly.
const SUPPORTED_MIMES: &[&str] = &["image/png", "image/jpeg", "image/webp"];

#[derive(Debug)]
pub enum LoadImageError {
    /// File missing, unreadable, or other I/O failure.
    Io {
        path: String,
        source: std::io::Error,
    },
    /// `infer` couldn't identify the file type from its magic bytes.
    Unrecognized { path: String },
    /// `infer` identified a non-image type.
    NotAnImage { path: String, detected: String },
    /// `infer` identified an image, but it's not in [`SUPPORTED_MIMES`].
    UnsupportedFormat { path: String, mime: String },
}

impl LoadImageError {
    /// One-line user-facing message. Caller decorates with a prefix like
    /// `[error] /attach: ` or `[error] see: ` as appropriate.
    pub fn user_message(&self) -> String {
        match self {
            LoadImageError::Io { path, source } => match source.kind() {
                std::io::ErrorKind::NotFound => format!("file not found: {path}"),
                std::io::ErrorKind::PermissionDenied => format!("permission denied: {path}"),
                _ => format!("{path}: {source}"),
            },
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

/// Read `path` from disk and validate it's a supported image format.
///
/// On success returns the [`Attachment`] (ready to hand to the LLM
/// conversation layer) plus the file size in bytes (so the caller can
/// render a "12 KB" annotation without re-stat-ing).
pub async fn load_image_attachment(path: &Path) -> Result<(Attachment, usize), LoadImageError> {
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
    use tempfile::tempdir;

    const PNG_BYTES: &[u8] = &[
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44,
        0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1F,
        0x15, 0xC4, 0x89, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0x00,
        0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00, 0x00, 0x00, 0x00, 0x49,
        0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
    ];

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
}
