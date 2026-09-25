use std::path::Path;

pub use assistd_ipc::attachment::{LoadImageError, MAX_IMAGE_BYTES};
use assistd_ipc::attachment::{LoadedImage, load_image};

use crate::command::Attachment;

/// Read `path` as a supported image, returning the attachment and its size
/// in bytes.
pub async fn load_image_attachment(path: &Path) -> Result<(Attachment, usize), LoadImageError> {
    let LoadedImage { mime, bytes } = load_image(path).await?;
    let size = bytes.len();
    Ok((Attachment::Image { mime, bytes }, size))
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;
    use crate::fixtures::PNG_BYTES;

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
}
