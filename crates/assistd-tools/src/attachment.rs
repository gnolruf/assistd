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
