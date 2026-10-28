//! Wire-attachment decoding and validation for the query handler.

use thiserror::Error;

use assistd_ipc::attachment::{
    ImageFormatError, MAX_IMAGE_BYTES, MAX_QUERY_IMAGE_BYTES, sniff_image,
};
use assistd_ipc::{ImageAttachment, ImageMime};
use assistd_tools::Attachment;

/// Why an image attachment from an IPC client was refused.
#[derive(Debug, Error)]
pub enum AttachmentError {
    #[error("base64 decode failed for {mime}: {source}")]
    Base64 {
        mime: ImageMime,
        #[source]
        source: base64::DecodeError,
    },

    #[error("{mime} image is {size} bytes; the limit is {MAX_IMAGE_BYTES}")]
    ImageTooLarge { mime: ImageMime, size: u64 },

    #[error("images total more than {MAX_QUERY_IMAGE_BYTES} bytes")]
    QueryTooLarge,

    #[error("declared {mime}: {source}")]
    Format {
        mime: ImageMime,
        #[source]
        source: ImageFormatError,
    },

    #[error("declared {declared} but the content is {detected}")]
    MimeMismatch {
        declared: ImageMime,
        detected: ImageMime,
    },
}

/// Decode and validate every attachment, dropping each base64 string once decoded. Fails on
/// the first that is not valid base64, is not a PNG, JPEG or WebP matching its declared MIME,
/// or breaks a per-image or per-query cap.
pub(super) fn decode_wire_attachments(
    wire: Vec<ImageAttachment>,
) -> Result<Vec<Attachment>, AttachmentError> {
    let mut total_bytes = 0u64;
    wire.into_iter()
        .map(|attachment| {
            let bytes = decode_image(&attachment)?;
            total_bytes += bytes.len() as u64;
            if total_bytes > MAX_QUERY_IMAGE_BYTES {
                return Err(AttachmentError::QueryTooLarge);
            }
            Ok(Attachment::Image {
                mime: attachment.mime,
                bytes,
            })
        })
        .collect()
}

fn decode_image(attachment: &ImageAttachment) -> Result<Vec<u8>, AttachmentError> {
    let mime = attachment.mime;
    let bytes = attachment
        .decode_bytes()
        .map_err(|source| AttachmentError::Base64 { mime, source })?;
    let size = bytes.len() as u64;
    if size > MAX_IMAGE_BYTES {
        return Err(AttachmentError::ImageTooLarge { mime, size });
    }
    let detected =
        sniff_image(&bytes).map_err(|source| AttachmentError::Format { mime, source })?;
    if detected != mime {
        return Err(AttachmentError::MimeMismatch {
            declared: mime,
            detected,
        });
    }
    Ok(bytes)
}
