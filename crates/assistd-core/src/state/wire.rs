//! Wire-attachment decoding for the query handler.

use assistd_ipc::ImageAttachment;
use assistd_tools::Attachment;

use super::DispatchError;

/// Decode every attachment, failing on the first that is not valid base64.
pub(super) fn decode_wire_attachments(
    wire: &[ImageAttachment],
) -> Result<Vec<Attachment>, DispatchError> {
    wire.iter()
        .map(|attachment| {
            let bytes =
                attachment
                    .decode_bytes()
                    .map_err(|source| DispatchError::InvalidAttachment {
                        mime: attachment.mime.clone(),
                        source,
                    })?;
            Ok(Attachment::Image {
                mime: attachment.mime.clone(),
                bytes,
            })
        })
        .collect()
}
