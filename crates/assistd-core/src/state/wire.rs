//! Wire-attachment decoding for the query handler.

use super::DispatchError;
use assistd_tools::Attachment;

/// Returns the first decode error.
pub(super) fn decode_wire_attachments(
    wire: &[assistd_ipc::ImageAttachment],
) -> Result<Vec<Attachment>, DispatchError> {
    wire.iter()
        .map(|w| {
            let bytes = w
                .decode_bytes()
                .map_err(|source| DispatchError::InvalidAttachment {
                    mime: w.mime.clone(),
                    source,
                })?;
            Ok(Attachment::Image {
                mime: w.mime.clone(),
                bytes,
            })
        })
        .collect()
}
