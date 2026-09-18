//! Wire-attachment decoding for the query handler.

use assistd_tools::Attachment;

/// Returns the first decode error.
pub(super) fn decode_wire_attachments(
    wire: &[assistd_ipc::ImageAttachment],
) -> std::result::Result<Vec<Attachment>, String> {
    wire.iter()
        .map(|w| {
            let bytes = w
                .decode_bytes()
                .map_err(|e| format!("base64 decode failed for {}: {e}", w.mime))?;
            Ok(Attachment::Image {
                mime: w.mime.clone(),
                bytes,
            })
        })
        .collect()
}
