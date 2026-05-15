//! Wire-attachment decoding shared between the query handler and the
//! IPC layer.

use assistd_tools::Attachment;

/// Convert wire-level [`assistd_ipc::ImageAttachment`] entries into the
/// internal [`Attachment`] type the agent loop expects. Returns the first
/// decode error so the caller can surface it cleanly to the client.
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
