//! Telling binary bytes from text by their first bytes.

/// How much of a buffer's head [`sniff_binary`] looks at for a NUL byte.
pub const SNIFF_LEN: usize = 8192;

const UNRECOGNISED_BINARY: &str = "application/octet-stream";

/// The MIME type of the non-text format whose magic number `bytes` start with.
pub fn binary_format(bytes: &[u8]) -> Option<&'static str> {
    infer::get(bytes)
        .map(|format| format.mime_type())
        .filter(|mime| !mime.starts_with("text/"))
}

/// `Some(mime)` if the bytes look binary: a [`binary_format`], or
/// `application/octet-stream` for a NUL byte in the first 8 KB (GNU grep's heuristic).
pub fn sniff_binary(bytes: &[u8]) -> Option<&'static str> {
    binary_format(bytes).or_else(|| {
        bytes[..bytes.len().min(SNIFF_LEN)]
            .contains(&0u8)
            .then_some(UNRECOGNISED_BINARY)
    })
}
