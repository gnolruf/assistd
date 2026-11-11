//! The image formats and size limits of a [`Request::Query`](crate::Request::Query)'s
//! attachments.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Largest accepted image: room for a 4K PNG screenshot, small enough to reject video or RAW.
pub const MAX_IMAGE_BYTES: u64 = 32 * 1024 * 1024;

/// Largest combined decoded size of every image on one query, matching the socket's query frame.
pub const MAX_QUERY_IMAGE_BYTES: u64 = MAX_IMAGE_BYTES;

/// An image format llama.cpp's vision adapters accept, serialized as its MIME type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ImageMime {
    #[serde(rename = "image/png")]
    Png,
    #[serde(rename = "image/jpeg")]
    Jpeg,
    #[serde(rename = "image/webp")]
    Webp,
}

impl ImageMime {
    const ALL: [ImageMime; 3] = [ImageMime::Png, ImageMime::Jpeg, ImageMime::Webp];

    /// The MIME type, as in `image/png`.
    pub const fn as_str(self) -> &'static str {
        match self {
            ImageMime::Png => "image/png",
            ImageMime::Jpeg => "image/jpeg",
            ImageMime::Webp => "image/webp",
        }
    }

    /// The format whose MIME type is `mime`, or `None` for any other type.
    pub fn from_mime(mime: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|format| format.as_str() == mime)
    }

    /// Every accepted MIME type, comma-separated.
    pub fn supported_list() -> String {
        Self::ALL.map(Self::as_str).join(", ")
    }
}

impl fmt::Display for ImageMime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
