//! The `User-Agent` assistd identifies itself with, and reading HTTP
//! response bodies under a size cap.

use reqwest::Response;
use thiserror::Error;

/// `User-Agent` sent on requests to hosts outside the machine.
pub const USER_AGENT: &str = concat!(
    "assistd/",
    env!("CARGO_PKG_VERSION"),
    " (+https://github.com/gnolruf/assistd)"
);

/// Why a response body was not read in full.
#[derive(Debug, Error)]
pub enum BodyError {
    #[error("response body exceeded {max} bytes")]
    TooLarge { max: usize },

    #[error("failed to read response body: {0}")]
    Read(#[source] reqwest::Error),
}

/// Read the whole body of `response`, failing as soon as it is known to
/// exceed `max` bytes.
pub async fn read_capped(mut response: Response, max: usize) -> Result<Vec<u8>, BodyError> {
    if response
        .content_length()
        .is_some_and(|len| usize::try_from(len).map_or(true, |len| len > max))
    {
        return Err(BodyError::TooLarge { max });
    }
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(BodyError::Read)? {
        if body.len() + chunk.len() > max {
            return Err(BodyError::TooLarge { max });
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}
