//! HTTP client builders for local and remote servers, and reading response
//! bodies under a size cap.

use reqwest::redirect::Policy;
use reqwest::{Client, ClientBuilder, Response};
use thiserror::Error;

const USER_AGENT: &str = concat!(
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

/// A client builder for servers on this machine: no proxy, no compression,
/// and no redirect followed, so a 3xx comes back as the response.
pub fn local_client_builder() -> ClientBuilder {
    Client::builder()
        .no_proxy()
        .no_gzip()
        .no_brotli()
        .redirect(Policy::none())
}

/// A client builder for hosts outside the machine: identifies as assistd and
/// takes bodies as sent, so callers wanting compressed transfer re-enable it.
pub fn remote_client_builder() -> ClientBuilder {
    Client::builder()
        .user_agent(USER_AGENT)
        .no_gzip()
        .no_brotli()
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

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::io;

    use reqwest::StatusCode;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use super::*;

    /// Answers a request for `/redirected` with 200 and any other with a
    /// redirect to it.
    async fn serve_redirects(listener: TcpListener) -> io::Result<Infallible> {
        loop {
            let (mut stream, _) = listener.accept().await?;
            let mut request = [0; 1024];
            let read = stream.read(&mut request).await?;
            let reply: &[u8] = if request[..read].starts_with(b"GET /redirected ") {
                b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            } else {
                b"HTTP/1.1 302 Found\r\nLocation: /redirected\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            };
            stream.write_all(reply).await?;
        }
    }

    #[tokio::test]
    async fn a_local_client_returns_a_redirect_instead_of_following_it() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let url = format!("http://{}/", listener.local_addr().expect("local address"));
        let client = local_client_builder().build().expect("client builds");
        let response = tokio::select! {
            response = client.get(url).send() => response.expect("request is answered"),
            stopped = serve_redirects(listener) => panic!("server stopped: {stopped:?}"),
        };
        assert_eq!(response.status(), StatusCode::FOUND);
    }
}
