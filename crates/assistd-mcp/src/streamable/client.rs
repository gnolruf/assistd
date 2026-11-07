//! The HTTP client under the Streamable HTTP transport: `reqwest` requests with every response
//! body and SSE event held under a byte cap.

use std::borrow::Cow;
use std::collections::HashMap;
use std::future;
use std::sync::Arc;

use assistd_utils::http::{BodyError, read_capped};
use futures_util::StreamExt;
use futures_util::stream::BoxStream;
use http::header::{ACCEPT, CONTENT_TYPE, WWW_AUTHENTICATE};
use http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use reqwest::{RequestBuilder, Response};
use rmcp::model::{
    ClientJsonRpcMessage, ErrorData, JsonRpcMessage, RequestId, ServerJsonRpcMessage,
};
use rmcp::transport::common::http_header::{
    EVENT_STREAM_MIME_TYPE, HEADER_LAST_EVENT_ID, HEADER_SESSION_ID, JSON_MIME_TYPE,
};
use rmcp::transport::streamable_http_client::{
    AuthRequiredError, HttpStatusError, SseError, StreamableHttpClient, StreamableHttpError,
    StreamableHttpPostResponse,
};
use sse_stream::{Sse, SseStream};
use thiserror::Error;

use crate::session::MAX_MESSAGE_BYTES;

/// Largest error-status body read in search of a JSON-RPC error.
const MAX_ERROR_BODY_BYTES: usize = 64 * 1024;

/// What a POST accepts back: a single JSON reply or an SSE stream of them.
const ACCEPT_REPLIES: &str = "text/event-stream, application/json";

/// The transport error this client produces.
pub(crate) type HttpError = StreamableHttpError<HttpClientError>;

/// A `reqwest` client whose JSON bodies stay under [`MAX_MESSAGE_BYTES`] and whose SSE events
/// stay under the cap the transport passes in.
#[derive(Debug, Clone)]
pub(crate) struct CappedHttpClient {
    http: reqwest::Client,
}

/// Why a request failed below the MCP layer. Never carries the request URL.
#[derive(Debug, Error)]
pub(crate) enum HttpClientError {
    #[error("request failed: {0}")]
    Request(#[source] reqwest::Error),
    #[error(transparent)]
    Body(BodyError),
}

/// Why an SSE response body stopped.
#[derive(Debug, Error)]
enum SseBodyError {
    #[error("failed to read SSE body: {0}")]
    Read(#[source] reqwest::Error),
    #[error("SSE event exceeded {0} bytes")]
    EventTooLarge(usize),
}

/// Tracks the size of the SSE event being received so an oversized one fails the stream
/// instead of being buffered without bound.
#[derive(Debug)]
struct SseEventLimit {
    max_bytes: usize,
    event_bytes: usize,
    line_bytes: usize,
    after_cr: bool,
}

impl CappedHttpClient {
    pub(crate) fn new(http: reqwest::Client) -> Self {
        Self { http }
    }
}

impl StreamableHttpClient for CappedHttpClient {
    type Error = HttpClientError;

    async fn post_message(
        &self,
        uri: Arc<str>,
        message: ClientJsonRpcMessage,
        session_id: Option<Arc<str>>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<StreamableHttpPostResponse, HttpError> {
        self.post_message_with_max_sse_event_size(
            uri,
            message,
            session_id,
            auth_header,
            custom_headers,
            MAX_MESSAGE_BYTES,
        )
        .await
    }

    async fn post_message_with_max_sse_event_size(
        &self,
        uri: Arc<str>,
        message: ClientJsonRpcMessage,
        session_id: Option<Arc<str>>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
        max_sse_event_size: usize,
    ) -> Result<StreamableHttpPostResponse, HttpError> {
        let session_attached = session_id.is_some();
        let request = self.http.post(uri.as_ref()).json(&message);
        let request = with_session_headers(request, custom_headers, session_id, auth_header)
            .header(ACCEPT, ACCEPT_REPLIES);
        let response = request.send().await.map_err(request_error)?;
        reject_unauthorized(&response)?;
        let status = response.status();
        if matches!(status, StatusCode::ACCEPTED | StatusCode::NO_CONTENT) {
            return Ok(StreamableHttpPostResponse::Accepted);
        }
        if status == StatusCode::NOT_FOUND && session_attached {
            return Err(StreamableHttpError::SessionExpired);
        }
        if !status.is_success() {
            return json_rpc_error_reply(response).await;
        }
        success_reply(response, &message, max_sse_event_size).await
    }

    async fn delete_session(
        &self,
        uri: Arc<str>,
        session_id: Arc<str>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<(), HttpError> {
        let request = self.http.delete(uri.as_ref());
        let request = with_session_headers(request, custom_headers, Some(session_id), auth_header);
        let response = request.send().await.map_err(request_error)?;
        match response.status() {
            StatusCode::METHOD_NOT_ALLOWED => Ok(()),
            status if status.is_success() => Ok(()),
            status => Err(status_error(status)),
        }
    }

    async fn get_stream(
        &self,
        uri: Arc<str>,
        session_id: Option<Arc<str>>,
        last_event_id: Option<String>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<BoxStream<'static, Result<Sse, SseError>>, HttpError> {
        self.get_stream_with_max_sse_event_size(
            uri,
            session_id,
            last_event_id,
            auth_header,
            custom_headers,
            MAX_MESSAGE_BYTES,
        )
        .await
    }

    async fn get_stream_with_max_sse_event_size(
        &self,
        uri: Arc<str>,
        session_id: Option<Arc<str>>,
        last_event_id: Option<String>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
        max_sse_event_size: usize,
    ) -> Result<BoxStream<'static, Result<Sse, SseError>>, HttpError> {
        let mut request = self.http.get(uri.as_ref());
        if let Some(last_event_id) = last_event_id {
            request = request.header(HEADER_LAST_EVENT_ID, last_event_id);
        }
        let request = with_session_headers(request, custom_headers, session_id, auth_header)
            .header(ACCEPT, EVENT_STREAM_MIME_TYPE);
        let response = request.send().await.map_err(request_error)?;
        reject_unauthorized(&response)?;
        match response.status() {
            StatusCode::METHOD_NOT_ALLOWED => Err(StreamableHttpError::ServerDoesNotSupportSse),
            status if !status.is_success() => Err(status_error(status)),
            _ if has_content_type(&response, EVENT_STREAM_MIME_TYPE) => {
                Ok(capped_sse(response, max_sse_event_size, None))
            }
            _ => Err(StreamableHttpError::UnexpectedContentType(content_type(
                &response,
            ))),
        }
    }
}

impl SseEventLimit {
    fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            event_bytes: 0,
            line_bytes: 0,
            after_cr: false,
        }
    }

    /// Count `chunk` toward the current event; errs once the event passes the cap.
    fn admit(&mut self, chunk: &[u8]) -> Result<(), SseBodyError> {
        for &byte in chunk {
            self.observe(byte);
            if self.event_bytes > self.max_bytes {
                return Err(SseBodyError::EventTooLarge(self.max_bytes));
            }
        }
        Ok(())
    }

    /// Advance over one byte; a blank line (after `\n`, `\r` or `\r\n`) ends the event.
    fn observe(&mut self, byte: u8) {
        let after_cr = std::mem::replace(&mut self.after_cr, byte == b'\r');
        match byte {
            b'\n' if after_cr => {}
            b'\r' | b'\n' => {
                if self.line_bytes == 0 {
                    self.event_bytes = 0;
                }
                self.line_bytes = 0;
            }
            _ => {
                self.line_bytes += 1;
                self.event_bytes += 1;
            }
        }
    }
}

/// Add the configured headers, then the session id and bearer token the transport manages.
fn with_session_headers(
    request: RequestBuilder,
    custom_headers: HashMap<HeaderName, HeaderValue>,
    session_id: Option<Arc<str>>,
    auth_header: Option<String>,
) -> RequestBuilder {
    let mut request = request.headers(custom_headers.into_iter().collect::<HeaderMap>());
    if let Some(session_id) = session_id {
        request = request.header(HEADER_SESSION_ID, session_id.as_ref());
    }
    if let Some(token) = auth_header {
        request = request.bearer_auth(token);
    }
    request
}

/// Fail a 401 or 403 as needing authorization, whether or not it carries a challenge.
fn reject_unauthorized(response: &Response) -> Result<(), HttpError> {
    if !matches!(
        response.status(),
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN
    ) {
        return Ok(());
    }
    let challenge = response
        .headers()
        .get(WWW_AUTHENTICATE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default();
    Err(StreamableHttpError::AuthRequired(AuthRequiredError::new(
        challenge.to_owned(),
    )))
}

/// The reply to a successful POST: an SSE stream, a JSON message, or accepted with no reply.
async fn success_reply(
    response: Response,
    message: &ClientJsonRpcMessage,
    max_sse_event_size: usize,
) -> Result<StreamableHttpPostResponse, HttpError> {
    let session_id = response
        .headers()
        .get(HEADER_SESSION_ID)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let request_id = match message {
        JsonRpcMessage::Request(request) => Some(request.id.clone()),
        _ => None,
    };
    let expects_reply = request_id.is_some();
    if has_content_type(&response, EVENT_STREAM_MIME_TYPE) {
        let stream = capped_sse(response, max_sse_event_size, request_id);
        return Ok(StreamableHttpPostResponse::Sse(stream, session_id));
    }
    if !has_content_type(&response, JSON_MIME_TYPE) {
        return match (expects_reply, response.content_length()) {
            (false, Some(0)) => Ok(StreamableHttpPostResponse::Accepted),
            _ => Err(StreamableHttpError::UnexpectedContentType(content_type(
                &response,
            ))),
        };
    }
    let body = read_body(response).await?;
    match serde_json::from_slice::<ServerJsonRpcMessage>(&body) {
        Ok(reply) => Ok(StreamableHttpPostResponse::Json(reply, session_id)),
        Err(_) if !expects_reply => Ok(StreamableHttpPostResponse::Accepted),
        Err(e) => Err(StreamableHttpError::UnexpectedServerResponse(Cow::Owned(
            format!("reply is not a JSON-RPC message: {e}"),
        ))),
    }
}

/// The JSON-RPC error an error status carries, or the status itself when it carries none.
async fn json_rpc_error_reply(response: Response) -> Result<StreamableHttpPostResponse, HttpError> {
    let status = response.status();
    if !has_content_type(&response, JSON_MIME_TYPE) {
        return Err(status_error(status));
    }
    let body = read_capped(response, MAX_ERROR_BODY_BYTES).await;
    match body.map(|body| serde_json::from_slice::<ServerJsonRpcMessage>(&body)) {
        Ok(Ok(error @ JsonRpcMessage::Error(_))) => {
            Ok(StreamableHttpPostResponse::Json(error, None))
        }
        _ => Err(status_error(status)),
    }
}

/// `response`'s SSE events, failing the stream once one passes `max_event_bytes`. On the
/// stream answering `request_id`, an oversized event instead ends it with an error reply to
/// that request, which the transport would otherwise retry by reconnecting.
fn capped_sse(
    response: Response,
    max_event_bytes: usize,
    request_id: Option<RequestId>,
) -> BoxStream<'static, Result<Sse, SseError>> {
    let mut limit = SseEventLimit::new(max_event_bytes);
    let chunks = response.bytes_stream().map(move |chunk| {
        let chunk = chunk.map_err(|e| SseBodyError::Read(e.without_url()))?;
        limit.admit(&chunk)?;
        Ok::<_, SseBodyError>(chunk)
    });
    let events = SseStream::from_bytes_stream(chunks);
    let Some(request_id) = request_id else {
        return events.boxed();
    };
    events
        .scan(false, move |replied, event| {
            if *replied {
                return future::ready(None);
            }
            let event = event.or_else(|error| {
                let reply = oversized_reply(&error, &request_id).ok_or(error)?;
                *replied = true;
                Ok(reply)
            });
            future::ready(Some(event))
        })
        .boxed()
}

/// An SSE event carrying a JSON-RPC error for `request_id`, when `error` is an oversized event.
fn oversized_reply(error: &SseError, request_id: &RequestId) -> Option<Sse> {
    let SseError::Body(source) = error else {
        return None;
    };
    let too_large @ SseBodyError::EventTooLarge(_) = source.downcast_ref::<SseBodyError>()? else {
        return None;
    };
    let reply = ServerJsonRpcMessage::error(
        ErrorData::internal_error(format!("server reply rejected: {too_large}"), None),
        Some(request_id.clone()),
    );
    Some(Sse {
        event: None,
        data: serde_json::to_string(&reply).ok(),
        id: None,
        retry: None,
    })
}

async fn read_body(response: Response) -> Result<Vec<u8>, HttpError> {
    read_capped(response, MAX_MESSAGE_BYTES).await.map_err(|e| {
        let e = match e {
            BodyError::Read(source) => BodyError::Read(source.without_url()),
            too_large @ BodyError::TooLarge { .. } => too_large,
        };
        StreamableHttpError::Client(HttpClientError::Body(e))
    })
}

fn has_content_type(response: &Response, mime: &str) -> bool {
    response
        .headers()
        .get(CONTENT_TYPE)
        .is_some_and(|value| value.as_bytes().starts_with(mime.as_bytes()))
}

fn content_type(response: &Response) -> Option<String> {
    response
        .headers()
        .get(CONTENT_TYPE)
        .map(|value| String::from_utf8_lossy(value.as_bytes()).into_owned())
}

fn status_error(status: StatusCode) -> HttpError {
    StreamableHttpError::HttpStatus(HttpStatusError::new(status.as_u16(), None))
}

fn request_error(error: reqwest::Error) -> HttpError {
    StreamableHttpError::Client(HttpClientError::Request(error.without_url()))
}
