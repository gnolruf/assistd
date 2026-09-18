//! JSON-RPC 2.0 frames and the request/response [`Correlator`] shared
//! by both transports.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::oneshot;

use crate::error::McpError;

/// Outbound JSON-RPC 2.0 request frame.
#[derive(Debug, Serialize)]
pub struct Request<'a> {
    pub jsonrpc: &'a str,
    pub id: u64,
    pub method: &'a str,
    pub params: Value,
}

/// Outbound JSON-RPC 2.0 notification frame (no `id`, no response).
#[derive(Debug, Serialize)]
pub struct Notification<'a> {
    pub jsonrpc: &'a str,
    pub method: &'a str,
    pub params: Value,
}

/// Inbound JSON-RPC 2.0 response. Either `result` or `error` is set;
/// notifications from the server have no `id`.
#[derive(Debug, Deserialize)]
pub struct Response {
    pub id: Option<u64>,
    pub result: Option<Value>,
    pub error: Option<RpcError>,
}

/// JSON-RPC 2.0 error object carried in a [`Response`] when the server reports failure.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RpcError {
    pub code: i64,
    pub message: String,
    #[serde(default)]
    pub data: Option<Value>,
}

/// Hard cap on outstanding requests; prevents a misbehaving server
/// from leaking memory by never replying.
pub const MAX_IN_FLIGHT: usize = 256;

/// Outcome of one JSON-RPC round trip.
pub type Reply = Result<Value, RpcError>;

/// Matches outbound JSON-RPC request ids to their waiting [`oneshot`] receivers.
pub struct Correlator {
    next_id: AtomicU64,
    pending: Mutex<HashMap<u64, oneshot::Sender<Reply>>>,
}

impl Correlator {
    pub fn new() -> Self {
        Self {
            next_id: AtomicU64::new(1),
            pending: Mutex::new(HashMap::new()),
        }
    }

    /// Reserve an id and a receiver for its reply. Errors with
    /// `TooManyInFlight` once [`MAX_IN_FLIGHT`] requests are pending.
    pub fn next_request(&self, method: &'static str, params: Value) -> Result<Pending, McpError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        {
            let mut guard = self.pending.lock();
            if guard.len() >= MAX_IN_FLIGHT {
                return Err(McpError::TooManyInFlight);
            }
            guard.insert(id, tx);
        }
        Ok(Pending {
            id,
            method,
            params,
            rx,
        })
    }

    /// Wake the caller waiting on `response.id`. Unknown ids are logged
    /// and dropped; they occur when a reply lands after a reconnect.
    pub fn deliver(&self, response: Response) {
        let Some(id) = response.id else {
            return;
        };
        let tx = {
            let mut guard = self.pending.lock();
            guard.remove(&id)
        };
        let Some(tx) = tx else {
            tracing::warn!(target: "assistd::mcp", id, "received response for unknown request id");
            return;
        };
        let reply = match (response.result, response.error) {
            (_, Some(err)) => Err(err),
            (Some(value), None) => Ok(value),
            (None, None) => Ok(Value::Null),
        };
        let _ = tx.send(reply);
    }

    /// Wake every pending caller with the error `err_factory` builds.
    pub fn fail_all(&self, err_factory: impl Fn() -> RpcError) {
        let drained: Vec<_> = {
            let mut guard = self.pending.lock();
            guard.drain().collect()
        };
        for (_, tx) in drained {
            let _ = tx.send(Err(err_factory()));
        }
    }

    pub fn in_flight(&self) -> usize {
        self.pending.lock().len()
    }
}

impl Default for Correlator {
    fn default() -> Self {
        Self::new()
    }
}

/// A reserved request: frame it, send it, then await `rx`.
#[derive(Debug)]
pub struct Pending {
    pub id: u64,
    method: &'static str,
    params: Value,
    pub rx: oneshot::Receiver<Reply>,
}

impl Pending {
    /// [`Self::frame_json`] plus a trailing newline.
    pub fn frame_line(&self) -> Result<Vec<u8>, McpError> {
        let mut bytes = self.frame_json()?;
        bytes.push(b'\n');
        Ok(bytes)
    }

    pub fn frame_json(&self) -> Result<Vec<u8>, McpError> {
        let req = Request {
            jsonrpc: "2.0",
            id: self.id,
            method: self.method,
            params: self.params.clone(),
        };
        Ok(serde_json::to_vec(&req)?)
    }
}

/// Encode a notification as a single newline-terminated line.
pub fn notification_line(method: &'static str, params: Value) -> Result<Vec<u8>, McpError> {
    let n = Notification {
        jsonrpc: "2.0",
        method,
        params,
    };
    let mut bytes = serde_json::to_vec(&n)?;
    bytes.push(b'\n');
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn closed_err() -> RpcError {
        RpcError {
            code: -1,
            message: "transport closed".into(),
            data: None,
        }
    }

    #[tokio::test]
    async fn round_trip_request_response() {
        let c = Correlator::new();
        let pending = c.next_request("ping", json!({})).unwrap();
        let id = pending.id;

        c.deliver(Response {
            id: Some(id),
            result: Some(json!({"ok": true})),
            error: None,
        });

        let value = pending.rx.await.unwrap().unwrap();
        assert_eq!(value, json!({"ok": true}));
    }

    #[tokio::test]
    async fn rpc_error_surfaces() {
        let c = Correlator::new();
        let pending = c.next_request("bad", json!({})).unwrap();
        c.deliver(Response {
            id: Some(pending.id),
            result: None,
            error: Some(RpcError {
                code: -32601,
                message: "method not found".into(),
                data: None,
            }),
        });
        let err = pending.rx.await.unwrap().unwrap_err();
        assert_eq!(err.code, -32601);
    }

    #[tokio::test]
    async fn unknown_id_does_not_panic() {
        let c = Correlator::new();
        c.deliver(Response {
            id: Some(999),
            result: Some(Value::Null),
            error: None,
        });
        // No panic, no waiter; pending stays empty.
        assert_eq!(c.in_flight(), 0);
    }

    #[tokio::test]
    async fn fail_all_wakes_pending() {
        let c = Correlator::new();
        let p1 = c.next_request("a", json!({})).unwrap();
        let p2 = c.next_request("b", json!({})).unwrap();
        assert_eq!(c.in_flight(), 2);

        c.fail_all(closed_err);
        assert_eq!(c.in_flight(), 0);

        assert!(p1.rx.await.unwrap().is_err());
        assert!(p2.rx.await.unwrap().is_err());
    }

    #[tokio::test]
    async fn rejects_when_in_flight_cap_reached() {
        let c = Correlator::new();
        let mut keep = Vec::new();
        for _ in 0..MAX_IN_FLIGHT {
            keep.push(c.next_request("x", json!({})).unwrap());
        }
        match c.next_request("y", json!({})) {
            Err(McpError::TooManyInFlight) => {}
            other => panic!("expected TooManyInFlight, got {other:?}"),
        }
    }

    #[test]
    fn frame_line_encodes_newline_terminated_json() {
        let c = Correlator::new();
        let pending = c.next_request("ping", json!({"echo": 1})).unwrap();
        let line = pending.frame_line().unwrap();
        assert!(line.ends_with(b"\n"));
        let parsed: serde_json::Value = serde_json::from_slice(&line[..line.len() - 1]).unwrap();
        assert_eq!(parsed["jsonrpc"], "2.0");
        assert_eq!(parsed["method"], "ping");
        assert_eq!(parsed["params"], json!({"echo": 1}));
        assert!(parsed["id"].is_number());
    }

    #[test]
    fn notification_has_no_id() {
        let line = notification_line("notifications/initialized", json!({})).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&line[..line.len() - 1]).unwrap();
        assert_eq!(parsed["jsonrpc"], "2.0");
        assert_eq!(parsed["method"], "notifications/initialized");
        assert!(parsed.get("id").is_none());
    }
}
